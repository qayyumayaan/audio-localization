# model_builder.py

import numpy as np
import pandas as pd
import joblib
import config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ─── Model Definition ──────────────────────────────────────────────────────

class DeepTDOARegressor(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.Linear(256, 256)
        self.drop4 = nn.Dropout(0.3)
        
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop4(F.relu(self.fc4(x)))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return self.out(x)

# ─── Training & Evaluation ─────────────────────────────────────────────────

def train_and_save_model(
    data_path   = f'synthetic_ping_dataset_fs{config.SAMPLE_RATE_FS}.npz',
    scaler_path = f'scaler_fs{config.SAMPLE_RATE_FS}.pkl',
    model_path  = f'ping_localization_model_d{int(config.SIDE_LENGTH*1000)}mm_fs{config.SAMPLE_RATE_FS}.pt',
    test_size   = 0.2,
    val_size    = 0.2,
    random_state= 42,
    batch_size  = 64,
    max_epochs  = 200,
    patience    = 15,
    lr          = 1e-3
):
    # 1) Load & split data
    D = np.load(data_path)
    df = pd.DataFrame({
        'tau21': D['tau21'],
        'tau31': D['tau31'],
        'x':     D['x'],
        'y':     D['y']
    })
    X = df[['tau21','tau31']].values.astype(np.float32)
    y = df[['x','y']].values.astype(np.float32)
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )

    # 2) Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # 3) DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    
    # 4) Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = DeepTDOARegressor(input_dim=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5) Early stopping setup
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None

    # 6) Training loop
    for epoch in range(1, max_epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch:03d}: Val Loss = {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement for {patience} epochs, stopping early.")
                break

    # Restore best weights
    model.load_state_dict(best_state)

    # 7) Evaluation function
    def report(name, X_split, y_true):
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(X_split).to(device)).cpu().numpy()
        mse  = mean_squared_error(y_true, preds)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_true, preds)
        errs = np.linalg.norm(preds - y_true, axis=1)
        acc  = np.mean(errs <= config.ACCURACY_RADIUS) * 100

        print(f"{name} set:")
        print(f"  MSE:   {mse:.6f} m²")
        print(f"  RMSE:  {rmse:.6f} m")
        print(f"  R²:    {r2:.6f}")
        print(f"  Within ±{config.ACCURACY_RADIUS} m: {acc:.2f}%\n")

    print("\n--- Final Metrics ---")
    report("Training", X_train, y_train)
    report("Test",     X_test,  y_test)

    # 8) Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to: {model_path}")

    # Save PyTorch model and scaler together as .pkl (backward compatibility)
    combo = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler
    }
    pkl_path = model_path.replace('.pt', '.pkl')
    joblib.dump(combo, pkl_path)
    print(f"PyTorch model + scaler saved to: {pkl_path}")


if __name__ == '__main__':
    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    train_and_save_model()
