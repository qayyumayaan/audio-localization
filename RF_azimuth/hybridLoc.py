import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from audioSim import Mic, Environment, Wave, getEstTDOA

# ----------- Two separate MLPs -----------

class AzimuthMLP(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[64, 256, 512, 1024]):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[0]),
            nn.ReLU()
        )
        self.output = nn.Linear(hidden_sizes[0], 2)  # sin, cos output

    def forward(self, x):
        x = self.shared(x)
        return self.output(x)

class DistanceMLP(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[64, 256, 512, 1024]):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[0]),
            nn.ReLU()
        )
        self.output = nn.Linear(hidden_sizes[0], 1)  # distance output

    def forward(self, x):
        x = self.shared(x)
        return self.output(x)

# ----------- AzimuthRandomForest (unchanged) -----------
class AzimuthRandomForest:
    def __init__(self, n_estimators=50, max_depth=None, random_state=42, bins=24):
        self.azimuth_bins = np.linspace(-180, 180, bins + 1)
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _digitize_azimuth(self, true_azimuth_deg):
        wrapped_azimuth = ((true_azimuth_deg + 180) % 360) - 180
        bin_idx = np.digitize(wrapped_azimuth, self.azimuth_bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(self.azimuth_bins) - 1)
        return bin_idx

    def generate_training_data(self, n_samples=25000, max_radius=100):
        print(f"Generating {n_samples} training samples...")
        mics = [Mic((0, 0), 48000), Mic((1, 0), 48000), Mic((0.5, 0.866), 48000)]
        X = []
        y_bins = []
        y_angles = []
        y_distances = []
        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"Generated {i}/{n_samples} samples...")
            r = max_radius * np.sqrt(np.random.uniform(0, 1))
            theta_true = np.random.uniform(-180, 180)
            x_pos = r * np.cos(np.radians(theta_true))
            y_pos = r * np.sin(np.radians(theta_true))
            env = Environment(mics, Wave((x_pos, y_pos)))
            tdoa1 = getEstTDOA(mics[0], mics[1], env.getWave()) * 1e6
            tdoa2 = getEstTDOA(mics[0], mics[2], env.getWave()) * 1e6
            az_bin = self._digitize_azimuth(theta_true)
            X.append([tdoa1, tdoa2])
            y_bins.append(az_bin)
            y_angles.append(theta_true)
            y_distances.append(r)
        return np.array(X), np.array(y_bins), np.array(y_angles), np.array(y_distances)

    def train(self, X, y_bins):
        print("Training Random Forest model...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_bins, test_size=0.2, random_state=42, stratify=y_bins
        )
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.rf_model.fit(X_train_scaled, y_train)
        self.is_trained = True
        y_pred = self.rf_model.predict(X_test_scaled)

        y_pred_azimuths = self.azimuth_bins[y_pred]
        y_true_azimuths = self.azimuth_bins[y_test]
        mae = np.mean(angular_diff(y_pred_azimuths, y_true_azimuths))
        acc = accuracy_score(y_test, y_pred)

        print(f"RF Accuracy: {acc:.3f}")
        print(f"RF MAE: {mae:.2f}°")
        return X_train_scaled, X_test_scaled, y_train, y_test, y_pred, mae

# ----------- Helper Functions -----------
def azimuth_to_sincos(azimuth_deg):
    radians = np.radians(azimuth_deg)
    return np.column_stack((np.sin(radians), np.cos(radians)))

def sincos_to_azimuth(sincos):
    sin, cos = sincos[:, 0], sincos[:, 1]
    angles_rad = np.arctan2(sin, cos)
    angles_deg = np.degrees(angles_rad)
    return angles_deg

def angular_diff(a, b):
    diff = np.abs(a - b) % 360
    return np.minimum(diff, 360 - diff)

# ----------- Train MLP -----------
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        epoch_loss = 0

        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= X_train_tensor.size(0)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_model_state)
    model.eval()
    return model

# ----------- Main -----------
def main():
    rf = AzimuthRandomForest(n_estimators=500, max_depth=80, bins=48)
    X_raw, y_bins, y_angles, y_dists = rf.generate_training_data(n_samples=25000)

    X_train_raw, X_test_raw, y_train_bins, y_test_bins, y_train_angles, y_test_angles, y_train_dists, y_test_dists = train_test_split(
        X_raw, y_bins, y_angles, y_dists, test_size=0.2, random_state=42, stratify=y_bins
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    rf.rf_model.fit(X_train, y_train_bins)
    y_pred_bins = rf.rf_model.predict(X_test)
    pred_azimuths = rf.azimuth_bins[y_pred_bins]
    true_azimuths = rf.azimuth_bins[y_test_bins]
    rf_mae = np.mean(angular_diff(pred_azimuths, true_azimuths))
    print(f"\nRF Accuracy: {accuracy_score(y_test_bins, y_pred_bins):.3f}, MAE: {rf_mae:.2f}°")

    # Prepare separate targets
    y_train_sincos = azimuth_to_sincos(y_train_angles)
    y_test_sincos = azimuth_to_sincos(y_test_angles)

    # Train Azimuth MLP
    az_model = AzimuthMLP()
    az_model = train_model(az_model, X_train, y_train_sincos, X_test, y_test_sincos)

    # Train Distance MLP
    # dist_model = DistanceMLP()
    # dist_model = train_model(dist_model, X_train, y_train_dists.reshape(-1, 1), X_test, y_test_dists.reshape(-1, 1))

    # Evaluate Azimuth MLP
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        az_pred_sincos = az_model(X_test_tensor).numpy()
    pred_angles = sincos_to_azimuth(az_pred_sincos)
    mae_angle = np.mean(angular_diff(pred_angles, y_test_angles))

    dist_scaler = StandardScaler()
    y_train_dists_scaled = dist_scaler.fit_transform(y_train_dists.reshape(-1,1))
    y_test_dists_scaled = dist_scaler.transform(y_test_dists.reshape(-1,1))
    # Train distance MLP on scaled target:
    dist_model = DistanceMLP()
    dist_model = train_model(dist_model, X_train, y_train_dists_scaled, X_test, y_test_dists_scaled)
    # When evaluating:
    with torch.no_grad():
        dist_pred_scaled = dist_model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
    # Inverse transform predictions back to meters:
    dist_pred = dist_scaler.inverse_transform(dist_pred_scaled.reshape(-1,1)).flatten()
    mae_dist = np.mean(np.abs(dist_pred - y_test_dists))

    # # Evaluate Distance MLP
    # with torch.no_grad():
    #     dist_pred = dist_model(X_test_tensor).numpy().flatten()
    # mae_dist = np.mean(np.abs(dist_pred - y_test_dists))

    print(f"\nMLP Separate: Azimuth MAE = {mae_angle:.2f}°, Distance MAE = {mae_dist:.2f} m")
    torch.save(az_model.state_dict(), "testModels/mlp48kHz_azimuth.pt")
    torch.save(dist_model.state_dict(), "testModels/mlp48kHz_distance.pt")

if __name__ == "__main__":
    main()
