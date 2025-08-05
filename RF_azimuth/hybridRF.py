import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from audioSim import Mic, Environment, Wave, getEstTDOA
import matplotlib.pyplot as plt
from triangulateSim import estimate_azimuth

# --- Helper Plotting Functions ---
def plot_epoch_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.semilogy(train_losses, label='Train Loss')
    plt.semilogy(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training vs Validation Loss (log scale)')
    plt.legend()
    plt.grid(True, which='both')
    plt.show()

def plot_angle_error_cdf(error_dict):
    plt.figure(figsize=(8, 5))
    for label, errors in error_dict.items():
        sorted_err = np.sort(errors)
        cdf = np.arange(len(errors)) / len(errors)
        plt.plot(sorted_err, cdf, label=label)
    plt.xlim([0, 10])  # Crop x-axis to 10 degrees
    plt.xlabel('Angular Error (°)')
    plt.ylabel('CDF')
    plt.title('CDF of Angular Errors (0–10°)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_angle_error_pdf(error_dict, bins=100):
    plt.figure(figsize=(8, 5))
    for label, errors in error_dict.items():
        plt.hist(errors, bins=bins, density=True, alpha=0.5, label=label, histtype='stepfilled')
    plt.xlim([0, 10])  # Crop to 10 degrees
    plt.xlabel('Angular Error (°)')
    plt.ylabel('Density')
    plt.title('PDF of Angular Errors (0–10°)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_boxplots(error_dict):
    plt.figure(figsize=(8, 5))
    data = list(error_dict.values())
    labels = list(error_dict.keys())
    plt.boxplot(data, labels=labels)
    plt.ylabel('Angular Error (°)')
    plt.title('Angular Error Distribution')
    plt.grid(True)
    plt.show()

def plot_ablation_study(maes, labels):
    plt.figure(figsize=(8, 5))
    plt.bar(labels, maes, color='skyblue')
    plt.ylabel('Mean Absolute Error (°)')
    plt.title('Ablation Study Results')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


# ----------- Existing AzimuthMLP -----------
class AzimuthMLP(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[64, 256, 512, 1024]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], 2)  # Output: sin and cos
        )

    def forward(self, x):
        return self.net(x)

# ----------- Fourier Feature MLP -----------
class FourierFeatureMLP(nn.Module):
    def __init__(self, input_size=2, mapping_size=256, scale=10, hidden_sizes=[256, 256, 256]):
        super().__init__()
        # Create Gaussian random matrix for Fourier features
        self.B = nn.Parameter(torch.randn(input_size, mapping_size) * scale, requires_grad=False)
        layers = []
        in_size = mapping_size * 2
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 2))  # Output sin-cos
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, input_size]
        x_proj = 2.0 * np.pi * x @ self.B  # [batch, mapping_size]
        fourier_features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(fourier_features)

# ----------- SIREN -----------
class Sine(nn.Module):
    def forward(self, input):
        return torch.sin(input)

class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SIREN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, hidden_layers=3, output_dim=2, omega_0=30):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(SIRENLayer(input_dim, hidden_dim, is_first=True, omega_0=omega_0))
        for _ in range(hidden_layers - 1):
            self.net.append(SIRENLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        self.final_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return self.final_linear(x)

# ----------- AzimuthRandomForest (unchanged) -----------
class AzimuthRandomForest:
    def __init__(self, n_estimators=50, max_depth=None, random_state=42, bins=24):
        myArray = []
        i = -180
        while i < 180:
            i += 360 / bins
            myArray.append(i)
        self.azimuth_bins = np.array(myArray)
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
    
    def generate_training_data(self, n_samples=10000, max_radius=100):
        print(f"Generating {n_samples} training samples...")
        mics = [Mic((0, 0), 10000), Mic((0.05, 0), 10000), Mic((0.025, 0.0433), 10000)]
        X = []
        y_bins = []
        y_angles = []
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
        X = np.array(X)
        y_bins = np.array(y_bins)
        y_angles = np.array(y_angles)
        print(f"Training data generated: {X.shape[0]} samples")
        return X, y_bins, y_angles
    
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
        accuracy = accuracy_score(y_test, y_pred)

        # Compute MAE in degrees for classification by converting bins to bin center azimuths
        y_pred_azimuths = self.azimuth_bins[y_pred]
        y_true_azimuths = self.azimuth_bins[y_test]
        def angular_diff(a, b):
            diff = np.abs(a - b) % 360
            return np.minimum(diff, 360 - diff)
        mae = np.mean(angular_diff(y_pred_azimuths, y_true_azimuths))

        print(f"RF Test Accuracy: {accuracy:.3f}")
        print(f"RF Mean Absolute Error (degrees): {mae:.2f}")

        print("\nClassification Report:")
        target_names = [f"Bin_{i}: {self.azimuth_bins[i]:.0f}°" for i in range(len(self.azimuth_bins))]
        labels = list(range(len(self.azimuth_bins)))
        print(classification_report(y_test, y_pred, target_names=target_names, labels=labels))

        return X_train_scaled, X_test_scaled, y_train, y_test, y_pred, mae

# ----------- Helpers for sin-cos encoding/decoding -----------
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

# ----------- Training function for MLP, Fourier MLP, SIREN -----------
def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
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

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_model_state)
    model.eval()
    return model, train_losses, val_losses

# ----------- Feature extraction from MLP penultimate layer -----------
def extract_mlp_features(model, X):
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # Forward pass through first layers only (assumes your MLP has net as Sequential)
        h1 = model.net[0](X_tensor)
        h1 = model.net[1](h1)  # ReLU
        h2 = model.net[2](h1)
        h2 = model.net[3](h2)  # ReLU
        return h2.numpy()

# ----------- Main -----------
def main():
    rf_azimuth = AzimuthRandomForest(n_estimators=500, max_depth=80, random_state=42, bins=96)

    # Generate data
    X_raw, y_bins, y_angles = rf_azimuth.generate_training_data(n_samples=25000, max_radius=100)

    # Train/test split
    X_train_raw, X_test_raw, y_train_bins, y_test_bins, y_train_angles, y_test_angles = train_test_split(
        X_raw, y_bins, y_angles, test_size=0.2, random_state=42, stratify=y_bins
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # --- Baseline RF ---
    rf_azimuth.rf_model.fit(X_train, y_train_bins)
    rf_azimuth.is_trained = True
    y_pred_rf = rf_azimuth.rf_model.predict(X_test)
    y_pred_az_rf = rf_azimuth.azimuth_bins[y_pred_rf]
    y_true_az_rf = rf_azimuth.azimuth_bins[y_test_bins]
    mae_rf = np.mean(angular_diff(y_pred_az_rf, y_true_az_rf))

    # --- MLP ---
    y_train_sincos = azimuth_to_sincos(y_train_angles)
    y_test_sincos = azimuth_to_sincos(y_test_angles)
    mlp_model, mlp_train_losses, mlp_val_losses = train_model(AzimuthMLP(), X_train, y_train_sincos, X_test, y_test_sincos, epochs=200)
    plot_epoch_loss(mlp_train_losses, mlp_val_losses)
    mlp_preds_angles = sincos_to_azimuth(mlp_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy())
    mae_mlp = np.mean(angular_diff(mlp_preds_angles, y_test_angles))

    # --- Fourier MLP ---
    ff_model, ff_train_losses, ff_val_losses = train_model(FourierFeatureMLP(), X_train, y_train_sincos, X_test, y_test_sincos, epochs=200)
    plot_epoch_loss(ff_train_losses, ff_val_losses)
    ff_preds_angles = sincos_to_azimuth(ff_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy())
    mae_ff = np.mean(angular_diff(ff_preds_angles, y_test_angles))

    # --- SIREN ---
    siren_model, siren_train_losses, siren_val_losses = train_model(SIREN(), X_train, y_train_sincos, X_test, y_test_sincos, epochs=8)
    plot_epoch_loss(siren_train_losses, siren_val_losses)
    siren_preds_angles = sincos_to_azimuth(siren_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy())
    mae_siren = np.mean(angular_diff(siren_preds_angles, y_test_angles))

    # --- Hybrid RF on MLP features ---
    mlp_feat_train = extract_mlp_features(mlp_model, X_train)
    mlp_feat_test = extract_mlp_features(mlp_model, X_test)
    rf_on_mlp = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
    rf_on_mlp.fit(mlp_feat_train, y_train_angles)
    preds_rf_on_mlp = rf_on_mlp.predict(mlp_feat_test)
    mae_rf_on_mlp = mean_absolute_error(y_test_angles, preds_rf_on_mlp)

    # --- Vanilla Algorithm Predictions ---
    mic_positions = [[0, 0], [0.05, 0], [0.025, 0.0433]]
    vanilla_preds = []
    for row in X_test_raw:
        tdoas = [row[0] * 1e-6, row[1] * 1e-6]  # convert µs to sec
        vanilla_angle = estimate_azimuth(tdoas, mic_positions)
        vanilla_preds.append(vanilla_angle)
    vanilla_preds = np.array(vanilla_preds)
    mae_vanilla = np.mean(angular_diff(vanilla_preds, y_test_angles))

    # --- Evaluation ---
    error_dict = {
        "Vanilla": angular_diff(vanilla_preds, y_test_angles),
        "RF": angular_diff(y_pred_az_rf, y_true_az_rf),
        "MLP": angular_diff(mlp_preds_angles, y_test_angles),
        "Fourier": angular_diff(ff_preds_angles, y_test_angles),
        "SIREN": angular_diff(siren_preds_angles, y_test_angles),
        "Hybrid RF-MLP": angular_diff(preds_rf_on_mlp, y_test_angles),
    }

    maes = [mae_vanilla, mae_rf, mae_mlp, mae_ff, mae_siren, mae_rf_on_mlp]
    labels = list(error_dict.keys())

    plot_angle_error_cdf(error_dict)
    plot_angle_error_pdf(error_dict)
    plot_error_boxplots(error_dict)
    plot_ablation_study(maes, labels)

    return {
        "rf_classifier": rf_azimuth,
        "mlp_model": mlp_model,
        "fourier_mlp": ff_model,
        "siren_model": siren_model,
        "rf_on_mlp": rf_on_mlp,
        "vanilla_mae": mae_vanilla
    }


if __name__ == "__main__":
    main()
