import numpy as np
import joblib
import torch
import torch.nn as nn
import config

from scipy.signal import chirp
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────────────────

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
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.drop4(torch.relu(self.fc4(x)))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        return self.out(x)


def triangle_coords_centered(sl):
    h = sl * np.sqrt(3) / 2
    M0 = np.array([-sl / 2, -h / 3])
    M1 = np.array([ sl / 2, -h / 3])
    M2 = np.array([  0.0,    2 * h / 3])
    return np.vstack([M0, M1, M2, M0]), [M0, M1, M2]

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / np.abs(R), n=interp * n)
    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift / float(interp * fs)

def posError(pos, d21, d31, mic_positions, max_r=1000.0):
    x, y = pos
    if x*x + y*y > max_r**2:
        return 1e6 + (x*x+y*y)
    d1 = np.hypot(x - mic_positions[0][0], y - mic_positions[0][1])
    d2 = np.hypot(x - mic_positions[1][0], y - mic_positions[1][1])
    d3 = np.hypot(x - mic_positions[2][0], y - mic_positions[2][1])
    return (d2 - d1 - d21)**2 + (d3 - d1 - d31)**2

def triangulatePosition(audio, mic_positions, max_tau=None):
    v_sound = config.SPEED_OF_SOUND
    (fs1, sig1), (fs2, sig2), (fs3, sig3) = audio
    tau21 = gcc_phat(sig2, sig1, fs=fs1, max_tau=max_tau)
    tau31 = gcc_phat(sig3, sig1, fs=fs1, max_tau=max_tau)
    d21 = tau21 * v_sound
    d31 = tau31 * v_sound
    res = minimize(
        posError,
        x0=[0, 0],
        args=(d21, d31, mic_positions, 1000.0),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol':1e-8, 'fatol':1e-8}
    )
    if not res.success:
        raise RuntimeError("Triangulation failed")
    return np.array(res.x)

def sample_source_positions(num_points, r_min, r_max, angle_bounds=(-30, 90)):
    angles = np.deg2rad(np.random.uniform(angle_bounds[0], angle_bounds[1], size=num_points))
    radii  = np.random.uniform(r_min, r_max, size=num_points)
    x = radii * np.cos(angles)
    y = radii * np.sin(radii)
    return np.stack((x, y), axis=1)


def run_benchmark(n_trials=1000):
    fs = config.SAMPLE_RATE_FS
    v_sound = config.SPEED_OF_SOUND
    side_length = config.SIDE_LENGTH
    r_min, r_max = config.PING_RADIUS_MIN, config.PING_RADIUS_MAX
    max_tau = r_max / v_sound
    acc_radius = config.ACCURACY_RADIUS

    # Load PyTorch model and scaler
    model_data = joblib.load(f'ping_localization_model_d{int(side_length*1000)}mm_fs{fs}.pkl')
    model = DeepTDOARegressor(input_dim=2)
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    scaler = model_data['scaler']

    # Mic setup
    _, mics = triangle_coords_centered(side_length)
    mic_positions = [m.tolist() for m in mics]

    # Signal setup
    duration  = 0.05
    t         = np.linspace(0, duration, int(fs * duration), endpoint=False)
    frequency = 300
    signal    = np.sin(2 * np.pi * frequency * t)

    acc2, acc4 = 0, 0

    for i in range(n_trials):
        B = sample_source_positions(1, r_min, r_max)[0]
        distances     = [np.linalg.norm(B - m) for m in mics]
        time_delays   = [d / v_sound for d in distances]
        sample_delays = [int(round(td * fs)) for td in time_delays]
        max_delay     = max(sample_delays)
        buf_len       = max_delay + len(signal)

        audio = []
        for sd in sample_delays:
            buf = np.zeros(buf_len)
            buf[sd:sd+len(signal)] = signal
            audio.append((fs, buf))

        try:
            tau21 = gcc_phat(audio[1][1], audio[0][1], fs, max_tau=max_tau)
            tau31 = gcc_phat(audio[2][1], audio[0][1], fs, max_tau=max_tau)
            x_input = np.array([[tau21, tau31]], dtype=np.float32)
            x_scaled = scaler.transform(x_input)
            with torch.no_grad():
                pred = model(torch.from_numpy(x_scaled)).numpy()[0]
            err = np.linalg.norm(pred - B)
            if err <= acc_radius:
                acc2 += 1
            if err <= 2 * acc_radius:
                acc4 += 1
        except Exception as e:
            print(f"[{i}] Error: {e}")
            continue

    print(f"\nTested {n_trials} examples")
    print(f"Within ±{acc_radius} m: {acc2} ({100*acc2/n_trials:.2f}%)")
    print(f"Within ±{2*acc_radius} m: {acc4} ({100*acc4/n_trials:.2f}%)")

if __name__ == '__main__':
    run_benchmark()
