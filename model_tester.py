import numpy as np
import joblib
from scipy.optimize import minimize
from scipy.signal import chirp
import config
import math

# --- Required functions ----------------------------------------------------

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
    # Penalty if outside circle of radius max_r
    if x*x + y*y > max_r**2:
        return 1e6 + (x*x+y*y)   # large penalty
    d1 = np.hypot(x - mic_positions[0][0], y - mic_positions[0][1])
    d2 = np.hypot(x - mic_positions[1][0], y - mic_positions[1][1])
    d3 = np.hypot(x - mic_positions[2][0], y - mic_positions[2][1])
    return (d2 - d1 - d21)**2 + (d3 - d1 - d31)**2

def triangulatePosition(audio, mic_positions, max_tau=None):
    """TDOA-based triangulation with a hard 1000m radius limit."""
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
        method='Nelder-Mead',       # still Nelder-Mead, but with penalty
        options={'maxiter': 10000, 'xatol':1e-8, 'fatol':1e-8}
    )
    if not res.success:
        raise RuntimeError("Triangulation failed")
    return np.array(res.x)


def sample_source_positions(num_points, r_min, r_max, angle_bounds=(-30, 90)):
    angles = np.deg2rad(np.random.uniform(angle_bounds[0], angle_bounds[1], size=num_points))
    radii  = np.random.uniform(r_min, r_max, size=num_points)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack((x, y), axis=1)

# --- Single-point demonstration --------------------------------------------

# Parameters
side_length = config.SIDE_LENGTH
fs = config.SAMPLE_RATE_FS
r_min, r_max = config.PING_RADIUS_MIN, config.PING_RADIUS_MAX
v_sound = config.SPEED_OF_SOUND
max_tau = r_max / v_sound  # ~0.0146 s

# 1) Source
B = sample_source_positions(1, r_min, r_max)[0]

# 2) Microphones
_, mics = triangle_coords_centered(side_length)
mic_positions = [m.tolist() for m in mics]

# 3) Wideband chirp ping
duration = 0.05
t = np.linspace(0, duration, int(fs*duration), endpoint=False)
signal = chirp(t, f0=500, f1=5000, t1=duration, method='linear')

# 4) Simulate delays
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

# 5) Raw triangulation (bounded search)
raw_est = triangulatePosition(audio, mic_positions, max_tau=max_tau)

# 6) TDOAs for ML input
# --- Print input to ML model (raw signal pairs before GCC-PHAT) ---
print("Input to ML model (pre-GCC PHAT):")

signals = [audio[0][1], audio[1][1], audio[2][1]]

# Find the first nonzero index in any of the three signals
first_nonzero_index = min(np.argmax(sig != 0) for sig in signals)

# Define how many samples to show
num_samples_to_show = 10
end_index = first_nonzero_index + num_samples_to_show

# Print the relevant slices
for i, sig in enumerate(signals, 1):
    print(f"Signal {i} [{first_nonzero_index}:{end_index}]: {sig[first_nonzero_index:end_index]}")

tau21 = gcc_phat(audio[1][1], audio[0][1], fs, max_tau=max_tau)
tau31 = gcc_phat(audio[2][1], audio[0][1], fs, max_tau=max_tau)

print(f"TDOA from Mic 2 to 1: {tau21} \nTDOA from Mic 3 to 1: {tau31} \n Max tau (longest time delay expected between any two mics) {max_tau}")

# 7) ML-corrected prediction
model = joblib.load(f'ping_localization_model_fs{fs}.pkl')
ml_est = model.predict([[tau21, tau31]])[0]

# 8) Errors
err_raw = np.linalg.norm(raw_est - B)
err_ml  = np.linalg.norm(ml_est  - B)

# --- Angle errors ---
def compute_angle_error(est, true_pos):
    angle_est  = np.arctan2(est[1], est[0])
    angle_true = np.arctan2(true_pos[1], true_pos[0])
    angle_err_deg = np.rad2deg(angle_est - angle_true)
    # Normalize to [-180, 180]
    angle_err_deg = (angle_err_deg + 180) % 360 - 180
    return abs(angle_err_deg)

angle_err_raw = compute_angle_error(raw_est, B)
angle_err_ml  = compute_angle_error(ml_est, B)

# 9) Report
print(f"True position:             [{B[0]:.3f}, {B[1]:.3f}]")
print(f"Raw estimate:              [{raw_est[0]:.3f}, {raw_est[1]:.3f}], error = {err_raw:.3f} m, angle error = {angle_err_raw:.2f}°")
print(f"ML-corrected estimate:     [{ml_est[0]:.3f}, {ml_est[1]:.3f}], error = {err_ml:.3f} m, angle error = {angle_err_ml:.2f}°")
error_reduction = 100 * (err_raw - err_ml) / err_raw
print(f"Percent error reduction:   {error_reduction:.2f}%")