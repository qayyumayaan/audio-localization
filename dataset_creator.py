import numpy as np
import pandas as pd
import config

def triangle_coords_centered(sl):
    """Equilateral triangle (M0, M1, M2) centered at origin with side length sl."""
    h = sl * np.sqrt(3) / 2
    M0 = np.array([-sl / 2, -h / 3])
    M1 = np.array([ sl / 2, -h / 3])
    M2 = np.array([  0.0,    2 * h / 3])
    return np.vstack([M0, M1, M2]), [M0, M1, M2]

def sample_source_positions(num_points, r_min, r_max, angle_bounds=(-30, 90)):
    """Uniformly sample 2D points in a sector (degrees) at radii [r_min, r_max]."""
    angles = np.deg2rad(np.random.uniform(angle_bounds[0], angle_bounds[1], size=num_points))
    radii  = np.random.uniform(r_min, r_max, size=num_points)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack((x, y), axis=1)

def generate_synthetic_dataset(
    num_samples   = config.NUM_SAMPLES,
    side_length   = config.SIDE_LENGTH,
    r_min         = config.PING_RADIUS_MIN,
    r_max         = config.PING_RADIUS_MAX,
    speed_of_sound= config.SPEED_OF_SOUND,
    noise_std     = config.NOISE_STD,
    fs            = config.SAMPLE_RATE_FS
):
    """
    Generate synthetic single-ping TDOA training data.
    Simulates quantized TDOA measurement as it would appear at a given sampling rate.
    """
    # Mic geometry
    _, mics = triangle_coords_centered(side_length)
    mics = np.array(mics)

    # Source positions
    sources = sample_source_positions(num_samples, r_min, r_max)

    # True TDOAs
    dists       = np.linalg.norm(sources[:, None, :] - mics[None, :, :], axis=2)
    tau21_true  = (dists[:,1] - dists[:,0]) / speed_of_sound
    tau31_true  = (dists[:,2] - dists[:,0]) / speed_of_sound

    # Use the true (float) delays directly
    tau21 = tau21_true.copy()
    tau31 = tau31_true.copy()

    # Optional noise
    if noise_std > 0:
        tau21 += np.random.normal(0, noise_std, size=num_samples)
        tau31 += np.random.normal(0, noise_std, size=num_samples)

    # Filter out anything outside physically possible window
    max_tau    = r_max / speed_of_sound
    valid_mask = (np.abs(tau21) <= max_tau) & (np.abs(tau31) <= max_tau)

    df = pd.DataFrame({
        'tau21': tau21[valid_mask],
        'tau31': tau31[valid_mask],
        'x':     sources[valid_mask, 0],
        'y':     sources[valid_mask, 1]
    })

    return df

if __name__ == '__main__':
    df = generate_synthetic_dataset()
    np.savez_compressed(
        f'synthetic_ping_dataset_fs{config.SAMPLE_RATE_FS}.npz',
        tau21=df['tau21'].to_numpy(dtype='float64'),
        tau31=df['tau31'].to_numpy(dtype='float64'),
        x=df['x'].to_numpy(dtype='float64'),
        y=df['y'].to_numpy(dtype='float64')
    )
    print("Synthetic dataset created. Shape:", df.shape)
