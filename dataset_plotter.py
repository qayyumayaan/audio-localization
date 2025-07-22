import numpy as np
import matplotlib.pyplot as plt

# NOTE: this code doesn't exactly call the dataset_creator logic in the exact same way, but it is very related. This makes a useful figure. 

def triangle_coords_centered(sl):
    """Return coordinates for an equilateral triangle centered at the origin with side length sl."""
    h = sl * np.sqrt(3) / 2
    M0 = np.array([-sl / 2, -h / 3])
    M1 = np.array([ sl / 2, -h / 3])
    M2 = np.array([  0.0,    2 * h / 3])
    return np.vstack([M0, M1, M2, M0]), [M0, M1, M2]

def sample_source_positions(num_points=1000, r_min=2.0, r_max=5.0):
    """Sample points uniformly in polar coordinates between -30° and 90°."""
    angles_deg = np.random.uniform(-30, 90, size=num_points)
    angles_rad = np.deg2rad(angles_deg)
    radii = np.random.uniform(r_min, r_max, size=num_points)
    x = radii * np.cos(angles_rad)
    y = radii * np.sin(angles_rad)
    return np.stack((x, y), axis=1)

def plot_mic_array_and_sources(triangle, mics, sources):
    plt.figure(figsize=(10, 8))

    # Plot microphone triangle
    plt.plot(triangle[:, 0], triangle[:, 1], 'r-', linewidth=2, label="Microphone Array")
    for i, mic in enumerate(mics):
        plt.plot(mic[0], mic[1], 'ro')
        plt.text(mic[0], mic[1], f'M{i}', fontsize=12, color='red', ha='right')

    # Plot sources
    plt.scatter(sources[:, 0], sources[:, 1], s=10, alpha=0.5, label="Source Points")
    for i in range(len(sources)):
        if i < 10 or i >= 990:
            x, y = sources[i]
            plt.text(x, y, f'B{i}', fontsize=6, color='blue')

    # Plot settings
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.title("Microphone Arrays and Source Positions (Centered Geometry)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    side_length = 0.045  # You can change this as needed
    triangle, mic_positions = triangle_coords_centered(side_length)
    source_positions = sample_source_positions(num_points=1000, r_min=1, r_max=5.0)
    plot_mic_array_and_sources(triangle, mic_positions, source_positions)
