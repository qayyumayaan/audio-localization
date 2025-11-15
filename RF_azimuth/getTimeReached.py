# Experimental setup: phone 1 at (0,0), phone 2 at (x, 0), phone 3 in between
# First 5: 1 meter apart
# Second 5: 0.5 meters
# Third 5: 0.1 meters
# (6,0), (6,3), (9,3), (9,8), (4,8)

import librosa
import numpy as np

audio_path = "recording.wav"

# Load audio
y, sr = librosa.load(audio_path)

# Compute amplitude envelope (absolute value + smoothing)
frame_length = 2048
hop_length = 512
amplitude = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

# Estimate noise level = median of first 0.5 seconds
noise_frames = int(0.5 * sr / hop_length)
noise_level = np.median(amplitude[:noise_frames])

# Scream threshold (tweak: 3× noise)
threshold = noise_level * 3

# Find first frame above threshold
frames_above = np.where(amplitude > threshold)[0]
if len(frames_above) == 0:
    print("No scream detected!")
else:
    first_frame = frames_above[0]
    scream_time = first_frame * hop_length / sr
    print(f"Scream first heard at: {scream_time:.3f} seconds")
