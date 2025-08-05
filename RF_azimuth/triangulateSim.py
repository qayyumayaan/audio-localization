import numpy as np
from scipy.optimize import minimize, differential_evolution
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from audioSim import Mic, Environment, Wave, getTrueTDOA, getRandomEnv, getEstTDOA

# class DistancePredictionNet(nn.Module):
#     """Small Neural Network for Distance Prediction from Azimuth and TDOAs"""
#     def __init__(self, input_size=3, dropout_rate=0.1):
#         super().__init__()
        
#         # Small architecture - only 3 hidden layers
#         self.network = nn.Sequential(
#             # Input layer
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
            
#             # Hidden layer 1
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
            
#             # Hidden layer 2
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
            
#             # Output layer
#             nn.Linear(64, 1),
#             nn.ReLU()  # Ensure positive distances
#         )
        
#         # Initialize weights
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 nn.init.zeros_(module.bias)
    
#     def forward(self, x):
#         return self.network(x)

def load_distance_model(model_path='small_distance_model.pth'):
    """Load the trained distance prediction model"""
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Using fallback distance estimation.")
        return None
    
    try:
        model = DistancePredictionNet()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading distance model: {e}")
        return None

def predict_distance(model, azimuth, tdoa1, tdoa2):
    """Predict distance using the neural network"""
    if model is None:
        # Improved fallback: better distance estimation based on TDOA magnitudes
        # Use the larger TDOA magnitude as it's more indicative of distance
        max_tdoa = max(abs(tdoa1), abs(tdoa2))
        
        # Convert TDOA to distance difference, then estimate source distance
        # Typical relationship: larger TDOA = source is further from mic array center
        v_sound = 343  # m/s
        distance_diff = abs(max_tdoa * v_sound)
        
        # Heuristic: source distance is proportional to TDOA magnitude
        # Scale factor based on typical microphone separations (~5cm)
        mic_separation = 0.05  # meters
        
        # Better heuristic based on geometry
        if distance_diff > 0:
            # Estimate distance using geometric relationship
            estimated_distance = distance_diff / (mic_separation / 50.0)  # Scale factor
            estimated_distance = np.clip(estimated_distance, 1.0, 150.0)  # Reasonable bounds
        else:
            estimated_distance = 25.0  # Default moderate distance
            
        return estimated_distance
    
    # Scale inputs the same way as training
    azimuth_wrapped = (azimuth + 180) % 360 - 180  # Wrap to [-180, 180]
    tdoa1_scaled = tdoa1 * 1000000  # Convert to microseconds
    tdoa2_scaled = tdoa2 * 1000000  # Convert to microseconds
    
    inputs = [azimuth_wrapped, tdoa1_scaled, tdoa2_scaled]
    input_tensor = torch.tensor([inputs], dtype=torch.float32)
    
    with torch.no_grad():
        predicted_distance = model(input_tensor).item()
    
    return predicted_distance

def posError(pos, d_2diff1, d_3diff1, mic_positions):
    x, y = pos
    d1 = np.sqrt((x - mic_positions[0][0])**2 + (y - mic_positions[0][1])**2)
    d2 = np.sqrt((x - mic_positions[1][0])**2 + (y - mic_positions[1][1])**2)
    d3 = np.sqrt((x - mic_positions[2][0])**2 + (y - mic_positions[2][1])**2)
    return (d2 - d1 - d_2diff1)**2 + (d3 - d1 - d_3diff1)**2

def triangulateSim(TDOAs, mic_positions, temp=68):
    """
    Robust triangulation with multiple optimization strategies
    Mic positions gives a list of lists with the coordinates of each microphone in meters from the center of the drone
    TDOAs should be in seconds (typically very small values like 1e-4)
    temp is the temperature in degrees Fahrenheit
    """
    v_sound = 331 + 0.6 * (5/9 * (temp - 32)) # in m/s

    tdoa_2diff1 = TDOAs[0]
    tdoa_3diff1 = TDOAs[1]

    d_2diff1 = tdoa_2diff1 * v_sound
    d_3diff1 = tdoa_3diff1 * v_sound
    
    # Strategy 1: Multiple initial guesses with Nelder-Mead
    initial_guesses = [
        [0, 0],           # Center
        [1, 0], [-1, 0],  # Left/right
        [0, 1], [0, -1],  # Up/down
        [1, 1], [-1, -1], [1, -1], [-1, 1],  # Diagonals
        [0.1, 0.1], [0.5, 0.5], [2, 2]  # Various distances
    ]
    
    best_result = None
    best_error = float('inf')
    
    for guess in initial_guesses:
        try:
            result = minimize(
                posError,
                x0=guess,
                args=(d_2diff1, d_3diff1, mic_positions),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun
                
        except Exception:
            continue
    
    # Strategy 2: Try L-BFGS-B with bounds
    try:
        bounds = [(-10, 10), (-10, 10)]  # Reasonable bounds for position
        for guess in initial_guesses[:5]:  # Try fewer guesses for bounded method
            result = minimize(
                posError,
                x0=guess,
                args=(d_2diff1, d_3diff1, mic_positions),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun
    except Exception:
        pass
    
    # Strategy 3: Differential Evolution (global optimizer)
    try:
        bounds = [(-10, 10), (-10, 10)]
        result = differential_evolution(
            posError,
            bounds,
            args=(d_2diff1, d_3diff1, mic_positions),
            maxiter=300,
            popsize=15,
            tol=1e-10,
            seed=42
        )
        
        if result.success and result.fun < best_error:
            best_result = result
            best_error = result.fun
    except Exception:
        pass
    
    # Check if we found a good solution
    if best_result is None:
        raise ValueError("All optimization methods failed")
    
    x, y = best_result.x
    x = float(x)
    y = float(y)

    return (x, y)

import numpy as np

def tdoa_linear_solver(TDOAs, mic_positions, temp=68):
    v_sound = 331 + 0.6 * (5/9 * (temp - 32)) # in m/s

    tdoa21 = TDOAs[0]
    tdoa31 = TDOAs[1]

    d21 = v_sound * tdoa21
    d31 = v_sound * tdoa31

    x1, y1 = mic_positions[0]
    x2, y2 = mic_positions[1]
    x3, y3 = mic_positions[2]

    # Construct matrix A and vector b
    A = np.array([
        [2*(x2 - x1), 2*(y2 - y1)],
        [2*(x3 - x1), 2*(y3 - y1)]
    ])
    b = np.array([
        d21**2 - (x2**2 + y2**2) + (x1**2 + y1**2),
        d31**2 - (x3**2 + y3**2) + (x1**2 + y1**2)
    ])

    # Solve the linear system
    pos = np.linalg.lstsq(A, b, rcond=None)[0]
    return pos

def estimate_azimuth(TDOAs, mic_positions, temp = 68):
    # mic_positions: list of microphone positions [(x1, y1), (x2, y2), (x3, y3)]
    # tdoas: list of TDOAs relative to mic 1 [Δt_21, Δt_31]
    v_sound = 331 + 0.6 * (5/9 * (temp - 32)) # in m/s

    ref = mic_positions[0]
    diffs = [np.array(m) - ref for m in mic_positions[1:]]
    dists = np.array([v_sound * t for t in TDOAs])  # distance differences

    A = np.vstack(diffs)
    v, _, _, _ = np.linalg.lstsq(A, dists, rcond=None)

    # Normalize direction vector
    v /= np.linalg.norm(v)
    azimuth = np.arctan2(v[1], v[0])
    return -180 + np.degrees(azimuth)  # in degrees

import numpy as np
from scipy.optimize import minimize_scalar



if __name__ == "__main__":
    
    mics = [Mic((0, 0), 10000), Mic((0.05, 0), 10000), Mic((0.025, 0.0433), 10000)]
    mic_positions = [[0, 0], [0.05, 0], [0.025, 0.0433]]

    for i in range(10):
        env = getRandomEnv(mics, 1)

        tdoa21 = getEstTDOA(mics[0], mics[1], env.getWave())
        tdoa31 = getEstTDOA(mics[0], mics[2], env.getWave())

        if abs(tdoa21) < 1e-6 and abs(tdoa31) < 1e-6:
            print("Threshold reached")
        print(f"True pos: ({env.getWave().pos[0]:.2f}, {env.getWave().pos[1]:.2f})")
        print()