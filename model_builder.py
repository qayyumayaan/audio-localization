import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib
import config

def train_and_save_model(
    data_path: str = f'synthetic_ping_dataset_fs{config.SAMPLE_RATE_FS}.npz',
    model_path: str = f'ping_localization_model_fs{config.SAMPLE_RATE_FS}.pkl',
    test_size: float = 0.2,
    random_state: int = 42
):
    # 1) Load data
    data = np.load(data_path)
    df = pd.DataFrame({
        'tau21': data['tau21'],
        'tau31': data['tau31'],
        'x': data['x'],
        'y': data['y']
    })
    
    X = df[['tau21', 'tau31']].values
    y = df[['x', 'y']].values

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("Building pipeline")
    # 3) Build pipeline: scaler + MLP regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=random_state
        ))
    ])

    # 4) Train
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # 5) Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test set MSE: {mse:.6f} (m^2)")

    # 6) Save pipeline
    joblib.dump(pipeline, model_path)
    print(f"Trained model saved to: {model_path}")

train_and_save_model()