import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.neural_network   import MLPRegressor
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import mean_squared_error, r2_score
import joblib
import config

def train_and_save_model(
    data_path  = f'synthetic_ping_dataset_fs{config.SAMPLE_RATE_FS}.npz',
    model_path = f'ping_localization_model_d{int(config.SIDE_LENGTH*1000)}mm_fs{config.SAMPLE_RATE_FS}.pkl',
    test_size=0.2, random_state=42
):
    # 1) Load data
    data = np.load(data_path)
    df = pd.DataFrame({
        'tau21': data['tau21'],
        'tau31': data['tau31'],
        'x': data['x'],
        'y': data['y']
    })

    X = df[['tau21','tau31']].values
    y = df[['x','y']].values

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("Building pipeline")
    # 3) Build pipeline: scaler + MLP regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp',    MLPRegressor(
            hidden_layer_sizes=(128,128),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=random_state
        ))
    ])

    # 3) Train
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # 4) Evaluate
    def report(name, X_split, y_true):
        y_pred  = pipeline.predict(X_split)
        mse     = mean_squared_error(y_true, y_pred)
        rmse    = np.sqrt(mse)
        r2      = r2_score(y_true, y_pred)
        # fraction within the accuracy radius
        errs    = np.linalg.norm(y_pred - y_true, axis=1)
        acc_pct = np.mean(errs <= config.ACCURACY_RADIUS) * 100

        print(f"{name} set:")
        print(f"  MSE:   {mse:.6f} m^2")
        print(f"  RMSE:  {rmse:.6f} m")
        print(f"  R²:    {r2:.6f}")
        print(f"  Within ±{config.ACCURACY_RADIUS} m: {acc_pct:.2f}%\n")

    print("Evaluation:")
    report("Training", X_train, y_train)
    report("Test",     X_test,  y_test)

    # 5) Save
    joblib.dump(pipeline, model_path)
    print("Trained model saved to:", model_path)

if __name__ == '__main__':
    train_and_save_model()
