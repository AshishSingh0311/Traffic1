import pandas as pd
import numpy as np
import json
import os
from ml_model import TrafficMLModel

DATA_PATH = os.path.join(os.path.dirname(__file__), 'attached_assets', 'traffic_weather_speed_dataset_1754508149737.csv')


def simple_preprocess(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)

    # Use pandas factorize to avoid extra sklearn dependency during import
    df['location_encoded'] = pd.factorize(df['location'])[0]
    df['weather_encoded'] = pd.factorize(df['weather'])[0]

    feature_columns = ['vehicle_count', 'location_encoded', 'weather_encoded', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour']
    X = df[feature_columns]
    y = df['average_speed_kmph']
    return X, y


if __name__ == '__main__':
    print('Starting run_model.py')
    if not os.path.exists(DATA_PATH):
        print(f'Data file not found at {DATA_PATH}')
        raise SystemExit(1)

    df = pd.read_csv(DATA_PATH)
    print(f'Loaded dataset with {len(df)} records')

    X, y = simple_preprocess(df)
    print(f'Prepared features: X shape={X.shape}, y shape={y.shape}')

    model = TrafficMLModel()

    selected_models = ['Random Forest', 'XGBoost', 'LSTM']
    print('Training models:', selected_models)

    results = model.train_models(X, y, selected_models, test_size=0.2, random_state=42)

    # Make results JSON-serializable
    def make_serializable(obj):
        # Handle pandas Series
        if isinstance(obj, (pd.Series,)):
            return obj.tolist()
        # Numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Numpy scalar types
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        # Pandas/Numpy dtypes
        try:
            # Attempt to convert to Python native
            if hasattr(obj, 'tolist'):
                return obj.tolist()
        except Exception:
            pass
        # Fallback to string
        return str(obj)

    serializable = {}
    for name, res in results.items():
        serializable[name] = {}
        if isinstance(res, dict):
            for k, v in res.items():
                if k == 'model':
                    serializable[name][k] = None
                else:
                    serializable[name][k] = make_serializable(v)
        else:
            serializable[name] = make_serializable(res)

    out_path = os.path.join(os.path.dirname(__file__), 'model_results.json')
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print('Training completed. Results saved to', out_path)
    # Print a short summary
    for name, res in results.items():
        if isinstance(res, dict):
            print(f"\nModel: {name}")
            print('  type:', res.get('type'))
            print('  mse:', res.get('mse'))
            print('  r2:', res.get('r2'))
            print('  accuracy:', res.get('accuracy'))
    print('\nDone.')
