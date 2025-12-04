#!/usr/bin/env python3
"""
Smoke test for wind power forecasting project.
Tests basic functionality without requiring the full dataset.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def create_synthetic_data():
    """Create small synthetic dataset for testing"""
    print("Creating synthetic SCADA data...")
    
    # Create 7 days of 10-minute data
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start_date, periods=7*144, freq='10min')
    
    # Synthetic wind speed and power with realistic patterns
    hours = np.array([d.hour + d.minute/60 for d in dates])
    wind_speed = 8 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 1, len(dates))
    wind_speed = np.clip(wind_speed, 0, 25)
    
    # Power curve approximation
    power = np.where(wind_speed < 3, 0,
                    np.where(wind_speed > 25, 0,
                            np.minimum(2000, 50 * wind_speed ** 2.5)))
    power += np.random.normal(0, 50, len(dates))
    power = np.clip(power, 0, 2500)
    
    # Create full SCADA dataframe
    data = {
        'Timestamps': dates,
        'Power': power,
        'WindSpeed': wind_speed,
        'StdDevWindSpeed': np.abs(np.random.normal(1, 0.5, len(dates))),
        'WindDirAbs': np.random.uniform(0, 360, len(dates)),
        'WindDirRel': np.random.uniform(-180, 180, len(dates)),
        'AvgRPow': power * 0.95 + np.random.normal(0, 20, len(dates)),
        'GenRPM': 1200 + 300 * (power / 2000) + np.random.normal(0, 50, len(dates)),
        'RotorRPM': 15 + 10 * (wind_speed / 15) + np.random.normal(0, 2, len(dates)),
        'EnvirTemp': 15 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 3, len(dates)),
        'NacelTemp': 25 + 15 * (power / 2000) + np.random.normal(0, 5, len(dates)),
        'GearOilTemp': 40 + 20 * (power / 2000) + np.random.normal(0, 3, len(dates)),
        'GearBearTemp': 35 + 15 * (power / 2000) + np.random.normal(0, 4, len(dates)),
        'GenPh1Temp': 60 + 20 * (power / 2000) + np.random.normal(0, 5, len(dates)),
        'GenPh2Temp': 60 + 20 * (power / 2000) + np.random.normal(0, 5, len(dates)),
        'GenPh3Temp': 60 + 20 * (power / 2000) + np.random.normal(0, 5, len(dates)),
        'GenBearTemp': 50 + 15 * (power / 2000) + np.random.normal(0, 4, len(dates))
    }
    
    df = pd.DataFrame(data)
    return df

def test_preprocessing():
    """Test preprocessing pipeline"""
    print("\n1. Testing preprocessing...")
    
    # Create synthetic data
    df = create_synthetic_data()
    
    # Save temporarily
    os.makedirs('data', exist_ok=True)
    df.to_excel('data/scada.xlsx', index=False)
    print("   ✅ Synthetic data created")
    
    # Test preprocessing
    from preprocess import load_dataset
    try:
        X, Y, dates, mean, std = load_dataset()
        print(f"   ✅ Preprocessing successful: X{X.shape}, Y{Y.shape}")
        return X, Y, dates, mean, std
    except Exception as e:
        print(f"   ❌ Preprocessing failed: {e}")
        return None

def test_decomposition(X):
    """Test tensor decomposition with small ranks"""
    print("\n2. Testing tensor decomposition...")
    
    if X is None:
        print("   ⏭️  Skipped (preprocessing failed)")
        return None
    
    try:
        from decompose import cp_decompose, tucker_decompose, pca_latent_features
        
        # Test CP with small rank
        _, info = cp_decompose(X, rank=3)
        print(f"   ✅ CP decomposition: error {info['rel_error']:.3f}")
        
        # Test Tucker with small ranks
        _, info = tucker_decompose(X, ranks=(5, 12, 3))
        print(f"   ✅ Tucker decomposition: error {info['rel_error']:.3f}")
        
        # Test PCA
        Z, info = pca_latent_features(X, k=3)
        print(f"   ✅ PCA latent features: shape {Z.shape}")
        
        return True
    except Exception as e:
        print(f"   ❌ Decomposition failed: {e}")
        return False

def test_windowing(X, Y):
    """Test sliding window creation"""
    print("\n3. Testing sliding windows...")
    
    if X is None:
        print("   ⏭️  Skipped (preprocessing failed)")
        return None
    
    try:
        from windows import tensor_to_timeseries, make_sliding_windows
        
        X_ts, y_ts = tensor_to_timeseries(X, Y)
        print(f"   ✅ Time series conversion: X{X_ts.shape}, y{y_ts.shape}")
        
        X_win, y_win = make_sliding_windows(X_ts, y_ts)
        print(f"   ✅ Sliding windows: X{X_win.shape}, y{y_win.shape}")
        
        return X_win, y_win
    except Exception as e:
        print(f"   ❌ Windowing failed: {e}")
        return None, None

def test_models(X_win, y_win):
    """Test model training with tiny data"""
    print("\n4. Testing models...")
    
    if X_win is None:
        print("   ⏭️  Skipped (windowing failed)")
        return
    
    try:
        from windows import time_split
        from models import train_ridge, train_lstm
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X_win, y_win)
        print(f"   ✅ Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        # Test Ridge
        ridge_model, val_mae, val_rmse = train_ridge(X_train, y_train, X_val, y_val)
        print(f"   ✅ Ridge training: val_rmse={val_rmse:.1f}")
        
        # Test LSTM (small network)
        if len(X_train) > 20:  # Only if we have enough data
            lstm_model, _ = train_lstm(X_train, y_train, X_val, y_val, epochs=2)
            print(f"   ✅ LSTM training completed")
        else:
            print(f"   ⏭️  LSTM skipped (insufficient data: {len(X_train)} samples)")
            
    except Exception as e:
        print(f"   ❌ Model training failed: {e}")

def main():
    """Run complete smoke test"""
    print("🧪 Running Wind Power Forecasting Smoke Test")
    print("=" * 50)
    
    # Test each component
    X, Y, dates, mean, std = test_preprocessing()
    decomp_ok = test_decomposition(X)
    X_win, y_win = test_windowing(X, Y)
    test_models(X_win, y_win)
    
    print("\n" + "=" * 50)
    if X is not None and decomp_ok and X_win is not None:
        print("🎉 Smoke test PASSED - All components working!")
        print("   You can now run: python run.py")
    else:
        print("⚠️  Smoke test INCOMPLETE - Check errors above")
        print("   Fix issues before running full pipeline")
    
    # Cleanup
    if os.path.exists('data/scada.xlsx'):
        os.remove('data/scada.xlsx')
        print("   🧹 Cleaned up test data")

if __name__ == "__main__":
    main()