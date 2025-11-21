# config.py

DATA_CSV = r"C:\Users\kapir\Documents\T1.csv"   # adjust path

# Exact column names from the CSV
TIMESTAMP_COL = "Date/Time"
TARGET_COL    = "LV ActivePower (kW)"
FEATURE_COLS  = [
    "Wind Speed (m/s)",
    "Theoretical_Power_Curve (KWh)",
    "Wind Direction (°)",
]

# Time resolution and daily shape (10-minute SCADA data)
RESAMPLE_RULE = "10min"
SLOTS_PER_DAY = 144   # 24 h * 60 / 10

# Sliding window forecast
WINDOW  = 36      # last 2 hours
HORIZON = 1       # 10 minutes ahead

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15

CP_RANKS = [3, 5, 10, 15, 20]

TUCKER_RANKS = [
    (10, 20, 3),
    (20, 40, 3),
    (30, 60, 3),
]

SEED = 42
