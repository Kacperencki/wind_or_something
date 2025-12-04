# config.py

# --- DATA PATHS -------------------------------------------------------------
# IMPORTANT: Set DATA_CSV environment variable or place scada.xlsx in data/ folder
import os
DATA_CSV = os.getenv("DATA_CSV", "data/scada.xlsx")

# Alternative: uncomment and set your absolute path
# DATA_CSV = r"C:\Users\your_username\path\to\scada.xlsx"

# --- COLUMN NAMES -----------------------------------------------------------
TIMESTAMP_COL = "Timestamps"
TARGET_COL    = "Power"

# 15 input features (X), target is Power
FEATURE_COLS = [
    "WindSpeed",
    "StdDevWindSpeed",
    "WindDirAbs",
    "WindDirRel",
    "AvgRPow",
    "GenRPM",
    "RotorRPM",
    "EnvirTemp",
    "NacelTemp",
    "GearOilTemp",
    "GearBearTemp",
    "GenPh1Temp",
    "GenPh2Temp",
    "GenPh3Temp",
    "GenBearTemp",
]

# --- TEMPORAL STRUCTURE -----------------------------------------------------
RESAMPLE_RULE = "10min"
SLOTS_PER_DAY = 144  # 24h * 60 / 10

MAX_DAYS = 5000  # for example; you can set to 365, 2000, or None

# --- SLIDING WINDOW FORECAST ------------------------------------------------
WINDOW  = 12   # last 2 hours as input
HORIZON = 1    # 1 step (10 min) ahead

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15   # test = 0.15

# --- DECOMPOSITION SETTINGS -------------------------------------------------
CP_RANKS = [ 12]

# Tucker ranks: (rank_days, rank_slots, rank_features)
# R_D: number of latent "day types"
# R_T: number of latent "time-of-day" patterns (<= 144)
# R_F: latent SCADA feature dim given to the forecaster
TUCKER_RANKS = [
    (50, 24, 5),
    (80, 24, 7),
    (80, 24, 9)
]


# PCA latent dims
PCA_K = [3, 5]

SEED = 42
