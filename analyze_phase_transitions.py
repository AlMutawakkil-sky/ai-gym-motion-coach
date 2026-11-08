
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "squat_phase_predictions.csv"

df = pd.read_csv(INPUT_FILE)
df = df.sort_values("time_seconds").reset_index(drop=True)

# Hitung perubahan sudut dari frame sebelumnya
df["angle_change"] = df["right_knee_angle"].diff()

print("=== STATISTIK PER FASE PREDIKSI ===")

summary = df.groupby("predicted_phase")["angle_change"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print(summary.round(2))

print("\n=== PROPORSI ARAH PER FASE ===")

tolerance = 1.0

df["direction"] = "stabil"
df.loc[df["angle_change"] < -tolerance, "direction"] = "turun"
df.loc[df["angle_change"] > tolerance, "direction"] = "naik"

cross_tab = pd.crosstab(
    df["predicted_phase"],
    df["direction"],
    normalize="index"
) * 100

print(cross_tab.round(2))

print("\n=== PERUBAHAN LABEL ANTAR-FRAME ===")

df["previous_phase"] = df["predicted_phase"].shift(1)
transitions = df[
    df["previous_phase"].notna()
    & (df["predicted_phase"] != df["previous_phase"])
]

print("Jumlah pergantian label:", len(transitions))
print("\nTransisi yang paling sering:")
print(
    transitions.groupby(
        ["previous_phase", "predicted_phase"]
    ).size().sort_values(ascending=False).head(10)
)
