
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "squat_phase_predictions.csv"

df = pd.read_csv(INPUT_FILE)
df = df.sort_values("time_seconds").reset_index(drop=True)

# Hitung perubahan sudut lutut per frame
df["angle_change_check"] = df["right_knee_angle"].diff()

# Gunakan toleransi kecil untuk mengabaikan perubahan yang sangat kecil
tolerance = 1.0

df["observed_direction"] = "stabil"
df.loc[df["angle_change_check"] < -tolerance, "observed_direction"] = "turun"
df.loc[df["angle_change_check"] > tolerance, "observed_direction"] = "naik"

# Bandingkan arah sudut dengan label fase prediksi.
# Fase bawah tidak dinilai berdasarkan arah, karena lutut bisa stabil
# di sekitar titik terdalam.
df["direction_match"] = pd.NA

mask = df["predicted_phase"].isin(["turun", "naik"])
df.loc[mask, "direction_match"] = (
    df.loc[mask, "predicted_phase"]
    == df.loc[mask, "observed_direction"]
)

print("=== EVALUASI ARAH GERAKAN ===")
print("\nDistribusi arah berdasarkan sudut:")
print(df["observed_direction"].value_counts())

print("\nKesesuaian arah dan label prediksi:")
print(
    df.loc[mask]
    .groupby("predicted_phase")["direction_match"]
    .value_counts()
    .unstack(fill_value=0)
)

valid = df.loc[mask, "direction_match"].dropna()

if len(valid) > 0:
    print(
        "\nPersentase kesesuaian arah:",
        round(valid.astype(bool).mean() * 100, 2),
        "%"
    )

print("\nContoh data:")
print(
    df[
        [
            "time_seconds",
            "right_knee_angle",
            "angle_change_check",
            "observed_direction",
            "predicted_phase",
            "direction_match",
        ]
    ].head(20).to_string(index=False)
)
