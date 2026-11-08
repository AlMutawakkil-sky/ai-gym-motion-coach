
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "outputs" / "squat_phase_predictions.csv"
)
OUTPUT_FILE = (
    BASE_DIR / "outputs" / "squat_phase_smoothed.csv"
)

df = pd.read_csv(INPUT_FILE)

required = ["time_seconds", "predicted_phase"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

df = df.sort_values("time_seconds").reset_index(drop=True)

# Simpan label asli agar dapat dibandingkan
df["smoothed_phase"] = df["predicted_phase"].astype(str)

# Aturan konservatif:
# Jika satu label hanya muncul satu frame dan diapit
# label yang sama, ganti label tersebut.
for i in range(1, len(df) - 1):
    previous_phase = df.loc[i - 1, "smoothed_phase"]
    current_phase = df.loc[i, "smoothed_phase"]
    next_phase = df.loc[i + 1, "smoothed_phase"]

    if (
        current_phase != previous_phase
        and current_phase != next_phase
        and previous_phase == next_phase
    ):
        df.loc[i, "smoothed_phase"] = previous_phase

changed = (
    df["predicted_phase"] != df["smoothed_phase"]
).sum()

print("=== HASIL SMOOTHING FASE ===")
print("Jumlah frame:", len(df))
print("Frame yang labelnya berubah:", int(changed))

print("\nDistribusi label asli:")
print(df["predicted_phase"].value_counts())

print("\nDistribusi label setelah smoothing:")
print(df["smoothed_phase"].value_counts())

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nHasil disimpan di: {OUTPUT_FILE}")
