from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "outputs" / "squat_phase_refined.csv"
)

df = pd.read_csv(INPUT_FILE)

# Statistik sudut lutut berdasarkan fase
summary = df.groupby("refined_phase")[
    "right_knee_angle"
].agg([
    "count",
    "mean",
    "min",
    "max"
])

print("=== STATISTIK SUDUT LUTUT BERDASARKAN FASE ===")
print(summary.round(2))

# Periksa jumlah frame standing
standing = df[
    df["refined_phase"] == "standing"
]

print("\n=== ANALISIS STANDING ===")
print("Jumlah frame standing:", len(standing))

print(
    "Rentang waktu standing:",
    round(standing["time_seconds"].min(), 2),
    "-",
    round(standing["time_seconds"].max(), 2),
    "detik"
)

print("\n=== URUTAN PERUBAHAN FASE ===")

df["phase_group"] = (
    df["refined_phase"] != df["refined_phase"].shift()
).cumsum()

segments = df.groupby("phase_group").agg(
    phase=("refined_phase", "first"),
    start_time=("time_seconds", "min"),
    end_time=("time_seconds", "max"),
    frame_count=("refined_phase", "count")
)

print(segments.round(2).to_string(index=False))