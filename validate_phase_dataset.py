
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = (
    BASE_DIR / "outputs" / "squat_phase_dataset.csv"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "time_seconds",
    "right_knee_angle",
    "angle_change",
    "angle_velocity",
    "phase_label",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Kolom yang hilang: " + ", ".join(missing)
    )

print("=== VALIDASI DATASET FASE SQUAT ===")
print("Jumlah baris:", len(df))

print("\nNilai kosong per kolom:")
print(df[required_columns].isna().sum())

print("\nDistribusi label:")
print(df["phase_label"].value_counts())

print("\nStatistik fitur:")
print(
    df[
        [
            "right_knee_angle",
            "angle_change",
            "angle_velocity",
        ]
    ].describe().round(2)
)

# Periksa apakah data waktu berurutan
times = pd.to_numeric(
    df["time_seconds"], errors="coerce"
)

time_differences = times.diff().dropna()

print("\nUrutan waktu:")
print("Semua selisih waktu positif:",
      bool((time_differences > 0).all()))

print("\nDurasi fase berdasarkan rentang waktu:")

for label, group in df.groupby("phase_label"):
    duration = (
        group["time_seconds"].max()
        - group["time_seconds"].min()
    )

    print(f"{label}: {duration:.2f} detik")

print("\nPemeriksaan selesai.")
