
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "squat_repetitions.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "squat_phase_dataset.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File tidak ditemukan: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("Kolom yang tersedia:")
print(df.columns.tolist())

required = [
    "time_seconds",
    "right_knee_angle",
    "phase",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(
        "Kolom berikut belum tersedia: "
        + ", ".join(missing)
        + "\nPeriksa file squat_repetitions.csv."
    )

# Konversi tipe data
df["time_seconds"] = pd.to_numeric(
    df["time_seconds"], errors="coerce"
)
df["right_knee_angle"] = pd.to_numeric(
    df["right_knee_angle"], errors="coerce"
)

# Urutkan frame berdasarkan waktu
df = df.sort_values("time_seconds").reset_index(drop=True)

# Normalisasi nama fase dari program sebelumnya
phase_map = {
    "DESCENDING": "turun",
    "BOTTOM": "bawah",
    "ASCENDING": "naik",
}

df["phase_normalized"] = (
    df["phase"].astype(str).str.strip().str.upper()
)

df["phase_label"] = df["phase_normalized"].map(phase_map)

# Hitung perubahan sudut dan waktu antar-frame
df["angle_change"] = df["right_knee_angle"].diff()
df["time_change"] = df["time_seconds"].diff()

# Hindari pembagian dengan nol
df["angle_velocity"] = (
    df["angle_change"] / df["time_change"]
)

# Ambil hanya frame yang memiliki label target dan fitur lengkap
feature_columns = [
    "right_knee_angle",
    "angle_change",
    "angle_velocity",
]

dataset = df.dropna(
    subset=feature_columns + ["phase_label"]
).copy()

dataset = dataset[
    dataset["time_change"] > 0
].copy()

OUTPUT_FILE.parent.mkdir(
    parents=True, exist_ok=True
)
dataset.to_csv(OUTPUT_FILE, index=False)

print("\nJumlah sampel berlabel:", len(dataset))
print("\nDistribusi kelas:")
print(dataset["phase_label"].value_counts())

print(f"\nDataset disimpan di: {OUTPUT_FILE}")
