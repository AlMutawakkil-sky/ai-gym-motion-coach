
from pathlib import Path

import pandas as pd

# Lokasi file
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "squat_features.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "squat_dataset.csv"

# Memastikan file input tersedia
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File dataset tidak ditemukan: {INPUT_FILE}\n"
        "Jalankan feature_extraction.py terlebih dahulu."
    )

# Membaca fitur setiap repetisi
df = pd.read_csv(INPUT_FILE)

print("Kolom yang tersedia:")
print(df.columns.tolist())

print("\nJumlah repetisi:", len(df))

# Memeriksa kelengkapan kolom
required_columns = [
    "min_knee_angle",
    "duration_seconds",
    "valid_ratio",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Kolom berikut belum tersedia: "
        + ", ".join(missing_columns)
        + "\nPeriksa nama kolom di squat_features.csv."
    )

# Mengubah fitur menjadi angka
for column in required_columns:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

# Memeriksa kelengkapan fitur utama
df["data_valid"] = (
    df[required_columns].notna().all(axis=1)
    & df["valid_ratio"].between(0, 1)
    & df["min_knee_angle"].between(0, 180)
    & (df["duration_seconds"] > 0)
)

# Label awal untuk memeriksa kualitas data.
# Ini BELUM menilai benar/salahnya teknik squat.
df["label"] = df["data_valid"].map({
    True: "siap_ditinjau",
    False: "data_kurang"
})

# Menyimpan dataset
OUTPUT_FILE.parent.mkdir(
    parents=True, exist_ok=True
)

df.to_csv(OUTPUT_FILE, index=False)

print("\nDistribusi label:")
print(df["label"].value_counts())

print(f"\nDataset disimpan di: {OUTPUT_FILE}")
print("\nPratinjau dataset:")
print(df.head())
