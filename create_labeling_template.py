
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "squat_dataset.csv"
OUTPUT_FILE = (
    BASE_DIR / "outputs" / "squat_labeling_template.csv"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File input tidak ditemukan: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

# Jangan menimpa label kualitas data yang sudah ada.
# Buat kolom terpisah untuk label kualitas gerakan.
df["label_teknik"] = ""
df["catatan_evaluasi"] = ""

# Menyimpan template untuk diisi setelah video diperiksa.
df.to_csv(OUTPUT_FILE, index=False)

print("Template labeling berhasil dibuat.")
print(f"Lokasi: {OUTPUT_FILE}")
print("\nDaftar repetisi yang perlu diperiksa:")

print(
    df[
        [
            "repetition",
            "min_knee_angle",
            "duration_seconds",
            "valid_ratio",
            "label_teknik",
            "catatan_evaluasi",
        ]
    ].to_string(index=False)
)
