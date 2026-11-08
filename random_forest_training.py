
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "squat_labeled.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

# Fitur yang akan dipertimbangkan untuk model.
feature_columns = [
    "min_knee_angle",
    "duration_seconds",
    "mean_knee_angle",
    "valid_ratio",
]

target_column = "label_teknik"

# Periksa keberadaan kolom.
required_columns = feature_columns + [target_column]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Kolom berikut tidak ditemukan: "
        + ", ".join(missing_columns)
    )

# Ambil fitur dan label, lalu buang baris yang tidak lengkap.
data = df[required_columns].copy()

for col in feature_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=required_columns)

print("=== PEMERIKSAAN DATASET ===")
print("Jumlah baris valid:", len(data))
print("Jumlah fitur:", len(feature_columns))
print("\nDistribusi label:")
print(data[target_column].value_counts())

number_of_classes = data[target_column].nunique()

print("\nJumlah kelas:", number_of_classes)

if number_of_classes < 2:
    print(
        "\nDATA BELUM SIAP UNTUK KLASIFIKASI."
        "\nLabel hanya memiliki satu kelas."
        "\nTambahkan contoh dari kelas lain yang "
        "dinilai menggunakan kriteria konsisten."
    )
else:
    print(
        "\nDataset memiliki lebih dari satu kelas."
        "\nLangkah berikutnya adalah memeriksa "
        "jumlah sampel per kelas dan menyiapkan "
        "evaluasi sebelum melatih model."
    )
