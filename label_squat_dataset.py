
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "squat_dataset.csv"
OUTPUT_FILE = (
    BASE_DIR / "outputs" / "squat_labeled.csv"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File tidak ditemukan: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

# Pastikan kolom identitas repetisi tersedia.
if "repetition" not in df.columns:
    raise ValueError(
        "Kolom 'repetition' tidak ditemukan."
    )

# Siapkan kolom label yang terpisah dari label kualitas data.
df["label_teknik"] = ""
df["status_verifikasi"] = "belum_diperiksa"
df["catatan_evaluasi"] = ""

# Karena sumbernya adalah video demonstrasi teknik baik,
# gunakan label awal yang menyatakan asal contoh,
# bukan klaim bahwa AI telah menilai tekniknya.
df["label_teknik"] = "contoh_teknik_baik"
df["status_verifikasi"] = "label_awal_dari_sumber_video"
df["catatan_evaluasi"] = (
    "Video demonstrasi bodyweight squat; "
    "perlu verifikasi visual per repetisi."
)

OUTPUT_FILE.parent.mkdir(
    parents=True, exist_ok=True
)
df.to_csv(OUTPUT_FILE, index=False)

print("Dataset labeling berhasil dibuat.")
print(f"Lokasi: {OUTPUT_FILE}")
print("\nJumlah repetisi:", len(df))
print("\nDistribusi label:")
print(df["label_teknik"].value_counts())
print("\nStatus verifikasi:")
print(df["status_verifikasi"].value_counts())
