
from pathlib import Path

import joblib
import pandas as pd


# 1. Lokasi file
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "outputs" / "squat_angles.csv"
MODEL_FILE = BASE_DIR / "models" / "random_forest_phase.joblib"
OUTPUT_FILE = BASE_DIR / "outputs" / "squat_phase_predictions.csv"


# 2. Periksa keberadaan file
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File input tidak ditemukan: {INPUT_FILE}")

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_FILE}")


# 3. Muat model dan daftar fitur training
saved = joblib.load(MODEL_FILE)

model = saved["model"]
features = saved["features"]

print("Fitur model:", features)


# 4. Baca data sudut
df = pd.read_csv(INPUT_FILE)

required = ["time_seconds", "right_knee_angle","right_knee_angle_smoothed"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

df["time_seconds"] = pd.to_numeric(
    df["time_seconds"], errors="coerce"
)
df["df"] = pd.to_numeric(
    df["right_knee_angle"], errors="coerce"
)
df["right_knee_angle_smoothed"] = pd.to_numeric(
    df["right_knee_angle_smoothed"], errors="coerce"
)

# Simpan sudut mentah untuk perbandingan
df["right_knee_angle_raw"] = df["right_knee_angle"]
# Gunakan sudut smoothed sebagai fitur model
df["right_knee_angle"] = df["right_knee_angle_smoothed"]

df = df.sort_values("time_seconds").reset_index(drop=True)


# 5. Hitung fitur dengan cara yang sama seperti training
df["angle_change"] = df["right_knee_angle"].diff()
df["time_change"] = df["time_seconds"].diff()

df["angle_velocity"] = (
    df["angle_change"] / df["time_change"]
)

# Tangani nilai tak hingga dan nilai yang hilang
df = df.replace([float("inf"), float("-inf")], float("nan"))

# Periksa jumlah nilai kosong pada setiap fitur
print("\n=== DIAGNOSTIK FITUR ===")

print("Jumlah NaN setiap fitur:")
print(df[features].isna().sum())

# Hitung jumlah baris yang tidak valid
invalid_mask = ~df[features].notna().all(axis=1)

print("\nJumlah baris tidak valid:", invalid_mask.sum())

# Tampilkan beberapa baris yang tidak valid
print("\nContoh baris tidak valid:")

print(
    df.loc[
        invalid_mask,
        [
            "frame",
            "time_seconds",
            "right_knee_angle",
            "angle_change",
            "time_change",
            "angle_velocity"
        ]
    ].head(15).to_string(index=False)
)

# Pilih baris dengan fitur lengkap
valid_mask = ~invalid_mask
valid_data = df.loc[valid_mask].copy()


# 6. Prediksi fase
if valid_data.empty:
    raise ValueError(
        "Tidak ada baris dengan fitur lengkap untuk diprediksi."
    )

valid_data["predicted_phase"] = model.predict(
    valid_data[features]
)

# Simpan hasil prediksi
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
valid_data.to_csv(OUTPUT_FILE, index=False)

print("\n=== HASIL PREDIKSI FASE SQUAT ===")
print("Jumlah frame input:", len(df))
print("Frame yang diprediksi:", len(valid_data))

print("\nDistribusi prediksi:")
print(valid_data["predicted_phase"].value_counts())

print(f"\nHasil disimpan di: {OUTPUT_FILE}")
print("\nPratinjau hasil:")
print(
    valid_data[
        [
            "time_seconds",
            "right_knee_angle",
            "angle_change",
            "angle_velocity",
            "predicted_phase",
        ]
    ].head(10).to_string(index=False)
)
