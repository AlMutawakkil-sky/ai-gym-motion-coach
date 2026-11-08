
from pathlib import Path
from collections import Counter

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "outputs" / "squat_phase_predictions.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "squat_phase_refined.csv"
STANDING_ANGLE=160

# 1. Baca hasil prediksi asli
df = pd.read_csv(INPUT_FILE)
df = df.sort_values("time_seconds").reset_index(drop=True)

required = ["time_seconds", "predicted_phase","right_knee_angle"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

original = df["predicted_phase"].astype(str).tolist()
refined = original.copy()

# 2. Majority voting dengan jendela 5 frame
radius = 2

for i in range(len(original)):
    start = max(0, i - radius)
    end = min(len(original), i + radius + 1)

    window = original[start:end]
    counts = Counter(window)

    most_common_label, most_common_count = counts.most_common(1)[0]

    # Perbaiki hanya jika label mayoritas benar-benar dominan
    # dan label asli berbeda dari label mayoritas.
    if (
        most_common_label != original[i]
        and most_common_count > len(window) / 2
    ):
        refined[i] = most_common_label

# 3. Simpan label asli dan label hasil perbaikan
df["refined_phase"] = refined
# 3. Deteksi posisi berdiri berdasarkan sudut lutut
df["right_knee_angle"] = pd.to_numeric(
    df["right_knee_angle"],
    errors="coerce"
)

standing_mask = (
    df["right_knee_angle"] >= STANDING_ANGLE
)

df.loc[standing_mask, "refined_phase"] = "standing"

changed = sum(
    old != new for old, new in zip(original, df["refined_phase"].tolist())
)

print("=== HASIL PERBAIKAN TRANSISI ===")
print("Jumlah frame:", len(df))
print("Frame yang berubah:", changed)
print(f"Persentase berubah: {changed / len(df) * 100:.2f}%")

print("\nDistribusi label asli:")
print(df["predicted_phase"].value_counts())

print("\nDistribusi label setelah perbaikan:")
print(df["refined_phase"].value_counts())

# 4. Hitung pergantian label sebelum dan sesudah
def count_transitions(labels):
    return sum(
        labels[i] != labels[i - 1]
        for i in range(1, len(labels))
    )

final_refined=df["refined_phase"].tolist()
print("\nJumlah pergantian label asli:",
      count_transitions(original))

print("Jumlah pergantian label setelah perbaikan:",
      count_transitions(refined))

# 5. Simpan ke CSV baru
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nFile disimpan di: {OUTPUT_FILE}")
