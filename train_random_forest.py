
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


# 1. Lokasi file
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "outputs" / "squat_phase_dataset.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "random_forest_phase.joblib"

PLOT_FILE = (
    BASE_DIR / "outputs" / "phase_confusion_matrix.png"
)


# 2. Baca dataset
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

FEATURES = [
    "right_knee_angle",
    "angle_change",
    "angle_velocity",
]

TARGET = "phase_label"

required_columns = FEATURES + [TARGET, "time_seconds"]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Kolom berikut tidak ditemukan: "
        + ", ".join(missing)
    )

# Pastikan data numerik dan label tersedia
for col in FEATURES + ["time_seconds"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)
df = df.sort_values("time_seconds").reset_index(drop=True)

print("Jumlah sampel:", len(df))
print("\nDistribusi label:")
print(df[TARGET].value_counts())


# 3. Bagi data berdasarkan urutan waktu
# 70% bagian awal untuk training, 30% bagian akhir untuk testing.
split_index = int(len(df) * 0.70)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print("\nJumlah data training:", len(train_df))
print("Jumlah data testing:", len(test_df))

print("\nDistribusi label training:")
print(y_train.value_counts())

print("\nDistribusi label testing:")
print(y_test.value_counts())

if y_train.nunique() < 2:
    raise ValueError(
        "Data training memiliki kurang dari dua kelas. "
        "Periksa distribusi label dan pembagian waktu."
    )


# 4. Buat dan latih model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)


# 5. Prediksi data testing
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n=== HASIL EVALUASI BASELINE ===")
print(f"Accuracy: {accuracy:.3f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["turun", "bawah", "naik"],
        zero_division=0,
    )
)


# 6. Simpan confusion matrix
PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)

display = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=["turun", "bawah", "naik"],
    display_labels=["Turun", "Bawah", "Naik"],
    cmap="Blues",
    values_format="d",
)

display.ax_.set_title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.close()

print(f"\nConfusion matrix disimpan di: {PLOT_FILE}")


# 7. Simpan model dan daftar fitur
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "classes": list(model.classes_),
    },
    MODEL_FILE,
)

print(f"Model disimpan di: {MODEL_FILE}")
print("\nPelatihan dan evaluasi selesai.")
