
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = (
    BASE_DIR / "outputs" / "squat_phase_predictions.csv"
)
OUTPUT_FILE = (
    BASE_DIR / "outputs" / "phase_predictions_plot.png"
)

df = pd.read_csv(INPUT_FILE)

colors = {
    "turun": "tab:blue",
    "bawah": "tab:orange",
    "naik": "tab:green",
}

plt.figure(figsize=(12, 5))

for label, group in df.groupby("predicted_phase"):
    plt.scatter(
        group["time_seconds"],
        group["right_knee_angle"],
        label=label,
        color=colors.get(label, "gray"),
        s=14,
    )

plt.xlabel("Waktu (detik)")
plt.ylabel("Sudut lutut kanan (derajat)")
plt.title("Prediksi Fase Squat oleh Random Forest")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=150)
print(f"Grafik disimpan di: {OUTPUT_FILE}")

plt.show()
