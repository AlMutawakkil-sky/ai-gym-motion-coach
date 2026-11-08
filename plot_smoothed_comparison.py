
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "squat_phase_smoothed.csv"

df = pd.read_csv(INPUT_FILE)

colors = {
    "turun": "tab:blue",
    "bawah": "tab:orange",
    "naik": "tab:green",
}

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

for ax, column, title in [
    (axes[0], "predicted_phase", "Label Asli Random Forest"),
    (axes[1], "smoothed_phase", "Label Setelah Smoothing"),
]:
    for phase, group in df.groupby(column):
        ax.scatter(
            group["time_seconds"],
            group["right_knee_angle"],
            label=phase,
            color=colors.get(phase, "gray"),
            s=12,
        )

    ax.set_ylabel("Sudut lutut kanan (derajat)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[1].set_xlabel("Waktu (detik)")
plt.tight_layout()
plt.show()
