
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "squat_phase_refined.csv"

df = pd.read_csv(INPUT_FILE)

colors = {
    "turun": "tab:blue",
    "bawah": "tab:orange",
    "naik": "tab:green",
}

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

plots = [
    ("predicted_phase", "Prediksi Asli Random Forest"),
    ("refined_phase", "Prediksi Setelah Majority Voting"),
]

for ax, (column, title) in zip(axes, plots):
    for phase in ["turun", "bawah", "naik"]:
        group = df[df[column] == phase]

        ax.scatter(
            group["time_seconds"],
            group["right_knee_angle"],
            color=colors[phase],
            label=phase,
            s=12,
        )

    ax.set_title(title)
    ax.set_ylabel("Sudut lutut kanan (derajat)")
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Waktu (detik)")

plt.tight_layout()
plt.show()
