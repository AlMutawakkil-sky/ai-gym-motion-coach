
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = (
    BASE_DIR / "outputs" / "squat_phase_dataset.csv"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)
df = df.sort_values("time_seconds")

colors = {
    "turun": "tab:blue",
    "bawah": "tab:orange",
    "naik": "tab:green",
}

plt.figure(figsize=(12, 5))

for label, group in df.groupby("phase_label"):
    plt.scatter(
        group["time_seconds"],
        group["right_knee_angle"],
        label=label,
        color=colors.get(label, "gray"),
        s=16,
    )

plt.xlabel("Waktu (detik)")
plt.ylabel("Sudut lutut kanan (derajat)")
plt.title("Sudut Lutut dan Label Fase Squat")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_file = BASE_DIR / "outputs" / "squat_phase_plot.png"
plt.savefig(output_file, dpi=150)
print(f"Grafik disimpan di: {output_file}")

plt.show()
