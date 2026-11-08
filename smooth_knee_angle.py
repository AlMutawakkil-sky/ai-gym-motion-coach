
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "outputs/squat_angles.csv"
OUTPUT_FILE = "outputs/squat_angles_smoothed.csv"

df = pd.read_csv(INPUT_FILE)

# Salin sudut lutut kiri sebagai sumber utama
df["left_knee_angle_smoothed"] = (
    df["left_knee_angle"]
    .rolling(window=5, center=True, min_periods=1)
    .median()
)

# Simpan sebagai file baru agar data asli tetap aman
df.to_csv(OUTPUT_FILE, index=False)

print("=== SMOOTHING SUDUT LUTUT ===")
print(f"Jumlah frame: {len(df)}")
print(
    "NaN sudut lutut kiri sebelum smoothing:",
    df["left_knee_angle"].isna().sum()
)
print(
    "NaN sudut lutut kiri setelah smoothing:",
    df["left_knee_angle_smoothed"].isna().sum()
)
print(f"File disimpan: {OUTPUT_FILE}")

# Visualisasi sebelum dan sesudah smoothing
plt.figure(figsize=(13, 6))

plt.plot(
    df["time_seconds"],
    df["left_knee_angle"],
    label="Sudut kiri asli",
    alpha=0.55
)

plt.plot(
    df["time_seconds"],
    df["left_knee_angle_smoothed"],
    label="Sudut kiri setelah smoothing",
    linewidth=2
)

plt.xlabel("Waktu (detik)")
plt.ylabel("Sudut lutut (derajat)")
plt.title("Perbandingan Sudut Lutut Sebelum dan Sesudah Smoothing")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
