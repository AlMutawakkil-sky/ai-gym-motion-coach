
import pandas as pd
import matplotlib.pyplot as plt

# Membaca data sudut lutut
df = pd.read_csv("outputs/squat_angles.csv")

# Memastikan kolom yang diperlukan tersedia
required_columns = [
    "time_seconds",
    "left_knee_angle",
    "right_knee_angle"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

# Menghitung selisih sudut ketika kedua lutut tersedia
valid = df.dropna(
    subset=["left_knee_angle", "right_knee_angle"]
).copy()

valid["angle_difference"] = (
    valid["left_knee_angle"] -
    valid["right_knee_angle"]
).abs()

print("=== PERBANDINGAN SUDUT LUTUT ===")
print(f"Total frame video: {len(df)}")
print(f"Frame dengan kedua sudut valid: {len(valid)}")
print(
    "Rata-rata selisih sudut: "
    f"{valid['angle_difference'].mean():.2f} derajat"
)
print(
    "Median selisih sudut: "
    f"{valid['angle_difference'].median():.2f} derajat"
)

# Membuat grafik perbandingan
plt.figure(figsize=(13, 6))

plt.plot(
    df["time_seconds"],
    df["left_knee_angle"],
    label="Lutut kiri",
    linewidth=1.8
)

plt.plot(
    df["time_seconds"],
    df["right_knee_angle"],
    label="Lutut kanan",
    linewidth=1.8
)

plt.xlabel("Waktu (detik)")
plt.ylabel("Sudut lutut (derajat)")
plt.title("Perbandingan Sudut Lutut Kiri dan Kanan")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
