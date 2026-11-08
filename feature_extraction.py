
from pathlib import Path
import pandas as pd


# Lokasi file
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_DIR / "outputs" / "squat_repetitions.csv"
OUTPUT_CSV = PROJECT_DIR / "outputs" / "squat_features.csv"


def extract_features(df):
    """Mengambil fitur sudut dan durasi untuk setiap repetisi."""

    required = [
        "time_seconds",
        "right_knee_angle",
        "phase",
        "repetitions",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}"
        )

    df = df.copy()

    # Pastikan kolom numerik terbaca sebagai angka
    for col in ["time_seconds", "right_knee_angle", "repetitions"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["phase"] = df["phase"].astype(str).str.upper()

    # Abaikan baris yang tidak memiliki waktu atau hitungan
    df = df.dropna(subset=["time_seconds", "repetitions"])
    df = df.reset_index(drop=True)

    results = []
    start_idx = None
    previous_reps = int(df.loc[0, "repetitions"]) if not df.empty else 0

    for i, row in df.iterrows():

        # Tandai awal gerakan turun
        if row["phase"] == "DESCENDING" and start_idx is None:
            start_idx = i

        current_reps = int(row["repetitions"])

        # Satu repetisi selesai saat hitungan bertambah
        if current_reps > previous_reps and start_idx is not None:

            segment = df.iloc[start_idx:i + 1].copy()

            angles = segment["right_knee_angle"].dropna()

            start_time = segment["time_seconds"].iloc[0]
            end_time = segment["time_seconds"].iloc[-1]

            if not angles.empty:
                results.append({
                    "repetition": current_reps,
                    "start_time_seconds": start_time,
                    "end_time_seconds": end_time,
                    "duration_seconds": end_time - start_time,
                    "min_knee_angle": angles.min(),
                    "max_knee_angle": angles.max(),
                    "mean_knee_angle": angles.mean(),
                    "valid_samples": len(angles),
                    "total_frames": len(segment),
                    "valid_ratio": len(angles) / len(segment),
                })

            # Siapkan pendeteksian repetisi berikutnya
            start_idx = None

        previous_reps = current_reps

    columns = [
        "repetition",
        "start_time_seconds",
        "end_time_seconds",
        "duration_seconds",
        "min_knee_angle",
        "max_knee_angle",
        "mean_knee_angle",
        "valid_samples",
        "total_frames",
        "valid_ratio",
    ]

    return pd.DataFrame(results, columns=columns)


if __name__ == "__main__":

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan: {INPUT_CSV}"
        )

    data = pd.read_csv(INPUT_CSV)
    features = extract_features(data)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_CSV, index=False)

    print("Ekstraksi fitur selesai.")
    print(f"Jumlah repetisi yang dianalisis: {len(features)}")
    print(f"Hasil disimpan di: {OUTPUT_CSV}")

    if features.empty:
        print(
            "Belum ada repetisi yang berhasil dianalisis. "
            "Periksa kolom phase dan repetitions."
        )
    else:
        print("\nHasil fitur per repetisi:")
        print(features.round(2).to_string(index=False))
