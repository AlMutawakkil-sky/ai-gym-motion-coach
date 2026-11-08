
import pandas as pd
from pathlib import Path


# Lokasi file
INPUT_CSV = Path("outputs/squat_angles.csv")
OUTPUT_CSV = Path("outputs/squat_repetitions.csv")


# Ambang awal untuk eksperimen
STANDING_ANGLE = 160
BOTTOM_ANGLE = 100
MIN_REP_DURATION = 0.8  # detik


def count_squats(df):
    """
    Menghitung repetisi squat berdasarkan sudut lutut.

    Satu repetisi dihitung setelah:
    berdiri -> turun -> posisi bawah -> kembali berdiri.
    """

    df = df.copy()

    # Gunakan sudut lutut kanan sebagai acuan awal
    angle_column = "right_knee_angle_smoothed"

    if angle_column not in df.columns:
        raise ValueError(
            f"Kolom {angle_column} tidak ditemukan."
        )

    # Ubah nilai sudut menjadi angka.
    # Nilai None atau kosong akan menjadi NaN.
    df[angle_column] = pd.to_numeric(
        df[angle_column], errors="coerce"
    )

    # Pastikan kolom waktu juga berupa angka
    df["time_seconds"] = pd.to_numeric(
        df["time_seconds"], errors="coerce"
    )

    state = "STANDING"
    reps = 0
    bottom_time = None
    rep_start_time = None

    phases = []
    rep_counts = []

    for _, row in df.iterrows():
        angle = row[angle_column]
        time = row["time_seconds"]

        # Jika data sudut tidak tersedia,
        # jangan mengubah status atau menghitung repetisi.
        if pd.isna(angle) or pd.isna(time):
            phases.append(state)
            rep_counts.append(reps)
            continue

        if state == "STANDING":
            if angle < STANDING_ANGLE:
                state = "DESCENDING"
                rep_start_time = time

        elif state == "DESCENDING":
            if angle <= BOTTOM_ANGLE:
                state = "BOTTOM"
                bottom_time = time

            # Jika kembali berdiri tanpa mencapai
            # ambang bawah, anggap gerakan belum lengkap.
            elif angle >= STANDING_ANGLE:
                state = "STANDING"
                rep_start_time = None

        elif state == "BOTTOM":
            # Sudut mulai membesar kembali
            if angle > BOTTOM_ANGLE + 10:
                state = "ASCENDING"

        elif state == "ASCENDING":
            if angle <= BOTTOM_ANGLE:
                # Jika turun kembali, tetap di siklus yang sama
                state = "BOTTOM"

            elif angle >= STANDING_ANGLE:
                duration = time - rep_start_time

                if duration >= MIN_REP_DURATION:
                    reps += 1

                state = "STANDING"
                rep_start_time = None
                bottom_time = None

        phases.append(state)
        rep_counts.append(reps)

    df["phase"] = phases
    df["repetitions"] = rep_counts

    return df


if __name__ == "__main__":
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_CSV}"
        )

    data = pd.read_csv(INPUT_CSV)

    result = count_squats(data)

    result.to_csv(OUTPUT_CSV, index=False)

    print("Analisis repetisi selesai.")
    print("Jumlah repetisi:", result["repetitions"].max())
    print("File hasil:", OUTPUT_CSV)

    print("\nContoh hasil:")
    print(
        result[
            [
                "time_seconds",
                "right_knee_angle",
                "phase",
                "repetitions",
            ]
        ].head(15).to_string(index=False)
    )
