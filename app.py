from pathlib import Path
import streamlit as st
import cv2
import pandas as pd
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lokasi target file input yang dibaca oleh skrip analisis backend
INPUT_VIDEO_ROOT = PROJECT_DIR / "squat.mp4"
INPUT_VIDEO_RAW = INPUT_DIR / "squat.mp4"

# File hasil analisis mentah
RAW_POSE_VIDEO = OUTPUT_DIR / "squat_pose.mp4"
RAW_ANGLE_VIDEO = OUTPUT_DIR / "squat_angles.mp4"

# File hasil konversi yang ramah web (Wajib menggunakan _h264.mp4 agar tampil di browser)
POSE_VIDEO = OUTPUT_DIR / "squat_pose_h264.mp4"
ANGLE_VIDEO = OUTPUT_DIR / "squat_angles_h264.mp4"

ANGLE_CSV = OUTPUT_DIR / "squat_angles.csv"
REPETITION_CSV = OUTPUT_DIR / "squat_repetitions.csv"
PREDICTION_SCRIPT = PROJECT_DIR / "predict_squat_phase.py"
REFINE_SCRIPT = PROJECT_DIR / "refine_phase_transitions.py"
PREDICTION_CSV = OUTPUT_DIR / "squat_phase_predictions.csv"
REFINED_CSV = OUTPUT_DIR / "squat_phase_refined.csv"
POSE_SCRIPT = PROJECT_DIR / "pose_detection.py"
ANGLE_SCRIPT = PROJECT_DIR / "pose_angle_analysis.py"
REPETITION_SCRIPT = PROJECT_DIR / "repetition_counter.py"
CONVERT_SCRIPT = PROJECT_DIR / "convert_video.py"


def run_script(script_path):
    """Menjalankan skrip Python eksternal menggunakan environment aktif."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        error_output = e.stdout + "\n" + e.stderr
        return False, error_output



def clear_old_results():
    """Menghapus hasil analisis lama agar tidak tertukar dengan hasil baru."""
    result_files = [
        RAW_POSE_VIDEO,
        RAW_ANGLE_VIDEO,
        POSE_VIDEO,
        ANGLE_VIDEO,
        ANGLE_CSV,
        REPETITION_CSV,
        PREDICTION_CSV,
        REFINED_CSV,  # Tambahan: hapus hasil majority voting lama
    ]

    for file_path in result_files:
        if file_path.exists():
            try:
                file_path.unlink()
            except OSError as error:
                st.warning(
                    f"Gagal menghapus {file_path.name}: {error}"
                )



st.subheader("Unggah Video Squat")
st.caption("Unggah MP4 untuk memperbarui file input analisis secara otomatis.")
uploaded_video = st.file_uploader(
    "Pilih video squat dari komputer",
    type=["mp4"],
    key="squat_video_upload",
)

if uploaded_video is not None:
    new_bytes = uploaded_video.getvalue()
    
    # Simpan file yang diunggah secara otomatis menimpa file squat.mp4 utama & di data/raw/
    INPUT_VIDEO_ROOT.write_bytes(new_bytes)
    INPUT_VIDEO_RAW.write_bytes(new_bytes)
    
    st.success("Video baru berhasil diunggah dan disinkronkan ke sistem!")
    st.video(new_bytes)

    cap = cv2.VideoCapture(str(INPUT_VIDEO_ROOT))
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        st.subheader("Informasi Video")
        c1, c2, c3 = st.columns(3)
        c1.metric("FPS", f"{fps:.2f}")
        c2.metric("Resolusi", f"{width} × {height}")
        c3.metric("Durasi", f"{duration:.2f} detik")

        ok, frame = cap.read()
        if ok:
            st.subheader("Frame Pertama")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=600)
        cap.release()
    else:
        st.error("OpenCV tidak dapat membaca video yang diunggah.")

    # Tombol untuk mengeksekusi ulang seluruh pipeline analisis berdasarkan video baru
    if st.button("Proses Video Terbaru", type="primary"):
        clear_old_results()
        pipeline = [
            ("Deteksi pose", POSE_SCRIPT),
            ("Analisis sudut", ANGLE_SCRIPT),
            ("Penghitungan repetisi", REPETITION_SCRIPT),
            ("Prediksi fase Random Forest", PREDICTION_SCRIPT),
            ("Perbaikan fase / majority voting", REFINE_SCRIPT),
            ("Konversi video web", CONVERT_SCRIPT),
        ]
        completed = True

        for step_name, script_path in pipeline:
            if script_path.exists():
                with st.status(f"Menjalankan: {step_name}...", expanded=True) as status:
                    success, log = run_script(script_path)
                    st.code(log, language="text")
                    if success:
                        status.update(label=f"{step_name}: selesai", state="complete")
                    else:
                        status.update(label=f"{step_name}: gagal", state="error")
                        st.error("Pipeline dihentikan. Periksa pesan error di atas.")
                        completed = False
                        break
            else:
                st.warning(f"Skrip {script_path.name} tidak ditemukan, melewati langkah ini.")

        if completed:
            st.success("Pipeline selesai! Hasil analisis diperbarui.")
            st.rerun()
else:
    st.info("Unggah video MP4 untuk memulai analisis.")

st.divider()

st.subheader("Hasil Deteksi Pose")
if POSE_VIDEO.exists() and POSE_VIDEO.stat().st_size > 0:
    st.video(str(POSE_VIDEO))
else:
    st.info("Hasil belum tersedia. Unggah video dan klik 'Proses Video Terbaru'.")

st.subheader("Analisis Sudut Sendi")
if ANGLE_VIDEO.exists() and ANGLE_VIDEO.stat().st_size > 0:
    st.video(str(ANGLE_VIDEO))

if ANGLE_CSV.exists() and ANGLE_CSV.stat().st_size > 0:
    angle_df = pd.read_csv(ANGLE_CSV)
    st.subheader("Data Sudut per Frame")
    st.dataframe(angle_df, use_container_width=True)

    angle_cols = [c for c in ["right_knee_angle", "right_knee_angle_smoothed"] if c in angle_df.columns]

    if "time_seconds" in angle_df.columns and angle_cols:
        st.subheader("Grafik Sudut Lutut Mentah dan Smoothed")
        st.line_chart(angle_df.set_index("time_seconds")[angle_cols])
else:
    st.info("Belum ada CSV sudut. Jalankan proses video terlebih dahulu.")

st.subheader("Penghitung Repetisi Squat")
if REPETITION_CSV.exists() and REPETITION_CSV.stat().st_size > 0:
    rep_df = pd.read_csv(REPETITION_CSV)
    if "repetitions" in rep_df.columns and not rep_df["repetitions"].dropna().empty:
        st.metric("Total Repetisi Terdeteksi", int(rep_df["repetitions"].max()))
    if "time_seconds" in rep_df.columns and "repetitions" in rep_df.columns:
        st.subheader("Perubahan Jumlah Repetisi")
        st.line_chart(rep_df.set_index("time_seconds")[["repetitions"]])
    display_cols = [c for c in ["time_seconds", "right_knee_angle", "phase", "repetitions"] if c in rep_df.columns]
    if display_cols:
        st.subheader("Fase Gerakan per Frame")
        st.dataframe(rep_df[display_cols], use_container_width=True)
else:
    st.info("Hasil repetisi belum tersedia. Jalankan proses video terlebih dahulu.")



st.divider()
st.subheader("Prediksi Fase Squat — Random Forest")

if PREDICTION_CSV.exists() and PREDICTION_CSV.stat().st_size > 0:
    prediction_df = pd.read_csv(PREDICTION_CSV)

    if "predicted_phase" not in prediction_df.columns:
        st.error("Kolom predicted_phase tidak ditemukan.")
    else:
        st.write("### 1. Prediksi asli Random Forest")

        
        # Ringkasan hasil prediksi
        total_frames = len(prediction_df)

        phase_counts = prediction_df["predicted_phase"].value_counts()
        bottom_frames = int(phase_counts.get("bawah", 0))
        bottom_percentage = (
            bottom_frames / total_frames * 100
            if total_frames > 0 else 0
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Total frame",
            total_frames
        )

        metric2.metric(
            "Jumlah fase",
            prediction_df["predicted_phase"].nunique()
        )

        metric3.metric(
            "Frame fase bawah",
            f"{bottom_percentage:.1f}%"
        )


        phase_counts = (
            prediction_df["predicted_phase"]
            .value_counts()
            .rename_axis("Fase")
            .reset_index(name="Jumlah frame")
        )

        st.dataframe(phase_counts, use_container_width=True)

        if {
            "time_seconds",
            "right_knee_angle",
            "predicted_phase",
        }.issubset(prediction_df.columns):

            original_plot = prediction_df[
                ["time_seconds", "right_knee_angle", "predicted_phase"]
            ].dropna().copy()

            st.scatter_chart(
                original_plot,
                x="time_seconds",
                y="right_knee_angle",
                color="predicted_phase",
                x_label="Waktu (detik)",
                y_label="Sudut lutut kanan (derajat)",
            )

        # Hasil setelah majority voting
        st.write("### 2. Prediksi setelah majority voting")

        if REFINED_CSV.exists() and REFINED_CSV.stat().st_size > 0:
            refined_df = pd.read_csv(REFINED_CSV)

            if "refined_phase" not in refined_df.columns:
                st.warning(
                    "Kolom refined_phase tidak ditemukan "
                    "dalam file hasil perbaikan."
                )
            else:
                refined_counts = (
                    refined_df["refined_phase"]
                    .value_counts()
                    .rename_axis("Fase")
                    .reset_index(name="Jumlah frame")
                )

                st.dataframe(refined_counts, use_container_width=True)

                if {
                    "time_seconds",
                    "right_knee_angle",
                    "refined_phase",
                }.issubset(refined_df.columns):

                    refined_plot = refined_df[
                        ["time_seconds", "right_knee_angle", "refined_phase"]
                    ].dropna().copy()

                    st.scatter_chart(
                        refined_plot,
                        x="time_seconds",
                        y="right_knee_angle",
                        color="refined_phase",
                        x_label="Waktu (detik)",
                        y_label="Sudut lutut kanan (derajat)",
                    )
                else:
                    st.warning(
                        "Kolom waktu atau sudut lutut tidak lengkap "
                        "pada file hasil perbaikan."
                    )
        else:
            st.info(
                "File hasil majority voting belum tersedia. "
                "Jalankan skrip perbaikan fase terlebih dahulu."
            )

        st.caption(
            "Fase adalah estimasi gerakan berdasarkan fitur sudut lutut. "
            "Hasil ini bukan penilaian klinis teknik squat."
        )
else:
    st.info(
        "Belum ada hasil prediksi. Unggah video dan klik "
        "'Proses Video Terbaru'."
    )
