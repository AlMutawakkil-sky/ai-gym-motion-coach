
import cv2
import mediapipe as mp
import csv
import pandas as pd

from pathlib import Path
from biomechanics import calculate_angle


VIDEO_PATH = Path("squat.mp4")
MODEL_PATH = Path("models/pose_landmarker.task")
OUTPUT_PATH = Path("outputs/squat_angles.mp4")
CSV_PATH = Path("outputs/squat_angles.csv")

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video tidak ditemukan: {VIDEO_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )


# Indeks landmark MediaPipe
LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)
RunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Video gagal dibuka.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30.0

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Gagal membuat video output.")


frame_index = 0
rows = []

with PoseLandmarker.create_from_options(
    options
) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = round(
            frame_index * 1000 / fps
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        left_knee_angle = None
        right_knee_angle = None
        left_hip_angle = None
        right_hip_angle = None

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # DEBUG: periksa visibility landmark sisi kanan
            right_indices = {
                "right_hip": 24,
                "right_knee": 26,
                "right_ankle": 28,
                }

            if frame_index < 20:
                print(f"\nFrame {frame_index}:")

                for name, index in right_indices.items():
                    point = landmarks[index]
                    print(
                        f"{name}: "
                        f"visibility={point.visibility}, "
                        f"x={point.x:.3f}, "
                        f"y={point.y:.3f}"
                        )
            
            
            def get_point(index):
                point = landmarks[index]

                # Abaikan landmark dengan visibility sangat rendah
                if (
                    point.visibility is not None
                    and point.visibility < 0.3
                ):
                    return None

                return (
                    point.x * width,
                    point.y * height
                )

            # Sudut lutut kiri
            hip = get_point(23)
            knee = get_point(25)
            ankle = get_point(27)

            if all(p is not None for p in [hip, knee, ankle]):
                left_knee_angle = calculate_angle(
                    hip, knee, ankle
                )

            # Sudut lutut kanan
            hip = get_point(24)
            knee = get_point(26)
            ankle = get_point(28)

            if all(p is not None for p in [hip, knee, ankle]):
                right_knee_angle = calculate_angle(
                    hip, knee, ankle
                )

            # Sudut pinggul kiri:
            # bahu - pinggul - lutut
            shoulder = get_point(11)
            hip = get_point(23)
            knee = get_point(25)

            if all(p is not None for p in [shoulder, hip, knee]):
                left_hip_angle = calculate_angle(
                    shoulder, hip, knee
                )

            # Sudut pinggul kanan
            shoulder = get_point(12)
            hip = get_point(24)
            knee = get_point(26)

            if all(p is not None for p in [shoulder, hip, knee]):
                right_hip_angle = calculate_angle(
                    shoulder, hip, knee
                )

            # Menggambar kerangka tubuh
            for connection in (
                mp.tasks.vision.PoseLandmarksConnections
                .POSE_LANDMARKS
            ):
                start = landmarks[connection.start]
                end = landmarks[connection.end]

                x1, y1 = int(start.x * width), int(start.y * height)
                x2, y2 = int(end.x * width), int(end.y * height)

                cv2.line(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

            # Menampilkan sudut pada frame
            text_y = 30

            for label, angle in [
                ("Left knee", left_knee_angle),
                ("Right knee", right_knee_angle),
                ("Left hip", left_hip_angle),
                ("Right hip", right_hip_angle),
            ]:
                value = (
                    f"{angle:.1f} deg"
                    if angle is not None
                    else "N/A"
                )

                cv2.putText(
                    frame,
                    f"{label}: {value}",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                text_y += 25

        # Simpan data tiap frame
        rows.append({
            "frame": frame_index,
            "time_seconds": frame_index / fps,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "left_hip_angle": left_hip_angle,
            "right_hip_angle": right_hip_angle,
        })

        writer.write(frame)
        frame_index += 1


cap.release()
writer.release()

# Mengubah hasil pengukuran menjadi DataFrame
data = pd.DataFrame(rows)

# Kolom sudut yang akan disaring
angle_columns = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle"
]

# Melakukan smoothing pada setiap sudut
for column in angle_columns:

    # Tahap 1: Median filter
    median_filtered = data[column].rolling(
        window=5,
        center=True,
        min_periods=1
    ).median()

    # Tahap 2: Exponential smoothing
    data[column + "_smoothed"] = median_filtered.ewm(
        span=5,
        adjust=False
    ).mean()

# Menyimpan data
data.to_csv(CSV_PATH, index=False)

print("Analisis sudut dan smoothing selesai.")
print("Total frame:", len(data))
print("Video hasil:", OUTPUT_PATH)
print("Data sudut:", CSV_PATH)