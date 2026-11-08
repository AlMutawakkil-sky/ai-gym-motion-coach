
import cv2
import mediapipe as mp
from pathlib import Path

# Lokasi file proyek
VIDEO_PATH = Path("squat.mp4")
MODEL_PATH = Path("models/pose_landmarker.task")
OUTPUT_PATH = Path("outputs/squat_pose.mp4")

# Pastikan input tersedia
if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video tidak ditemukan: {VIDEO_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

# API MediaPipe Pose Landmarker
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)
RunningMode = mp.tasks.vision.RunningMode

# Konfigurasi model untuk video
options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

# Membuka video
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Video gagal dibuka.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30.0

# Menyiapkan file video keluaran
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        "Video hasil tidak dapat dibuat."
    )

frame_index = 0
detected_frames = 0

# Membuat objek detektor
with PoseLandmarker.create_from_options(
    options
) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            break

        # OpenCV BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        # Membuat objek gambar MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp harus meningkat untuk setiap frame
        timestamp_ms = round(
            frame_index * 1000 / fps
        )

        # Mendeteksi pose
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Menggambar landmark dan koneksi tubuh
        if result.pose_landmarks:
            detected_frames += 1

            for pose_landmarks in result.pose_landmarks:
                for connection in (
                    mp.tasks.vision.PoseLandmarksConnections
                    .POSE_LANDMARKS
                ):
                    start = pose_landmarks[
                        connection.start
                    ]
                    end = pose_landmarks[
                        connection.end
                    ]

                    x1 = int(start.x * width)
                    y1 = int(start.y * height)
                    x2 = int(end.x * width)
                    y2 = int(end.y * height)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                # Menggambar titik landmark
                for landmark in pose_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(
                            frame,
                            (x, y),
                            3,
                            (0, 0, 255),
                            -1
                        )

        else:
            cv2.putText(
                frame,
                "Pose not detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        writer.write(frame)
        frame_index += 1

# Menutup file dengan benar
cap.release()
writer.release()

print("Analisis selesai.")
print("Total frame:", frame_index)
print("Frame dengan pose:", detected_frames)

if frame_index > 0:
    percentage = detected_frames / frame_index * 100
    print("Persentase deteksi:", round(percentage, 2), "%")

print("Video hasil:", OUTPUT_PATH)
