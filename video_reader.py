
import cv2

# Ganti dengan lokasi video milikmu nanti
video_path = "squat.mp4"

# Membuka file video
video = cv2.VideoCapture(video_path)

# Memeriksa apakah video berhasil dibuka
if not video.isOpened():
    print("Video gagal dibuka.")
    print("Periksa nama dan lokasi file.")
else:
    # Membaca informasi video
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Menghitung durasi video
    duration = total_frames / fps if fps > 0 else 0

    print("Video berhasil dibuka!")
    print("FPS:", fps)
    print("Resolusi:", width, "x", height)
    print("Jumlah frame:", total_frames)
    print("Durasi:", round(duration, 2), "detik")

# Melepaskan resource video
video.release()
