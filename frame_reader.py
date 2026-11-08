
import cv2

video = cv2.VideoCapture("squat.mp4")

if not video.isOpened():
    print("Video gagal dibuka.")
else:
    # Membaca satu frame
    success, frame = video.read()

    if success:
        # Menyimpan frame pertama menjadi gambar
        cv2.imwrite("frame_pertama.jpg", frame)
        print("Frame berhasil disimpan.")
    else:
        print("Frame tidak berhasil dibaca.")
video.release()
