
from pathlib import Path
import subprocess
import imageio_ffmpeg

input_video = Path("outputs/squat_pose.mp4")
output_video = Path("outputs/squat_pose_h264.mp4")

if not input_video.exists():
    raise FileNotFoundError(
        f"Video tidak ditemukan: {input_video}"
    )

# Mengambil lokasi executable FFmpeg
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

command = [
    ffmpeg,
    "-y",
    "-i", str(input_video),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-an",
    str(output_video)
]

print("Mengonversi video ke H.264...")

subprocess.run(command, check=True)

print("Konversi selesai!")
print("File hasil:", output_video)
print("Ukuran file:", output_video.stat().st_size, "byte")
