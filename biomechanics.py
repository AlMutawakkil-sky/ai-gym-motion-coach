
import numpy as np


def calculate_angle(a, b, c):
    """
    Menghitung sudut ABC dalam derajat.

    a, b, c: koordinat 2D berupa (x, y)
    b: titik tempat sudut dihitung
    """

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    # Membuat dua vektor dari titik B
    ba = a - b
    bc = c - b

    # Menghitung panjang kedua vektor
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Mencegah pembagian dengan nol
    if norm_ba == 0 or norm_bc == 0:
        return None

    # Dot product kedua vektor
    dot_product = np.dot(ba, bc)

    # Cosinus sudut
    cosine = dot_product / (norm_ba * norm_bc)

    # Membatasi nilai agar aman dari error numerik
    cosine = np.clip(cosine, -1.0, 1.0)

    # Mengubah radian menjadi derajat
    angle = np.degrees(np.arccos(cosine))

    return float(angle)


# Uji fungsi dengan tiga titik
if __name__ == "__main__":
    hip = (0, 0)
    knee = (1, 0)
    ankle = (1, 1)

    angle = calculate_angle(hip, knee, ankle)

    print("Sudut lutut:", angle, "derajat")
