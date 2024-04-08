import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def read_coordinates_from_file(file_path):
    with open(file_path, 'r') as file:
        next(file)  # Pomijanie pierwszego wiersza z nagłówkami
        coordinates = np.loadtxt(file, delimiter=',', usecols=(0, 1, 2))
    return coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]


def choose_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Wybierz plik")
    return file_path


def fit_plane_ransac(x, y, z, n_iterations=1000, threshold=0.1):
    best_plane = None
    best_inliers = None
    max_inliers = 0
    distances = None  # Zdefiniowanie distances na zewnątrz pętli

    for _ in range(n_iterations):
        indices = np.random.choice(len(x), 3, replace=False)
        points = np.vstack([x[indices], y[indices], z[indices]]).T

        p1, p2, p3 = points
        normal_vector = np.cross(p2 - p1, p3 - p1)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector
        d = -np.dot(normal_vector, p1)

        distances = np.abs(np.dot(points, normal_vector) + d) / np.linalg.norm(normal_vector)

        inliers = np.where(distances < threshold)[0]

        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_inliers = inliers
            best_plane = normal_vector.tolist() + [d]

    return best_plane, best_inliers, distances  # Zwracanie distances


def main():
    file_path = choose_file()
    if not file_path:
        print("404 Not Found.")
        return

    try:
        x_coords, y_coords, z_coords = read_coordinates_from_file(file_path)
        best_plane, inliers, distances = fit_plane_ransac(x_coords, y_coords, z_coords)  # Poprawiony sposób przypisania

        # Wyświetlanie informacji o płaszczyźnie
        A, B, C, D = best_plane
        normal_vector = np.array(best_plane[:3])
        print(f"Wektor normalny do płaszczyzny: {normal_vector}")
        avg_distance = np.mean(distances)
        print(f"Średnia odległość do płaszczyzny: {avg_distance}")

        # Ustalanie wartości eps dla DBSCAN w zależności od płaszczyzny
        eps_value = 10000 if np.isclose(A, 0, atol=1e-6) and np.isclose(B, 0, atol=1e-6) else 1.2
        print("Płaszczyzna pozioma" if np.isclose(A, 0, atol=1e-6) and np.isclose(B, 0, atol=1e-6)
              else "Płaszczyzna pionowa równoległa do osi z." if np.isclose(A, 0, atol=1e-6) and np.isclose(C, 0,
                                                                                                            atol=1e-6)
        else "Płaszczyzna ogólna")

        # Konfiguracja DBSCAN
        dbscan = DBSCAN(eps=eps_value, min_samples=1000)
        labels = dbscan.fit_predict(np.column_stack((x_coords, y_coords, z_coords)))

        # Obliczanie liczby klastrów i punktów szumowych
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"Liczba znalezionych klastrów: {n_clusters}")
        print(f"Liczba punktów szumowych: {n_noise}")

    except FileNotFoundError:
        print("404 Not Found.")


if __name__ == "__main__":
    main()