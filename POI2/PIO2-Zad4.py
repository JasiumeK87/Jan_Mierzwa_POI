import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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
        print("Nie wybrano pliku.")
        return

    try:
        x_coords, y_coords, z_coords = read_coordinates_from_file(file_path)
        points = np.column_stack((x_coords, y_coords, z_coords))
        kmeans = KMeans(n_clusters=3)
        labels = kmeans.fit_predict(points)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(3):
            cluster_points = points[labels == i]
            x_cluster, y_cluster, z_cluster = cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2]

            if (np.all(x_cluster) and np.all(y_cluster) and np.all(z_cluster)):
                print(f"cloud {i} jest płaszczyzną cylindryczną")
                ax.scatter(x_cluster, y_cluster, z_cluster, color=np.random.rand(3,), marker='o', label=f'cloud {i}')
                ax.set_title('Chmury z klasteryzacją k-średnich')
                continue

            best_plane, inliers, distances = fit_plane_ransac(x_cluster, y_cluster, z_cluster)

            if best_plane is not None:
                A, B, C, D = best_plane
                normal_vector = np.array(best_plane[:3])
                print(f"cloud {i} - wektor normalny do płaszczyzny: {normal_vector}")
                plane_eq = lambda p: A * p[0] + B * p[1] + C * p[2] + D
                avg_distance = np.mean(distances)
                print(f"cloud {i} - średnia odległość do płaszczyzny: {avg_distance}")

                if avg_distance == 0:
                    print(f"cloud {i} jest płaszczyzną.")
                else:
                    print(f"cloud {i} nie jest płaszczyzną.")

                if np.isclose(A, 0, atol=1e-6) and np.isclose(C, 0, atol=1e-6):
                    # Płaszczyzna pionowa równoległa do osi Z (Y=constant), A = 0 i C = 0
                    Y_const = -D / B
                    xx, zz = np.meshgrid(np.linspace(x_cluster.min(), x_cluster.max(), 10),
                                         np.linspace(z_cluster.min(), z_cluster.max(), 10))
                    yy = Y_const * np.ones_like(xx)
                    ax.plot_surface(xx, yy, zz, alpha=0.5, color=np.random.rand(3,), label='Płaszczyzna pionowa')
                else:
                    xx, yy = np.meshgrid(np.linspace(x_cluster.min(), x_cluster.max(), 10),
                                         np.linspace(y_cluster.min(), y_cluster.max(), 10))
                    zz = (-D - A * xx - B * yy) / C
                    ax.plot_surface(xx, yy, zz, alpha=0.5, color=np.random.rand(3,))

            ax.scatter(x_cluster, y_cluster, z_cluster, color=np.random.rand(3,), marker='o', label=f'cloud {i}')
            ax.set_title('Chmury punktów 3D z klasteryzacją i dopasowanymi płaszczyznami k-średnich')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    except FileNotFoundError:
        print("404 Not Found.")


if __name__ == "__main__":
    main()