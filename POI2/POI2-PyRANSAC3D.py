import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

from pyransac3d import Plane

# Funkcja do odczytu współrzędnych z pliku tekstowego
def read_coordinates_from_file(file_path):
    with open(file_path, 'r') as file:
        coords = []
        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if line and all(char.isdigit() or char in '. ,' for char in line):
                try:
                    coords.append([float(val) for val in line.split(',')])
                except ValueError:
                    print(f"Nieprawidłowa wartość {line_num}: {line}")
        return np.array(coords)

# Funkcja do wyboru pliku z interfejsu użytkownika
def choose_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Wybierz plik")
    return file_path

# Funkcja do dopasowania płaszczyzny za pomocą algorytmu RANSAC
def fit_plane_ransac(coords):
    plane = Plane()
    best_eq, best_inliers = plane.fit(coords, thresh=0.01, minPoints=100, maxIteration=1000)
    return best_eq, best_inliers

# Funkcja sprawdzająca czy płaszczyzna jest pozioma na podstawie wektora normalnego
def is_plane_horizontal(normal_vector):
    # Płaszczyzna jest uważana za poziomą, jeśli jej wektor normalny ma małe składowe X i Y w porównaniu z Z
    return abs(normal_vector[0]) < 0.1 and abs(normal_vector[1]) < 0.1

# Funkcja do wizualizacji płaszczyzny
def visualize_plane(coords, best_eq):
    # Tworzenie wykresu chmury punktów
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], color='b')

    # Wykreślenie dopasowanej płaszczyzny
    point  = np.array([0, 0, best_eq[3]])  # Punkt na płaszczyźnie
    normal = np.array(best_eq[:3])  # Wektor normalny do płaszczyzny
    # Utworzenie meshgrid punktów
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    if np.isclose(normal[0], 0, atol=1e-6) and np.isclose(normal[2], 0, atol=1e-6):
        Y_const = normal[2] / normal[1]
        X,Z = np.meshgrid(np.linspace(xlim[0], xlim[1],10), np.linspace(zlim[0], zlim[1],10))
        Y = Y_const * np.ones_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.5, color='r')
    else:
        X,Y = np.meshgrid(np.linspace(xlim[0], xlim[1],10), np.linspace(ylim[0], ylim[1],10))
        Z = (-normal[0] * X - normal[1] * Y - best_eq[3]) * 1. /normal[2]
        ax.plot_surface(X, Y, Z, color='r', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Funkcja do analizy płaszczyzny
def analyze_plane(coords):
    best_eq, best_inliers = fit_plane_ransac(coords)
    normal_vector = best_eq[:3]
    print("Wektor normalny do płaszczyzny", normal_vector)

    if len(best_inliers) > 0.8 * len(coords):  # Dostosuj ten próg według potrzeb
        print("Chmura punktów jest płaszczyzną")
        if is_plane_horizontal(normal_vector):
            print("Chmura punktów jest płaszczyzną horyzontalną.")
        else:
            print("Chmura punktów jest płaszczyzną pionową")
        visualize_plane(coords, best_eq)
    else:
        print("Chmura punktów nie jest płaszczyzną lub jest cylindryczna")

# Funkcja główna programu
def main():
    file_path = choose_file()
    coords = read_coordinates_from_file(file_path)
    analyze_plane(coords)

# Uruchomienie funkcji głównej
if __name__ == "__main__":
    main()