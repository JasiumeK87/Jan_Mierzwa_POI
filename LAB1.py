from scipy.stats import norm, uniform
from csv import writer
import numpy as np

def generate_points_horizontal(num_points: int = 2000, width: float = 100, length: float = 100, height: float = 20):
    distribution_x = norm(loc=0, scale=width)
    distribution_y = norm(loc=0, scale=length)
    distribution_z = norm(loc=0, scale=0)  # Dystrybucja dla osi z w poziomie

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points

def generate_points_vertical(num_points: int = 2000, width: float = 100, length: float = 100, height: float = 20):
    distribution_x = norm(loc=0, scale=width)
    distribution_y = norm(loc=0, scale=0)  # Dystrybucja dla osi y w pionie
    distribution_z = norm(loc=0, scale=height)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points

def generate_cylindrical_surface_points(num_points: int = 2000, radius: float = 10, height: float = 50):
    # Generowanie punktów na powierzchni cylindra
    distribution_theta = uniform(loc=0, scale=2 * np.pi)  # Kąt  (od 0 do 2pi)

    theta = distribution_theta.rvs(size=num_points)
    z = uniform(loc=0, scale=height).rvs(size=num_points)  # Wysokość z rozkładu jednostajnego

    # Konwersja współrzędnych cylindrycznych na kartezjańskie
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    points = zip(x, y, z)
    return points

if __name__ == '__main__':
    cloud_points = generate_points_vertical(num_points=2000, width=100, length=100, height=20)
    with open('cloud_vertical.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        csvwriter.writerow(['x', 'y', 'z'])
        for p in cloud_points:
            csvwriter.writerow(p)

    cloud_points2 = generate_points_horizontal(num_points=2000, width=100, length=100, height=20)
    with open('cloud_horizontal.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        csvwriter.writerow(['x', 'y', 'z'])
        for l in cloud_points2:
            csvwriter.writerow(l)
    cloud_points3 = generate_cylindrical_surface_points(num_points=2000, radius=10, height=20)
    with open('cloud_cyli.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        csvwriter.writerow(['x', 'y', 'z'])
        for i in cloud_points3:
            csvwriter.writerow(i)