import os
import numpy as np
import pandas as pd
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
import tkinter as tk
from tkinter import filedialog

# Funkcja do wyboru katalogu przez użytkownika
def select_folder(prompt):
    root = tk.Tk()
    root.withdraw()  # Ukrycie głównego okna
    folder_selected = filedialog.askdirectory(title=prompt)
    root.destroy()
    return folder_selected

# Funkcja do wczytywania obrazów i wycinania próbek tekstury
def load_and_cut_images(directory, size):
    images = []

    for filename in os.listdir(directory):
        img = io.imread(os.path.join(directory, filename))
        for i in range(0, img.shape[0], size):
            for j in range(0, img.shape[1], size):
                if i + size <= img.shape[0] and j + size <= img.shape[1]:
                    images.append(img[i:i + size, j:j + size])

    return images

# Funkcja do obliczania cech tekstury
def texture_features(images, distances, angles, properties):
    features = []

    for img in images:
        gray = color.rgb2gray(img)
        gray = img_as_ubyte(gray)
        gray //= 4
        glcm = graycomatrix(gray, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)
        feature_vector = []
        for prop in properties:
            feature_vector.extend([graycoprops(glcm, prop).ravel()])
        features.append(np.concatenate(feature_vector))
    return np.array(features)

# Wczytywanie i przetwarzanie obrazów
texture_dirs = [select_folder(f'Wybierz katalog dla tekstury {i + 1}') for i in range(3)]
all_features = []
labels = []
for texture_dir in texture_dirs:
    images = load_and_cut_images(texture_dir, 128)
    features = texture_features(images, [1, 3, 5], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM'])
    all_features.append(features)
    labels.extend([os.path.basename(texture_dir)] * len(features))

all_features = np.vstack(all_features)
df = pd.DataFrame(all_features)
df['label'] = labels
df.to_csv('texture_features.csv', index=True)