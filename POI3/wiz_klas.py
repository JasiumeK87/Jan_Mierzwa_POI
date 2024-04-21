import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Wczytanie cech z pliku CSV
features = pd.read_csv('texture_features.csv', sep=',')

# WIZUALIZACJA
data = np.array(features)
X = data[:, :-1].astype('float64')
Y = data[:, -1]
x_transform = PCA(n_components=3)
Xt = x_transform.fit_transform(X)
red = Y == 'drewno'
blue = Y == 'gresik'
cyan = Y == 'tynk'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xt[red, 0], Xt[red, 1], Xt[red, 2], c="r", alpha=0.5)
ax.scatter(Xt[blue, 0], Xt[blue, 1], Xt[blue, 2], c="b", alpha=0.5)
ax.scatter(Xt[cyan, 0], Xt[cyan, 1], Xt[cyan, 2], c="c",  alpha=0.5)
plt.show()

# KLASYFIKACJA
classifier = svm.SVC(gamma='auto')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

cm = confusion_matrix(y_test, y_pred, normalize="true")
print("Confusion Matrix:")
print(cm)

# Wizualizacja macierzy pomyłek za pomocą biblioteki seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, cmap="Blues", fmt=".2f", xticklabels=['drewno', 'gresik', 'tynk'], yticklabels=['drewno', 'gresik', 'tynk'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()