import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_landmarks(image, landmarks):
    """Dibuja landmarks sobre la imagen."""
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image

def plot_differences(vector1, vector2, output_path):
    """Genera un gráfico de barras con las diferencias entre vectores."""
    labels = ["Ojos", "Nariz", "Boca", "Mentón"]
    differences = np.abs(vector1 - vector2)

    plt.figure(figsize=(8, 6))
    plt.bar(labels, differences, color='skyblue')
    plt.title("Diferencias por segmento facial")
    plt.ylabel("Diferencia normalizada")
    plt.savefig(output_path)
    plt.close()
