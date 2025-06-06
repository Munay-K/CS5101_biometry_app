import cv2
import matplotlib.pyplot as plt
import numpy as np

import mediapipe as mp

# Obtener conexiones estándar de MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh
FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION

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

def compare_landmarks_visual(image1, image2, landmarks1, landmarks2, output_path):
    """Dibuja líneas entre puntos equivalentes de dos rostros para comparar landmarks."""
    if image1 is None or image2 is None or not landmarks1 or not landmarks2:
        print("Error: Imágenes o landmarks no válidos para la comparación.")
        return

    combined_image = np.hstack((image1, image2))
    offset = image1.shape[1]  # Desplazamiento para el segundo rostro
    for (p1, p2) in zip(landmarks1, landmarks2):
        if p1 and p2:
            cv2.line(combined_image, p1, (p2[0] + offset, p2[1]), (0, 0, 255), 1)
    cv2.imwrite(output_path, combined_image)

def draw_face_mesh(image, landmarks):
    """Dibuja una malla facial conectando landmarks."""
    if not landmarks or len(landmarks) < max((max(connection) for connection in FACE_CONNECTIONS), default=0):
        # No dibujar nada si los landmarks no son válidos
        return image

    for connection in FACE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            cv2.line(image, landmarks[start_idx], landmarks[end_idx], (255, 0, 0), 1)
    return image
