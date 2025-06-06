import os
import cv2
import numpy as np
from utils.landmarks import detect_landmarks
from utils.verification import calculate_biometric_vector, compare_vectors
from utils.visualization import visualize_landmarks, plot_differences, draw_face_mesh

# Configuración
ASSETS_DIR = "assets"
RESULTS_DIR = "results"
THRESHOLD = 0.5  # Umbral de similitud

def main():
    # Cargar imágenes
    img1_path = os.path.join(ASSETS_DIR, "img1.jpeg")
    img2_path = os.path.join(ASSETS_DIR, "img2.jpeg")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("Error: No se pudieron cargar las imágenes.")
        return

    # Detectar landmarks
    landmarks1 = detect_landmarks(img1)
    landmarks2 = detect_landmarks(img2)

    # Calcular vectores biométricos
    vector1 = calculate_biometric_vector(landmarks1)
    vector2 = calculate_biometric_vector(landmarks2)

    # Comparar vectores
    similarity, is_same_person = compare_vectors(vector1, vector2, threshold=THRESHOLD)

    # Visualizar resultados
    overlay1 = draw_face_mesh(img1.copy(), landmarks1)
    overlay2 = draw_face_mesh(img2.copy(), landmarks2)
    overlay1 = visualize_landmarks(overlay1, landmarks1)
    overlay2 = visualize_landmarks(overlay2, landmarks2)
    plot_differences(vector1, vector2, os.path.join(RESULTS_DIR, "comparacion_1.png"))

    # Mostrar conclusiones
    conclusion = "Misma persona" if is_same_person else "Persona diferente"
    print(f"Conclusión: {conclusion} (Similitud: {similarity:.2f})")

    # Guardar resultados
    cv2.imwrite(os.path.join(RESULTS_DIR, "overlay1.jpg"), overlay1)
    cv2.imwrite(os.path.join(RESULTS_DIR, "overlay2.jpg"), overlay2)

if __name__ == "__main__":
    main()
