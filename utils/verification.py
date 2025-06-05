import numpy as np

def calculate_biometric_vector(landmarks):
    """Calcula un vector biométrico basado en distancias euclidianas normalizadas."""
    eyes = np.linalg.norm(np.array(landmarks[33]) - np.array(landmarks[263]))
    nose = np.linalg.norm(np.array(landmarks[1]) - np.array(landmarks[168]))
    mouth = np.linalg.norm(np.array(landmarks[61]) - np.array(landmarks[291]))
    chin = np.linalg.norm(np.array(landmarks[152]) - np.array(landmarks[8]))

    # Normalizar distancias
    distances = np.array([eyes, nose, mouth, chin])
    return distances / np.linalg.norm(distances)

def compare_vectors(vector1, vector2, threshold=0.5):
    """Compara dos vectores biométricos y determina si son similares."""
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return similarity, similarity >= threshold
