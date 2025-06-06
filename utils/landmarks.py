import mediapipe as mp
import cv2
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh.FaceMesh

def detect_landmarks(image):
    """Detecta landmarks faciales en una imagen usando MediaPipe."""
    with mp_face_mesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise ValueError("No se detectaron rostros en la imagen.")

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

def extract_geometric_features(landmarks):
    """Extrae características geométricas a partir de los landmarks faciales."""
    # Distancias clave
    eye_distance = np.linalg.norm(np.array(landmarks[33]) - np.array(landmarks[263]))
    mouth_width = np.linalg.norm(np.array(landmarks[61]) - np.array(landmarks[291]))
    mouth_height = np.linalg.norm(np.array(landmarks[13]) - np.array(landmarks[14]))
    nose_height = np.linalg.norm(np.array(landmarks[1]) - np.array(landmarks[168]))

    # Razones geométricas
    nose_to_mouth_ratio = nose_height / mouth_width if mouth_width != 0 else 0
    eye_to_nose_ratio = eye_distance / nose_height if nose_height != 0 else 0

    # Ángulos básicos
    def calculate_angle(p1, p2, p3):
        """Calcula el ángulo formado por tres puntos."""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.arccos(cosine_angle)

    eye_nose_mouth_angle = calculate_angle(landmarks[33], landmarks[1], landmarks[61])

    # Eye Aspect Ratio (EAR)
    def calculate_ear(eye_indices):
        """Calcula el Eye Aspect Ratio para un ojo."""
        vertical1 = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
        vertical2 = np.linalg.norm(np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]]))
        horizontal = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
        return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal != 0 else 0

    left_eye_ear = calculate_ear([33, 159, 158, 133, 153, 144])
    right_eye_ear = calculate_ear([263, 386, 385, 362, 387, 373])

    # Simetría izquierda/derecha
    symmetry = np.mean([np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[mirror_index]))
                        for i, mirror_index in zip(range(0, len(landmarks) // 2), range(len(landmarks) // 2, len(landmarks)))])

    # Vector de características
    features = np.array([
        eye_distance, mouth_width, mouth_height, nose_height,
        nose_to_mouth_ratio, eye_to_nose_ratio, eye_nose_mouth_angle,
        left_eye_ear, right_eye_ear, symmetry
    ])
    return features
