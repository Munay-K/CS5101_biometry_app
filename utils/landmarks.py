import mediapipe as mp
import cv2
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh

# Índices clave de MediaPipe FaceMesh
LANDMARK_INDICES = {
    # Ojos
    'left_eye_inner': 133,
    'left_eye_outer': 33,
    'left_eye_top': 159,
    'left_eye_bottom': 145,
    'right_eye_inner': 362,
    'right_eye_outer': 263,
    'right_eye_top': 386,
    'right_eye_bottom': 374,
    
    # Cejas
    'left_eyebrow_inner': 46,
    'left_eyebrow_outer': 55,
    'right_eyebrow_inner': 285,
    'right_eyebrow_outer': 276,
    
    # Nariz
    'nose_tip': 1,
    'nose_bridge': 6,
    'nose_left': 129,
    'nose_right': 358,
    'nose_bottom': 94,
    
    # Boca
    'mouth_left': 61,
    'mouth_right': 291,
    'mouth_top': 13,
    'mouth_bottom': 14,
    'upper_lip': 12,
    'lower_lip': 15,
    
    # Mentón y contorno
    'chin': 152,
    'jaw_left': 172,
    'jaw_right': 397,
    'forehead': 9,
    
    # Pómulos
    'cheek_left': 36,
    'cheek_right': 266
}

def detect_landmarks(image):
    """Detecta landmarks faciales en una imagen usando MediaPipe."""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            raise ValueError("No se detectaron rostros en la imagen.")
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        
        # Convertir a coordenadas de píxeles
        landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        
        return landmark_points
    finally:
        face_mesh.close()

def extract_geometric_features(landmarks):
    """Extrae características geométricas avanzadas de los landmarks faciales."""
    features = []
    
    # 1. Distancias normalizadas por el ancho de la cara
    face_width = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['jaw_left']]) - 
        np.array(landmarks[LANDMARK_INDICES['jaw_right']])
    )
    
    if face_width == 0:
        face_width = 1  # Evitar división por cero
    
    # Agregar más medidas de normalización
    nose_to_chin = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['nose_tip']]) - 
        np.array(landmarks[LANDMARK_INDICES['chin']])
    )
    
    # Distancias oculares con más detalle
    eye_distance = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['left_eye_outer']]) - 
        np.array(landmarks[LANDMARK_INDICES['right_eye_outer']])
    ) / face_width
    
    inter_eye_distance = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['left_eye_inner']]) - 
        np.array(landmarks[LANDMARK_INDICES['right_eye_inner']])
    ) / face_width
    
    left_eye_width = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['left_eye_inner']]) - 
        np.array(landmarks[LANDMARK_INDICES['left_eye_outer']])
    ) / face_width
    
    right_eye_width = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['right_eye_inner']]) - 
        np.array(landmarks[LANDMARK_INDICES['right_eye_outer']])
    ) / face_width
    
    # Distancias verticales de ojos
    left_eye_height = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['left_eye_top']]) - 
        np.array(landmarks[LANDMARK_INDICES['left_eye_bottom']])
    ) / face_width
    
    right_eye_height = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['right_eye_top']]) - 
        np.array(landmarks[LANDMARK_INDICES['right_eye_bottom']])
    ) / face_width
    
    # Distancias de nariz
    nose_length = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['nose_bridge']]) - 
        np.array(landmarks[LANDMARK_INDICES['nose_tip']])
    ) / face_width
    
    nose_width = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['nose_left']]) - 
        np.array(landmarks[LANDMARK_INDICES['nose_right']])
    ) / face_width
    
    # Distancias de boca
    mouth_width = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['mouth_left']]) - 
        np.array(landmarks[LANDMARK_INDICES['mouth_right']])
    ) / face_width
    
    mouth_height = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['mouth_top']]) - 
        np.array(landmarks[LANDMARK_INDICES['mouth_bottom']])
    ) / face_width
    
    # Distancias verticales
    face_height = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['forehead']]) - 
        np.array(landmarks[LANDMARK_INDICES['chin']])
    ) / face_width
    
    # 2. Proporciones faciales
    eye_to_mouth_ratio = eye_distance / mouth_width if mouth_width > 0 else 0
    nose_to_face_ratio = nose_length / face_height if face_height > 0 else 0
    eye_aspect_ratio = (left_eye_width + right_eye_width) / (2 * eye_distance) if eye_distance > 0 else 0
    
    # 3. Asimetría facial
    left_eye_to_nose = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['left_eye_outer']]) - 
        np.array(landmarks[LANDMARK_INDICES['nose_tip']])
    )
    right_eye_to_nose = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['right_eye_outer']]) - 
        np.array(landmarks[LANDMARK_INDICES['nose_tip']])
    )
    eye_asymmetry = abs(left_eye_to_nose - right_eye_to_nose) / face_width
    
    left_mouth_to_nose = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['mouth_left']]) - 
        np.array(landmarks[LANDMARK_INDICES['nose_tip']])
    )
    right_mouth_to_nose = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['mouth_right']]) - 
        np.array(landmarks[LANDMARK_INDICES['nose_tip']])
    )
    mouth_asymmetry = abs(left_mouth_to_nose - right_mouth_to_nose) / face_width
    
    # 4. Ángulos faciales
    def calculate_angle(p1, p2, p3):
        """Calcula el ángulo en p2 formado por p1-p2-p3."""
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cosine = np.clip(cosine, -1, 1)  # Evitar errores de dominio
        return np.arccos(cosine)
    
    # Ángulo de los ojos con respecto a la nariz
    eye_nose_angle = calculate_angle(
        landmarks[LANDMARK_INDICES['left_eye_outer']],
        landmarks[LANDMARK_INDICES['nose_tip']],
        landmarks[LANDMARK_INDICES['right_eye_outer']]
    )
    
    # Ángulo de la mandíbula
    jaw_angle = calculate_angle(
        landmarks[LANDMARK_INDICES['jaw_left']],
        landmarks[LANDMARK_INDICES['chin']],
        landmarks[LANDMARK_INDICES['jaw_right']]
    )
    
    # 5. Características adicionales
    eyebrow_distance = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['left_eyebrow_inner']]) - 
        np.array(landmarks[LANDMARK_INDICES['right_eyebrow_inner']])
    ) / face_width
    
    cheek_width = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['cheek_left']]) - 
        np.array(landmarks[LANDMARK_INDICES['cheek_right']])
    ) / face_width
    
    # Distancia entre ojos y cejas
    left_eye_eyebrow_dist = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['left_eye_top']]) - 
        np.array(landmarks[LANDMARK_INDICES['left_eyebrow_inner']])
    ) / face_width
    
    # Compilar todas las características (expandido)
    features = np.array([
        eye_distance,
        inter_eye_distance,
        left_eye_width,
        right_eye_width,
        left_eye_height,
        right_eye_height,
        nose_length,
        nose_width,
        mouth_width,
        mouth_height,
        face_height,
        eye_to_mouth_ratio,
        nose_to_face_ratio,
        eye_aspect_ratio,
        eye_asymmetry,
        mouth_asymmetry,
        eye_nose_angle,
        jaw_angle,
        eyebrow_distance,
        cheek_width,
        left_eye_eyebrow_dist,
        # Nuevas características
        nose_to_chin / face_width,
        inter_eye_distance / eye_distance if eye_distance > 0 else 0,
        (left_eye_height + right_eye_height) / (left_eye_width + right_eye_width) if (left_eye_width + right_eye_width) > 0 else 0
    ])
    
    return features

def get_feature_labels():
    """Retorna las etiquetas de las características para visualización."""
    return [
        "Distancia entre ojos",
        "Distancia inter-ocular",
        "Ancho ojo izquierdo",
        "Ancho ojo derecho",
        "Altura ojo izquierdo",
        "Altura ojo derecho",
        "Longitud de nariz",
        "Ancho de nariz",
        "Ancho de boca",
        "Altura de boca",
        "Altura facial",
        "Razón ojos/boca",
        "Razón nariz/cara",
        "Razón aspecto ojos",
        "Asimetría ocular",
        "Asimetría bucal",
        "Ángulo ojos-nariz",
        "Ángulo mandibular",
        "Distancia entre cejas",
        "Ancho de pómulos",
        "Dist. ojo-ceja izq.",
        "Dist. nariz-mentón",
        "Razón inter/extra ocular",
        "Aspecto altura/ancho ojos"
    ]