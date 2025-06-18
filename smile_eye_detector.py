import numpy as np
import cv2
from utils.landmarks import LANDMARK_INDICES

def ensure_numpy_array(point):
    """Convierte un punto a numpy array si no lo es."""
    return np.array(point) if not isinstance(point, np.ndarray) else point

class SmileEyeDetector:
    def __init__(self):
        # Umbrales ajustables
        self.smile_threshold = 0.35  # Ratio mínimo para considerar sonrisa
        self.eye_open_threshold = 0.15  # EAR mínimo para ojo abierto
        self.mouth_curve_threshold = 15  # Grados mínimos de curvatura
        
    def detect_smile(self, landmarks):
        """
        Detecta si hay una sonrisa en base a los landmarks faciales.
        
        Returns:
            dict: {
                'is_smiling': bool,
                'confidence': float (0-1),
                'metrics': dict con métricas detalladas
            }
        """
        # Convertir todos los landmarks necesarios a numpy arrays
        # 1. Calcular ancho de la boca
        mouth_left = ensure_numpy_array(landmarks[LANDMARK_INDICES['mouth_left']])
        mouth_right = ensure_numpy_array(landmarks[LANDMARK_INDICES['mouth_right']])
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        
        # 2. Calcular altura de la boca
        mouth_top = ensure_numpy_array(landmarks[LANDMARK_INDICES['mouth_top']])
        mouth_bottom = ensure_numpy_array(landmarks[LANDMARK_INDICES['mouth_bottom']])
        mouth_height = np.linalg.norm(mouth_bottom - mouth_top)
        
        # 3. Calcular ratio de apertura
        mouth_ratio = mouth_width / mouth_height if mouth_height > 0 else 0
        
        # 4. Calcular curvatura de las comisuras
        # Punto medio de la boca
        mouth_center = (mouth_left + mouth_right) / 2
        
        # Vector desde el centro a cada comisura
        left_vector = mouth_left - mouth_center
        right_vector = mouth_right - mouth_center
        
        # Ángulo de elevación de las comisuras
        left_angle = np.arctan2(left_vector[1], left_vector[0]) * 180 / np.pi
        right_angle = np.arctan2(right_vector[1], right_vector[0]) * 180 / np.pi
        
        # Promedio de curvatura (negativo = hacia arriba = sonrisa)
        avg_curve = -(left_angle + right_angle) / 2
        
        # 5. Distancia entre labios (para sonrisas con dientes)
        upper_lip = ensure_numpy_array(landmarks[LANDMARK_INDICES['upper_lip']])
        lower_lip = ensure_numpy_array(landmarks[LANDMARK_INDICES['lower_lip']])
        lip_distance = np.linalg.norm(upper_lip - lower_lip)
        
        # Normalizar por el ancho de la boca
        lip_separation_ratio = lip_distance / mouth_width if mouth_width > 0 else 0
        
        # 6. Calcular métricas adicionales
        # Distancia de las comisuras a la nariz (sonrisas tienden a acercarlas)
        nose_tip = ensure_numpy_array(landmarks[LANDMARK_INDICES['nose_tip']])
        left_to_nose = np.linalg.norm(mouth_left - nose_tip)
        right_to_nose = np.linalg.norm(mouth_right - nose_tip)
        
        # 7. Determinar si hay sonrisa
        smile_score = 0
        
        # Factor 1: Ratio de boca (más ancha = más sonrisa)
        if mouth_ratio > 3.5:
            smile_score += 0.3
        elif mouth_ratio > 2.8:
            smile_score += 0.2
            
        # Factor 2: Curvatura hacia arriba
        if avg_curve > self.mouth_curve_threshold:
            smile_score += 0.3
        elif avg_curve > self.mouth_curve_threshold / 2:
            smile_score += 0.15
            
        # Factor 3: Separación de labios
        if lip_separation_ratio > 0.1:
            smile_score += 0.2
            
        # Factor 4: Elevación de comisuras respecto al centro
        if mouth_left[1] < mouth_center[1] and mouth_right[1] < mouth_center[1]:
            smile_score += 0.2
            
        # Normalizar score
        smile_score = min(smile_score, 1.0)
        is_smiling = smile_score >= self.smile_threshold
        
        return {
            'is_smiling': is_smiling,
            'confidence': smile_score,
            'metrics': {
                'mouth_ratio': mouth_ratio,
                'mouth_curve_angle': avg_curve,
                'lip_separation': lip_separation_ratio,
                'mouth_width': mouth_width,
                'mouth_height': mouth_height
            }
        }
    
    def detect_eyes_open(self, landmarks):
        """
        Detecta si los ojos están abiertos usando Eye Aspect Ratio (EAR).
        
        Returns:
            dict: {
                'left_eye_open': bool,
                'right_eye_open': bool,
                'both_eyes_open': bool,
                'confidence': float (0-1),
                'metrics': dict con métricas detalladas
            }
        """
        def calculate_ear(eye_points):
            """Calcula Eye Aspect Ratio para un ojo."""
            # Convertir todos los puntos a numpy arrays
            eye_points = [np.array(p) if not isinstance(p, np.ndarray) else p for p in eye_points]
            
            # Distancias verticales
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])  # Superior-inferior izq
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])  # Superior-inferior der
            
            # Distancia horizontal
            h = np.linalg.norm(eye_points[0] - eye_points[3])
            
            # EAR formula
            ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
            return ear
        
        # Obtener puntos de los ojos
        left_eye_points = [
            np.array(landmarks[LANDMARK_INDICES['left_eye_outer']]),
            np.array(landmarks[LANDMARK_INDICES['left_eye_top']]),
            np.array(landmarks[159]),  # Punto superior adicional
            np.array(landmarks[LANDMARK_INDICES['left_eye_inner']]),
            np.array(landmarks[145]),  # Punto inferior adicional
            np.array(landmarks[LANDMARK_INDICES['left_eye_bottom']])
        ]
        
        right_eye_points = [
            np.array(landmarks[LANDMARK_INDICES['right_eye_inner']]),
            np.array(landmarks[LANDMARK_INDICES['right_eye_top']]),
            np.array(landmarks[386]),  # Punto superior adicional
            np.array(landmarks[LANDMARK_INDICES['right_eye_outer']]),
            np.array(landmarks[374]),  # Punto inferior adicional
            np.array(landmarks[LANDMARK_INDICES['right_eye_bottom']])
        ]
        
        # Calcular EAR para cada ojo
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        
        # Calcular apertura adicional (distancia directa superior-inferior)
        # Convertir a numpy arrays si son tuplas
        left_top = np.array(landmarks[LANDMARK_INDICES['left_eye_top']])
        left_bottom = np.array(landmarks[LANDMARK_INDICES['left_eye_bottom']])
        right_top = np.array(landmarks[LANDMARK_INDICES['right_eye_top']])
        right_bottom = np.array(landmarks[LANDMARK_INDICES['right_eye_bottom']])
        
        left_eye_height = np.linalg.norm(left_top - left_bottom)
        right_eye_height = np.linalg.norm(right_top - right_bottom)
        
        # Normalizar por el ancho del ojo
        left_inner = np.array(landmarks[LANDMARK_INDICES['left_eye_inner']])
        left_outer = np.array(landmarks[LANDMARK_INDICES['left_eye_outer']])
        right_inner = np.array(landmarks[LANDMARK_INDICES['right_eye_inner']])
        right_outer = np.array(landmarks[LANDMARK_INDICES['right_eye_outer']])
        
        left_eye_width = np.linalg.norm(left_inner - left_outer)
        right_eye_width = np.linalg.norm(right_inner - right_outer)
        
        left_aperture_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0
        right_aperture_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0
        
        # Determinar si están abiertos
        left_open = left_ear > self.eye_open_threshold
        right_open = right_ear > self.eye_open_threshold
        
        # Calcular confianza
        avg_ear = (left_ear + right_ear) / 2
        confidence = min(avg_ear / 0.3, 1.0)  # Normalizar a 0-1
        
        return {
            'left_eye_open': left_open,
            'right_eye_open': right_open,
            'both_eyes_open': left_open and right_open,
            'confidence': confidence,
            'metrics': {
                'left_ear': left_ear,
                'right_ear': right_ear,
                'avg_ear': avg_ear,
                'left_aperture': left_aperture_ratio,
                'right_aperture': right_aperture_ratio
            }
        }
    
    def analyze_expression(self, landmarks):
        """
        Análisis completo de la expresión facial.
        
        Returns:
            dict con análisis completo de sonrisa y ojos
        """
        smile_result = self.detect_smile(landmarks)
        eyes_result = self.detect_eyes_open(landmarks)
        
        # Clasificación general
        if smile_result['is_smiling'] and eyes_result['both_eyes_open']:
            expression = "Sonriente con ojos abiertos"
            quality_score = (smile_result['confidence'] + eyes_result['confidence']) / 2
        elif smile_result['is_smiling'] and not eyes_result['both_eyes_open']:
            expression = "Sonriente con ojos cerrados/entrecerrados"
            quality_score = smile_result['confidence'] * 0.7
        elif not smile_result['is_smiling'] and eyes_result['both_eyes_open']:
            expression = "Neutral con ojos abiertos"
            quality_score = eyes_result['confidence'] * 0.8
        else:
            expression = "Neutral con ojos cerrados"
            quality_score = 0.3
        
        return {
            'expression': expression,
            'quality_score': quality_score,
            'smile': smile_result,
            'eyes': eyes_result,
            'is_ideal': smile_result['is_smiling'] and eyes_result['both_eyes_open']
        }
    
    def visualize_detection(self, image, landmarks, analysis_result):
        """
        Visualiza los resultados de detección en la imagen.
        """
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        # Colores
        smile_color = (0, 255, 0) if analysis_result['smile']['is_smiling'] else (0, 0, 255)
        eye_color = (0, 255, 0) if analysis_result['eyes']['both_eyes_open'] else (0, 0, 255)
        
        # Dibujar contorno de boca
        mouth_points = [
            landmarks[LANDMARK_INDICES['mouth_left']],
            landmarks[LANDMARK_INDICES['mouth_top']],
            landmarks[LANDMARK_INDICES['mouth_right']],
            landmarks[LANDMARK_INDICES['mouth_bottom']]
        ]
        # Convertir a numpy array de enteros
        mouth_points = np.array([ensure_numpy_array(p) for p in mouth_points], np.int32)
        cv2.polylines(img_copy, [mouth_points], True, smile_color, 2)
        
        # Dibujar contorno de ojos
        for eye_indices in [['left_eye_inner', 'left_eye_top', 'left_eye_outer', 'left_eye_bottom'],
                           ['right_eye_inner', 'right_eye_top', 'right_eye_outer', 'right_eye_bottom']]:
            eye_points = [landmarks[LANDMARK_INDICES[idx]] for idx in eye_indices]
            eye_points = np.array([ensure_numpy_array(p) for p in eye_points], np.int32)
            cv2.polylines(img_copy, [eye_points], True, eye_color, 2)
        
        # Añadir texto con resultados
        y_offset = 30
        cv2.putText(img_copy, f"Expression: {analysis_result['expression']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        smile_text = f"Smile: {'Yes' if analysis_result['smile']['is_smiling'] else 'No'} ({analysis_result['smile']['confidence']:.2f})"
        cv2.putText(img_copy, smile_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, smile_color, 2)
        
        y_offset += 25
        eyes_text = f"Eyes: {'Open' if analysis_result['eyes']['both_eyes_open'] else 'Closed'} ({analysis_result['eyes']['confidence']:.2f})"
        cv2.putText(img_copy, eyes_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        
        y_offset += 25
        quality_text = f"Quality: {analysis_result['quality_score']:.2f}"
        quality_color = (0, 255, 0) if analysis_result['quality_score'] > 0.7 else (0, 165, 255)
        cv2.putText(img_copy, quality_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
        return img_copy