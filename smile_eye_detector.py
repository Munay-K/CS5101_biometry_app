import numpy as np
import cv2
from utils.landmarks import LANDMARK_INDICES
import mediapipe as mp

def ensure_numpy_array(point):
    """Convierte un punto a numpy array si no lo es."""
    return np.array(point) if not isinstance(point, np.ndarray) else point

class MultiFaceSmileEyeDetector:
    def __init__(self):
        # Umbrales ajustables
        self.smile_threshold = 0.35  
        self.eye_open_threshold = 0.15  
        self.mouth_curve_threshold = 15  
        
        # Configurar MediaPipe para múltiples caras
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,  # Detectar hasta 10 caras
            refine_landmarks=True,
            min_detection_confidence=0.2
        )
        
    def detect_multiple_faces(self, image):
        """
        Detecta múltiples caras en una imagen.
        
        Returns:
            list: Lista de landmarks para cada cara detectada
        """
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return []
            
            h, w, _ = image.shape
            all_faces_landmarks = []
            
            # Procesar cada cara detectada
            for face_landmarks in results.multi_face_landmarks:
                # Convertir a coordenadas de píxeles
                landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                all_faces_landmarks.append(landmark_points)
            
            return all_faces_landmarks
            
        except Exception as e:
            print(f"Error al detectar caras: {e}")
            return []
    
    def detect_smile(self, landmarks):
        """
        Detecta si hay una sonrisa en base a los landmarks faciales.
        (Misma lógica que antes, pero ahora se puede llamar para múltiples caras)
        """
        try:
            # Convertir todos los landmarks necesarios a numpy arrays
            mouth_left = ensure_numpy_array(landmarks[LANDMARK_INDICES['mouth_left']])
            mouth_right = ensure_numpy_array(landmarks[LANDMARK_INDICES['mouth_right']])
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            
            mouth_top = ensure_numpy_array(landmarks[LANDMARK_INDICES['mouth_top']])
            mouth_bottom = ensure_numpy_array(landmarks[LANDMARK_INDICES['mouth_bottom']])
            mouth_height = np.linalg.norm(mouth_bottom - mouth_top)
            
            mouth_ratio = mouth_width / mouth_height if mouth_height > 0 else 0
            
            # Calcular curvatura de las comisuras
            mouth_center = (mouth_left + mouth_right) / 2
            left_vector = mouth_left - mouth_center
            right_vector = mouth_right - mouth_center
            
            left_angle = np.arctan2(left_vector[1], left_vector[0]) * 180 / np.pi
            right_angle = np.arctan2(right_vector[1], right_vector[0]) * 180 / np.pi
            avg_curve = -(left_angle + right_angle) / 2
            
            # Distancia entre labios
            upper_lip = ensure_numpy_array(landmarks[LANDMARK_INDICES['upper_lip']])
            lower_lip = ensure_numpy_array(landmarks[LANDMARK_INDICES['lower_lip']])
            lip_distance = np.linalg.norm(upper_lip - lower_lip)
            lip_separation_ratio = lip_distance / mouth_width if mouth_width > 0 else 0
            
            # Calcular score de sonrisa
            smile_score = 0
            
            if mouth_ratio > 3.5:
                smile_score += 0.3
            elif mouth_ratio > 2.8:
                smile_score += 0.2
                
            if avg_curve > self.mouth_curve_threshold:
                smile_score += 0.3
            elif avg_curve > self.mouth_curve_threshold / 2:
                smile_score += 0.15
                
            if lip_separation_ratio > 0.1:
                smile_score += 0.2
                
            if mouth_left[1] < mouth_center[1] and mouth_right[1] < mouth_center[1]:
                smile_score += 0.2
            
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
        except Exception as e:
            print(f"Error al detectar sonrisa: {e}")
            return {
                'is_smiling': False,
                'confidence': 0.0,
                'metrics': {}
            }
    
    def detect_eyes_open(self, landmarks):
        """
        Detecta si los ojos están abiertos usando Eye Aspect Ratio (EAR).
        (Misma lógica que antes)
        """
        try:
            def calculate_ear(eye_points):
                eye_points = [np.array(p) if not isinstance(p, np.ndarray) else p for p in eye_points]
                v1 = np.linalg.norm(eye_points[1] - eye_points[5])
                v2 = np.linalg.norm(eye_points[2] - eye_points[4])
                h = np.linalg.norm(eye_points[0] - eye_points[3])
                ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
                return ear
            
            # Obtener puntos de los ojos
            left_eye_points = [
                np.array(landmarks[LANDMARK_INDICES['left_eye_outer']]),
                np.array(landmarks[LANDMARK_INDICES['left_eye_top']]),
                np.array(landmarks[159]),
                np.array(landmarks[LANDMARK_INDICES['left_eye_inner']]),
                np.array(landmarks[145]),
                np.array(landmarks[LANDMARK_INDICES['left_eye_bottom']])
            ]
            
            right_eye_points = [
                np.array(landmarks[LANDMARK_INDICES['right_eye_inner']]),
                np.array(landmarks[LANDMARK_INDICES['right_eye_top']]),
                np.array(landmarks[386]),
                np.array(landmarks[LANDMARK_INDICES['right_eye_outer']]),
                np.array(landmarks[374]),
                np.array(landmarks[LANDMARK_INDICES['right_eye_bottom']])
            ]
            
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            
            # Calcular apertura adicional
            left_top = np.array(landmarks[LANDMARK_INDICES['left_eye_top']])
            left_bottom = np.array(landmarks[LANDMARK_INDICES['left_eye_bottom']])
            right_top = np.array(landmarks[LANDMARK_INDICES['right_eye_top']])
            right_bottom = np.array(landmarks[LANDMARK_INDICES['right_eye_bottom']])
            
            left_eye_height = np.linalg.norm(left_top - left_bottom)
            right_eye_height = np.linalg.norm(right_top - right_bottom)
            
            left_inner = np.array(landmarks[LANDMARK_INDICES['left_eye_inner']])
            left_outer = np.array(landmarks[LANDMARK_INDICES['left_eye_outer']])
            right_inner = np.array(landmarks[LANDMARK_INDICES['right_eye_inner']])
            right_outer = np.array(landmarks[LANDMARK_INDICES['right_eye_outer']])
            
            left_eye_width = np.linalg.norm(left_inner - left_outer)
            right_eye_width = np.linalg.norm(right_inner - right_outer)
            
            left_aperture_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0
            right_aperture_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0
            
            left_open = left_ear > self.eye_open_threshold
            right_open = right_ear > self.eye_open_threshold
            
            avg_ear = (left_ear + right_ear) / 2
            confidence = min(avg_ear / 0.3, 1.0)
            
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
        except Exception as e:
            print(f"Error al detectar ojos: {e}")
            return {
                'left_eye_open': False,
                'right_eye_open': False,
                'both_eyes_open': False,
                'confidence': 0.0,
                'metrics': {}
            }
    
    def analyze_expression(self, landmarks):
        """
        Análisis completo de la expresión facial para una cara.
        """
        smile_result = self.detect_smile(landmarks)
        eyes_result = self.detect_eyes_open(landmarks)
        
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
    
    def analyze_multiple_faces(self, image):
        """
        Analiza expresiones de múltiples caras en una imagen.
        
        Returns:
            dict: {
                'num_faces': int,
                'faces': [análisis por cara],
                'summary': resumen general
            }
        """
        # Detectar todas las caras
        all_landmarks = self.detect_multiple_faces(image)
        
        if not all_landmarks:
            return {
                'num_faces': 0,
                'faces': [],
                'summary': 'No se detectaron caras en la imagen'
            }
        
        # Analizar cada cara
        face_analyses = []
        for i, landmarks in enumerate(all_landmarks):
            try:
                analysis = self.analyze_expression(landmarks)
                analysis['face_id'] = i + 1
                
                # Calcular centro de la cara para identificación
                nose_tip = ensure_numpy_array(landmarks[LANDMARK_INDICES['nose_tip']])
                analysis['center'] = tuple(nose_tip.astype(int))
                
                face_analyses.append(analysis)
            except Exception as e:
                print(f"Error al analizar cara {i+1}: {e}")
                continue
        
        # Generar resumen
        num_faces = len(face_analyses)
        smiling_faces = sum(1 for face in face_analyses if face['smile']['is_smiling'])
        open_eyes_faces = sum(1 for face in face_analyses if face['eyes']['both_eyes_open'])
        ideal_faces = sum(1 for face in face_analyses if face['is_ideal'])
        
        summary = f"{num_faces} cara(s) detectada(s). "
        summary += f"{smiling_faces} sonriendo, "
        summary += f"{open_eyes_faces} con ojos abiertos, "
        summary += f"{ideal_faces} con expresión ideal."
        
        return {
            'num_faces': num_faces,
            'faces': face_analyses,
            'summary': summary,
            'statistics': {
                'total_faces': num_faces,
                'smiling_faces': smiling_faces,
                'open_eyes_faces': open_eyes_faces,
                'ideal_faces': ideal_faces,
                'smile_percentage': (smiling_faces / num_faces * 100) if num_faces > 0 else 0,
                'eyes_percentage': (open_eyes_faces / num_faces * 100) if num_faces > 0 else 0,
                'ideal_percentage': (ideal_faces / num_faces * 100) if num_faces > 0 else 0
            }
        }
    
    def visualize_multiple_faces(self, image, analysis_result):
        """
        Visualiza los resultados de detección para múltiples caras.
        """
        img_copy = image.copy()
        
        if analysis_result['num_faces'] == 0:
            cv2.putText(img_copy, "No faces detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img_copy
        
        # Colores para diferentes estados
        colors = {
            'ideal': (0, 255, 0),      # Verde
            'smile_only': (0, 255, 255), # Amarillo
            'eyes_only': (255, 0, 0),    # Azul
            'neutral': (0, 0, 255)       # Rojo
        }
        
        # Re-detectar landmarks para visualización
        all_landmarks = self.detect_multiple_faces(image)
        
        for i, (face_analysis, landmarks) in enumerate(zip(analysis_result['faces'], all_landmarks)):
            # Determinar color según estado
            if face_analysis['is_ideal']:
                color = colors['ideal']
                status = "IDEAL"
            elif face_analysis['smile']['is_smiling']:
                color = colors['smile_only']
                status = "SMILE"
            elif face_analysis['eyes']['both_eyes_open']:
                color = colors['eyes_only']
                status = "EYES"
            else:
                color = colors['neutral']
                status = "NEUTRAL"
            
            # Dibujar bounding box alrededor de la cara
            face_points = np.array(landmarks)
            x_min, y_min = face_points.min(axis=0)
            x_max, y_max = face_points.max(axis=0)
            
            # Expandir un poco el bounding box
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img_copy.shape[1], x_max + margin)
            y_max = min(img_copy.shape[0], y_max + margin)
            
            # Dibujar rectángulo
            cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, 3)
            
            # Etiqueta con número de cara y estado
            label = f"Face {face_analysis['face_id']}: {status}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Fondo para el texto
            cv2.rectangle(img_copy, 
                         (x_min, y_min - label_size[1] - 10), 
                         (x_min + label_size[0] + 10, y_min), 
                         color, -1)
            
            # Texto
            cv2.putText(img_copy, label, (x_min + 5, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Información detallada cerca del centro de la cara
            center = face_analysis['center']
            info_y = center[1] + 40
            
            # Sonrisa
            smile_text = f"Smile: {face_analysis['smile']['confidence']:.1%}"
            cv2.putText(img_copy, smile_text, (center[0] - 50, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Ojos
            eyes_text = f"Eyes: {face_analysis['eyes']['confidence']:.1%}"
            cv2.putText(img_copy, eyes_text, (center[0] - 50, info_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Resumen en la parte superior
        summary_lines = [
            f"Faces: {analysis_result['num_faces']}",
            f"Smiling: {analysis_result['statistics']['smiling_faces']}",
            f"Eyes open: {analysis_result['statistics']['open_eyes_faces']}",
            f"Ideal: {analysis_result['statistics']['ideal_faces']}"
        ]
        
        for i, line in enumerate(summary_lines):
            y_pos = 30 + i * 25
            # Fondo para el texto
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(img_copy, (10, y_pos - text_size[1] - 5), 
                         (20 + text_size[0], y_pos + 5), (0, 0, 0), -1)
            # Texto
            cv2.putText(img_copy, line, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img_copy
    
    def get_face_regions(self, image, analysis_result):
        """
        Extrae regiones individuales de cada cara detectada.
        
        Returns:
            list: Lista de diccionarios con información de cada cara
        """
        if analysis_result['num_faces'] == 0:
            return []
        
        all_landmarks = self.detect_multiple_faces(image)
        face_regions = []
        
        for i, (face_analysis, landmarks) in enumerate(zip(analysis_result['faces'], all_landmarks)):
            # Calcular bounding box
            face_points = np.array(landmarks)
            x_min, y_min = face_points.min(axis=0)
            x_max, y_max = face_points.max(axis=0)
            
            # Expandir un poco
            margin = 30
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(image.shape[1], x_max + margin)
            y_max = min(image.shape[0], y_max + margin)
            
            # Extraer región
            face_crop = image[y_min:y_max, x_min:x_max].copy()
            
            face_regions.append({
                'face_id': face_analysis['face_id'],
                'image': face_crop,
                'bbox': (x_min, y_min, x_max, y_max),
                'analysis': face_analysis,
                'landmarks': landmarks
            })
        
        return face_regions
    
    def __del__(self):
        """Limpieza al destruir el objeto."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()