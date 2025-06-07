import numpy as np
from scipy.spatial.distance import cosine, euclidean, mahalanobis
from sklearn.preprocessing import StandardScaler
from utils.landmarks import extract_geometric_features, LANDMARK_INDICES
import cv2

def calculate_advanced_biometric_vector(landmarks):
    """
    Calcula un vector biométrico avanzado usando múltiples características faciales.
    """
    # Extraer características geométricas básicas
    geometric_features = extract_geometric_features(landmarks)
    
    # Calcular características adicionales más discriminativas
    additional_features = []
    
    # 1. Triángulos faciales únicos
    triangles = [
        # Triángulo ojo-nariz-boca
        (LANDMARK_INDICES['left_eye_outer'], LANDMARK_INDICES['nose_tip'], LANDMARK_INDICES['mouth_left']),
        (LANDMARK_INDICES['right_eye_outer'], LANDMARK_INDICES['nose_tip'], LANDMARK_INDICES['mouth_right']),
        # Triángulo frente-mejillas
        (LANDMARK_INDICES['forehead'], LANDMARK_INDICES['cheek_left'], LANDMARK_INDICES['cheek_right']),
        # Triángulo mandíbula
        (LANDMARK_INDICES['jaw_left'], LANDMARK_INDICES['chin'], LANDMARK_INDICES['jaw_right']),
    ]
    
    for tri in triangles:
        # Calcular área del triángulo
        p1, p2, p3 = landmarks[tri[0]], landmarks[tri[1]], landmarks[tri[2]]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        additional_features.append(area)
        
        # Calcular perímetro
        perimeter = (np.linalg.norm(np.array(p1) - np.array(p2)) +
                    np.linalg.norm(np.array(p2) - np.array(p3)) +
                    np.linalg.norm(np.array(p3) - np.array(p1)))
        additional_features.append(perimeter)
    
    # 2. Curvaturas y ángulos específicos
    # Curvatura de la mandíbula
    jaw_points = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]
    jaw_curvature = calculate_curvature(landmarks, jaw_points)
    additional_features.append(jaw_curvature)
    
    # Ángulo de inclinación de ojos
    left_eye_angle = calculate_eye_tilt(
        landmarks[LANDMARK_INDICES['left_eye_inner']], 
        landmarks[LANDMARK_INDICES['left_eye_outer']]
    )
    right_eye_angle = calculate_eye_tilt(
        landmarks[LANDMARK_INDICES['right_eye_inner']], 
        landmarks[LANDMARK_INDICES['right_eye_outer']]
    )
    additional_features.extend([left_eye_angle, right_eye_angle])
    
    # 3. Proporciones muy específicas
    # Proporción dorada facial
    face_height = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['forehead']]) - 
        np.array(landmarks[LANDMARK_INDICES['chin']])
    )
    face_width = np.linalg.norm(
        np.array(landmarks[LANDMARK_INDICES['jaw_left']]) - 
        np.array(landmarks[LANDMARK_INDICES['jaw_right']])
    )
    golden_ratio = face_height / face_width if face_width > 0 else 0
    additional_features.append(golden_ratio)
    
    # 4. Distancias cruzadas entre TODOS los puntos clave
    key_indices = list(LANDMARK_INDICES.values())
    cross_distances = []
    for i in range(len(key_indices)):
        for j in range(i + 1, len(key_indices)):
            dist = np.linalg.norm(
                np.array(landmarks[key_indices[i]]) - 
                np.array(landmarks[key_indices[j]])
            )
            cross_distances.append(dist)
    
    # Normalizar distancias cruzadas con método robusto
    if cross_distances:
        cross_distances = np.array(cross_distances)
        # Normalización por face_width para hacer invariante a escala
        cross_distances = cross_distances / face_width
        # Usar media y desviación estándar para normalización Z-score
        mean_dist = np.mean(cross_distances)
        std_dist = np.std(cross_distances)
        if std_dist > 0:
            cross_distances = (cross_distances - mean_dist) / std_dist
            # Aplicar función tanh para comprimir outliers
            cross_distances = np.tanh(cross_distances / 2)
    
    # 5. Momentos de Hu para la forma facial
    face_contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 340, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 227, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
    hu_moments = calculate_hu_moments(landmarks, face_contour_indices)
    
    # Combinar todas las características con pesos adaptativos
    # Dar más peso a características geométricas que son más estables
    geometric_weight = 0.6
    additional_weight = 0.2
    cross_weight = 0.15
    hu_weight = 0.05
    
    # Normalizar cada grupo por separado
    geometric_norm = geometric_features / (np.linalg.norm(geometric_features) + 1e-8)
    additional_norm = np.array(additional_features) / (np.linalg.norm(additional_features) + 1e-8)
    cross_norm = cross_distances[:30] / (np.linalg.norm(cross_distances[:30]) + 1e-8)
    hu_norm = hu_moments / (np.linalg.norm(hu_moments) + 1e-8)
    
    # Combinar con pesos
    all_features = np.concatenate([
        geometric_norm * geometric_weight,
        additional_norm * additional_weight,
        cross_norm * cross_weight,
        hu_norm * hu_weight
    ])
    
    # NO aplicar transformación no lineal agresiva que amplifica diferencias
    # Esto causaba que pequeñas variaciones naturales se amplificaran demasiado
    
    # Normalización final suave
    norm = np.linalg.norm(all_features)
    if norm > 0:
        all_features = all_features / norm
    
    return all_features

def calculate_curvature(landmarks, indices):
    """Calcula la curvatura de una serie de puntos."""
    if len(indices) < 3:
        return 0
    
    points = [landmarks[i] for i in indices if i < len(landmarks)]
    if len(points) < 3:
        return 0
    
    # Calcular la curvatura usando diferencias finitas
    curvatures = []
    for i in range(1, len(points) - 1):
        p1 = np.array(points[i-1])
        p2 = np.array(points[i])
        p3 = np.array(points[i+1])
        
        # Vectores
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Ángulo entre vectores
        angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
        curvatures.append(angle)
    
    return np.mean(curvatures) if curvatures else 0

def calculate_eye_tilt(inner_point, outer_point):
    """Calcula el ángulo de inclinación del ojo."""
    dx = outer_point[0] - inner_point[0]
    dy = outer_point[1] - inner_point[1]
    return np.arctan2(dy, dx)

def calculate_hu_moments(landmarks, contour_indices):
    """Calcula los momentos de Hu de la forma facial."""
    # Crear una imagen binaria con el contorno
    points = [landmarks[i] for i in contour_indices if i < len(landmarks)]
    if len(points) < 3:
        return np.zeros(7)
    
    # Encontrar el bounding box
    points = np.array(points)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    
    # Crear imagen binaria
    width = int(x_max - x_min + 1)
    height = int(y_max - y_min + 1)
    if width <= 0 or height <= 0:
        return np.zeros(7)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Trasladar puntos al origen
    translated_points = points - [x_min, y_min]
    translated_points = translated_points.astype(np.int32)
    
    # Dibujar el polígono
    cv2.fillPoly(mask, [translated_points], 255)
    
    # Calcular momentos
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform para estabilidad
    hu_moments = np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return hu_moments

def compare_vectors_advanced(vector1, vector2, method='ensemble'):
    """
    Compara dos vectores biométricos usando un enfoque de conjunto más estricto.
    """
    if len(vector1) != len(vector2):
        raise ValueError("Los vectores deben tener la misma longitud")
    
    # Calcular múltiples métricas de distancia
    metrics = {}
    
    # 1. Distancia Euclidiana
    euclidean_dist = euclidean(vector1, vector2)
    # Para vectores idénticos, la distancia es 0, entonces la similitud debe ser 1
    metrics['euclidean'] = 1 / (1 + euclidean_dist * 2)  # Escalar más agresivamente
    
    # 2. Distancia del Coseno
    cosine_dist = cosine(vector1, vector2)
    metrics['cosine'] = max(0, 1 - cosine_dist)  # Asegurar no negativos
    
    # 3. Correlación de Pearson
    if np.std(vector1) > 1e-10 and np.std(vector2) > 1e-10:
        pearson_corr = np.corrcoef(vector1, vector2)[0, 1]
        metrics['pearson'] = (pearson_corr + 1) / 2
    else:
        # Si los vectores son idénticos o constantes
        if np.allclose(vector1, vector2):
            metrics['pearson'] = 1.0
        else:
            metrics['pearson'] = 0.5
    
    # 4. Distancia de Manhattan
    manhattan_dist = np.sum(np.abs(vector1 - vector2))
    metrics['manhattan'] = 1 / (1 + manhattan_dist)
    
    # 5. Distancia de Chebyshev (máxima diferencia)
    chebyshev_dist = np.max(np.abs(vector1 - vector2))
    metrics['chebyshev'] = 1 - min(chebyshev_dist, 1.0)
    
    # 6. Índice de Jaccard para características binarias
    # Binarizar características para Jaccard
    threshold = 0.5
    binary1 = vector1 > threshold
    binary2 = vector2 > threshold
    intersection = np.sum(binary1 & binary2)
    union = np.sum(binary1 | binary2)
    metrics['jaccard'] = intersection / union if union > 0 else 1.0
    
    if method == 'ensemble':
        # Enfoque de conjunto con votación ponderada
        weights = {
            'euclidean': 0.25,
            'cosine': 0.20,
            'pearson': 0.15,
            'manhattan': 0.20,
            'chebyshev': 0.10,
            'jaccard': 0.10
        }
        
        similarity = sum(metrics[m] * weights[m] for m in metrics)
        
        # Para imágenes idénticas o casi idénticas, no aplicar transformación
        if similarity > 0.98:
            # Mantener la similitud alta tal cual
            threshold = 0.95
        else:
            # Aplicar función sigmoide adaptativa solo para casos no idénticos
            if similarity > 0.95:
                # Para similitudes muy altas, aplicar transformación suave
                k = 5  # Factor de pendiente más suave
                midpoint = 0.96
            else:
                # Para similitudes medias, aplicar transformación moderada
                k = 8  # Factor de pendiente moderado
                midpoint = 0.90
            
            similarity_adjusted = 1 / (1 + np.exp(-k * (similarity - midpoint)))
            
            # Umbral adaptativo basado en la distribución de características
            feature_variance = np.var(np.abs(vector1 - vector2))
            if feature_variance < 0.01:  # Muy poca variación
                threshold = 0.55  # Más permisivo
            elif feature_variance < 0.03:  # Variación moderada
                threshold = 0.60
            else:  # Alta variación
                threshold = 0.65  # Más estricto
            
            similarity = similarity_adjusted
        
    else:
        # Método simple
        similarity = metrics.get(method, metrics['euclidean'])
        threshold = 0.90
    
    is_same_person = similarity >= threshold
    
    # Calcular confianza
    if is_same_person:
        confidence = (similarity - threshold) / (1 - threshold)
    else:
        confidence = (threshold - similarity) / threshold
    
    confidence = np.clip(confidence, 0, 1)
    
    return similarity, is_same_person, confidence

def verify_identity(landmarks1, landmarks2, strict_mode=True):
    """
    Verifica si dos conjuntos de landmarks pertenecen a la misma persona.
    Balanceado para minimizar tanto falsos positivos como falsos negativos.
    """
    # Calcular vectores biométricos avanzados
    vector1 = calculate_advanced_biometric_vector(landmarks1)
    vector2 = calculate_advanced_biometric_vector(landmarks2)
    
    # Comparar usando enfoque de conjunto
    sim_ensemble, same_ensemble, conf_ensemble = compare_vectors_advanced(
        vector1, vector2, method='ensemble'
    )
    
    # Calcular métricas adicionales para tomar una decisión más informada
    individual_sims = []
    for method in ['euclidean', 'cosine', 'manhattan']:
        sim, _, _ = compare_vectors_advanced(vector1, vector2, method=method)
        individual_sims.append(sim)
    
    # Análisis de consistencia
    sim_std = np.std(individual_sims)
    sim_mean = np.mean(individual_sims)
    
    # Si hay alta consistencia entre métodos, confiar más en el resultado
    consistency_factor = 1.0
    if sim_std < 0.05:  # Alta consistencia
        consistency_factor = 1.1
    elif sim_std > 0.15:  # Baja consistencia
        consistency_factor = 0.9
    
    # Ajustar similitud por consistencia
    sim_adjusted = sim_ensemble * consistency_factor
    sim_adjusted = min(sim_adjusted, 1.0)  # Cap at 1.0
    
    # En modo estricto, aplicar verificaciones adicionales
    if strict_mode:
        # Si la similitud está en zona ambigua, hacer análisis más profundo
        if 0.45 <= sim_ensemble <= 0.65 and sim_ensemble < 0.98:
            # Verificar características críticas
            critical_features = vector1[:10]  # Primeras 10 características son las más importantes
            critical_diff = np.mean(np.abs(critical_features - vector2[:10]))
            
            # Si las características críticas son muy diferentes, es persona diferente
            if critical_diff > 0.15:
                same_ensemble = False
                conf_ensemble = 0.8
            # Si las características críticas son muy similares, es la misma persona
            elif critical_diff < 0.05:
                same_ensemble = True
                conf_ensemble = 0.8
    
    # Decisión final considerando el contexto
    # Para imágenes idénticas
    if sim_ensemble > 0.99 or np.allclose(vector1, vector2, rtol=1e-5, atol=1e-8):
        same_ensemble = True
        conf_ensemble = 1.0
        final_similarity = 1.0  # Mostrar 1.0 para imágenes idénticas
    # Para similitudes muy altas (>0.7 en escala ajustada)
    elif sim_adjusted > 0.70:
        same_ensemble = True
        conf_ensemble = min((sim_adjusted - 0.70) / 0.30, 1.0)
        final_similarity = sim_ensemble
    # Para similitudes muy bajas (<0.3)
    elif sim_adjusted < 0.30:
        same_ensemble = False
        conf_ensemble = min((0.30 - sim_adjusted) / 0.30, 1.0)
        final_similarity = sim_ensemble
    else:
        # Zona ambigua - usar el resultado del ensemble
        conf_ensemble = conf_ensemble * 0.7  # Reducir confianza en zona ambigua
        final_similarity = sim_ensemble
    
    # Almacenar información de debug
    debug_info = {
        'raw_similarity': sim_mean,  # Similitud promedio sin transformar
        'is_identical': np.allclose(vector1, vector2, rtol=1e-5, atol=1e-8),
        'feature_variance': np.var(np.abs(vector1 - vector2))
    }
    
    results = {
        'is_same_person': same_ensemble,
        'confidence': conf_ensemble,
        'similarity_score': final_similarity,  # Usar similitud final corregida
        'methods': {
            'ensemble': {'similarity': final_similarity, 'match': same_ensemble},
            'euclidean': {'similarity': individual_sims[0] if individual_sims else 0, 'match': individual_sims[0] > 0.85 if individual_sims else False},
            'cosine': {'similarity': individual_sims[1] if len(individual_sims) > 1 else 0, 'match': individual_sims[1] > 0.90 if len(individual_sims) > 1 else False},
            'weighted': {'similarity': sim_mean, 'match': sim_mean > 0.87}
        },
        'vector1': vector1,
        'vector2': vector2,
        'debug': debug_info  # Incluir información de debug
    }
    
    return results

# Funciones de compatibilidad
def calculate_biometric_vector(landmarks):
    """Función de compatibilidad con la versión anterior."""
    return calculate_advanced_biometric_vector(landmarks)

def compare_vectors(vector1, vector2, threshold=0.90):
    """Función de compatibilidad con la versión anterior."""
    similarity, is_same_person, confidence = compare_vectors_advanced(
        vector1, vector2, method='ensemble'
    )
    return similarity, is_same_person