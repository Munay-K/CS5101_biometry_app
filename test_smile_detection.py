import cv2
import numpy as np
from utils.landmarks import detect_landmarks
from smile_eye_detector import SmileEyeDetector

def test_image(image_path):
    """Prueba básica en una imagen."""
    print(f"\nProbando: {image_path}")
    
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar {image_path}")
        return
    
    try:
        # Detectar landmarks
        print("Detectando landmarks...")
        landmarks = detect_landmarks(image)
        print(f"Landmarks detectados: {len(landmarks)}")
        
        # Verificar tipo de datos
        print(f"Tipo de primer landmark: {type(landmarks[0])}")
        
        # Crear detector
        detector = SmileEyeDetector()
        
        # Analizar expresión
        print("Analizando expresión...")
        result = detector.analyze_expression(landmarks)
        
        # Mostrar resultados
        print(f"\nResultados:")
        print(f"- Expresión: {result['expression']}")
        print(f"- Calidad: {result['quality_score']:.2f}")
        print(f"- Sonriendo: {'Sí' if result['smile']['is_smiling'] else 'No'} ({result['smile']['confidence']:.2%})")
        print(f"- Ojos abiertos: {'Sí' if result['eyes']['both_eyes_open'] else 'No'} ({result['eyes']['confidence']:.2%})")
        
        # Visualizar
        vis_image = detector.visualize_detection(image, landmarks, result)
        
        # Mostrar imagen
        cv2.imshow(f"Resultado - {image_path}", vis_image)
        
        # Guardar resultado
        output_path = f"result_{image_path.split('/')[-1]}"
        cv2.imwrite(output_path, vis_image)
        print(f"\nResultado guardado en: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Probar con las imágenes disponibles
    test_images = ["assets/img9.jpeg"]
    
    for img_path in test_images:
        test_image(img_path)
    
    print("\nPresiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()