import cv2
import numpy as np
from smile_eye_detector import MultiFaceSmileEyeDetector

def test_multiple_faces(image_path):
    """Prueba detección de múltiples caras en una imagen."""
    print(f"\n{'='*50}")
    print(f"Probando: {image_path}")
    print(f"{'='*50}")
    
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar {image_path}")
        return
    
    try:
        # Crear detector
        detector = MultiFaceSmileEyeDetector()
        
        # Analizar todas las caras
        print("Analizando múltiples caras...")
        result = detector.analyze_multiple_faces(image)
        
        # Mostrar resultados generales
        print(f"\n📊 RESUMEN GENERAL:")
        print(f"   {result['summary']}")
        
        if result['num_faces'] > 0:
            stats = result['statistics']
            print(f"\n📈 ESTADÍSTICAS:")
            print(f"   • Total de caras: {stats['total_faces']}")
            print(f"   • Sonriendo: {stats['smiling_faces']} ({stats['smile_percentage']:.1f}%)")
            print(f"   • Ojos abiertos: {stats['open_eyes_faces']} ({stats['eyes_percentage']:.1f}%)")
            print(f"   • Expresión ideal: {stats['ideal_faces']} ({stats['ideal_percentage']:.1f}%)")
            
            # Detalles por cara
            print(f"\n👥 ANÁLISIS INDIVIDUAL:")
            for face in result['faces']:
                print(f"\n   🎭 CARA {face['face_id']}:")
                print(f"      Expresión: {face['expression']}")
                print(f"      Calidad: {face['quality_score']:.2f}")
                print(f"      Posición: {face['center']}")
                
                # Detalles de sonrisa
                smile = face['smile']
                status_smile = "✅ Sí" if smile['is_smiling'] else "❌ No"
                print(f"      Sonrisa: {status_smile} (confianza: {smile['confidence']:.1%})")
                
                if smile['metrics']:
                    print(f"         - Ratio boca: {smile['metrics'].get('mouth_ratio', 0):.2f}")
                    print(f"         - Curvatura: {smile['metrics'].get('mouth_curve_angle', 0):.1f}°")
                
                # Detalles de ojos
                eyes = face['eyes']
                status_eyes = "✅ Abiertos" if eyes['both_eyes_open'] else "❌ Cerrados"
                print(f"      Ojos: {status_eyes} (confianza: {eyes['confidence']:.1%})")
                
                if eyes['metrics']:
                    print(f"         - EAR izquierdo: {eyes['metrics'].get('left_ear', 0):.3f}")
                    print(f"         - EAR derecho: {eyes['metrics'].get('right_ear', 0):.3f}")
                
                # Estado general
                if face['is_ideal']:
                    print(f"      Estado: 🌟 IDEAL")
                elif smile['is_smiling']:
                    print(f"      Estado: 😊 SONRIENDO")
                elif eyes['both_eyes_open']:
                    print(f"      Estado: 👀 OJOS ABIERTOS")
                else:
                    print(f"      Estado: 😐 NEUTRAL")
        
        # Visualizar resultados
        print(f"\n🎨 Generando visualización...")
        vis_image = detector.visualize_multiple_faces(image, result)
        
        # Mostrar imagen
        window_name = f"Multi-Face Results - {image_path.split('/')[-1]}"
        cv2.imshow(window_name, vis_image)
        
        # Guardar resultado
        output_path = f"multi_face_result_{image_path.split('/')[-1]}"
        cv2.imwrite(output_path, vis_image)
        print(f"✅ Resultado guardado en: {output_path}")
        
        # Extraer regiones individuales de caras
        if result['num_faces'] > 0:
            print(f"\n💾 Extrayendo regiones individuales...")
            face_regions = detector.get_face_regions(image, result)
            
            for region in face_regions:
                face_id = region['face_id']
                face_img = region['image']
                analysis = region['analysis']
                
                status = "ideal" if analysis['is_ideal'] else "normal"
                print(f"   👤 Cara {face_id}: {face_img.shape[:2]} px, estado: {status}")
        
        # Esperar tecla para continuar
        print(f"\n⌨️  Presiona cualquier tecla para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_batch_images(image_paths):
    """Prueba múltiples imágenes y genera un reporte conjunto."""
    print(f"\n🚀 INICIANDO ANÁLISIS EN LOTE")
    print(f"{'='*60}")
    
    detector = MultiFaceSmileEyeDetector()
    batch_results = []
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        try:
            result = detector.analyze_multiple_faces(image)
            batch_results.append({
                'image_path': image_path,
                'result': result
            })
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
    
    # Generar reporte conjunto
    print(f"\n📋 REPORTE CONJUNTO:")
    print(f"{'='*60}")
    
    total_images = len(batch_results)
    total_faces = sum(r['result']['num_faces'] for r in batch_results)
    total_smiling = sum(r['result']['statistics']['smiling_faces'] for r in batch_results)
    total_eyes_open = sum(r['result']['statistics']['open_eyes_faces'] for r in batch_results)
    total_ideal = sum(r['result']['statistics']['ideal_faces'] for r in batch_results)
    
    print(f"📸 Imágenes procesadas: {total_images}")
    print(f"👥 Total de caras detectadas: {total_faces}")
    print(f"😊 Caras sonriendo: {total_smiling} ({total_smiling/total_faces*100:.1f}%)" if total_faces > 0 else "😊 Caras sonriendo: 0")
    print(f"👀 Caras con ojos abiertos: {total_eyes_open} ({total_eyes_open/total_faces*100:.1f}%)" if total_faces > 0 else "👀 Caras con ojos abiertos: 0")
    print(f"🌟 Caras con expresión ideal: {total_ideal} ({total_ideal/total_faces*100:.1f}%)" if total_faces > 0 else "🌟 Caras con expresión ideal: 0")
    
    print(f"\n📝 DETALLE POR IMAGEN:")
    for i, batch_result in enumerate(batch_results, 1):
        image_name = batch_result['image_path'].split('/')[-1]
        result = batch_result['result']
        print(f"   {i:2d}. {image_name:20} -> {result['num_faces']} cara(s), {result['statistics']['ideal_faces']} ideal(es)")

def create_comparison_montage(image_paths, output_path="multi_face_montage.jpg"):
    """Crea un montaje comparativo de múltiples imágenes con análisis."""
    detector = MultiFaceSmileEyeDetector()
    processed_images = []
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        try:
            result = detector.analyze_multiple_faces(image)
            vis_image = detector.visualize_multiple_faces(image, result)
            
            # Redimensionar para el montaje
            height = 400
            aspect_ratio = vis_image.shape[1] / vis_image.shape[0]
            width = int(height * aspect_ratio)
            vis_image = cv2.resize(vis_image, (width, height))
            
            # Agregar título con información
            title = f"{image_path.split('/')[-1]} - {result['num_faces']} cara(s)"
            title_height = 40
            title_img = np.zeros((title_height, width, 3), dtype=np.uint8)
            cv2.putText(title_img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Combinar título con imagen
            combined = np.vstack([title_img, vis_image])
            processed_images.append(combined)
            
        except Exception as e:
            print(f"Error procesando {image_path} para montaje: {e}")
    
    if not processed_images:
        print("No se pudieron procesar imágenes para el montaje")
        return
    
    # Crear montaje horizontal o en grid
    if len(processed_images) <= 4:
        # Montaje horizontal
        montage = np.hstack(processed_images)
    else:
        # Grid 2x2 o similar
        rows = []
        for i in range(0, len(processed_images), 2):
            if i + 1 < len(processed_images):
                row = np.hstack([processed_images[i], processed_images[i+1]])
            else:
                row = processed_images[i]
            rows.append(row)
        
        # Ajustar anchos de filas
        max_width = max(row.shape[1] for row in rows)
        for i, row in enumerate(rows):
            if row.shape[1] < max_width:
                padding = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
                rows[i] = np.hstack([row, padding])
        
        montage = np.vstack(rows)
    
    # Guardar montaje
    cv2.imwrite(output_path, montage)
    print(f"🖼️  Montaje guardado en: {output_path}")
    
    return montage

if __name__ == "__main__":
    # Ejemplo de uso con una sola imagen
    single_image = "assets/img20.jpeg"  # Cambia por tu imagen con múltiples caras
    
    print("🔍 MODO 1: Análisis de imagen individual")
    # Descomenta la línea siguiente si tienes una imagen con múltiples caras
    test_multiple_faces(single_image)
    
    # Ejemplo con múltiples imágenes
    test_images = [
        "assets/img1.png",
        # "assets/img2.png", 
        # "assets/img7.jpeg",
        # Agrega más rutas de imágenes aquí
    ]
    
    # print("\n🔍 MODO 2: Análisis individual por imagen")
    # for img_path in test_images:
    #     test_multiple_faces(img_path)
    # 
    # print("\n🔍 MODO 3: Análisis en lote")
    # test_batch_images(test_images)
    # 
    # print("\n🔍 MODO 4: Crear montaje comparativo")
    # create_comparison_montage(test_images)
    # 
    # print(f"\n✅ Análisis completado!")
    # print(f"📁 Revisa los archivos generados:")
    # print(f"   • multi_face_result_*.jpg - Imágenes con análisis")
    # print(f"   • face_*_*.jpg - Caras individuales extraídas")
    # print(f"   • multi_face_montage.jpg - Montaje comparativo")

# Funciones adicionales de utilidad

def analyze_single_face_in_image(image_path, face_index=0):
    """Analiza una cara específica en una imagen con múltiples caras."""
    detector = MultiFaceSmileEyeDetector()
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"No se pudo cargar {image_path}")
        return None
    
    result = detector.analyze_multiple_faces(image)
    
    if face_index >= result['num_faces']:
        print(f"Face index {face_index} no válido. Solo hay {result['num_faces']} caras.")
        return None
    
    return result['faces'][face_index]

def filter_faces_by_expression(image_path, expression_type="ideal"):
    """Filtra caras según el tipo de expresión."""
    detector = MultiFaceSmileEyeDetector()
    image = cv2.imread(image_path)
    
    if image is None:
        return []
    
    result = detector.analyze_multiple_faces(image)
    
    filtered_faces = []
    for face in result['faces']:
        if expression_type == "ideal" and face['is_ideal']:
            filtered_faces.append(face)
        elif expression_type == "smiling" and face['smile']['is_smiling']:
            filtered_faces.append(face)
        elif expression_type == "eyes_open" and face['eyes']['both_eyes_open']:
            filtered_faces.append(face)
        elif expression_type == "neutral" and not face['is_ideal']:
            filtered_faces.append(face)
    
    return filtered_faces

def get_best_face(image_path):
    """Encuentra la cara con la mejor expresión en la imagen."""
    detector = MultiFaceSmileEyeDetector()
    image = cv2.imread(image_path)
    
    if image is None:
        return None
    
    result = detector.analyze_multiple_faces(image)
    
    if result['num_faces'] == 0:
        return None
    
    # Encontrar la cara con mayor quality_score
    best_face = max(result['faces'], key=lambda x: x['quality_score'])
    return best_face

# Ejemplo de uso de funciones adicionales
def demo_additional_functions():
    """Demuestra las funciones adicionales."""
    image_path = "assets/img11.png" 
    
    print("\n🎯 FUNCIONES ADICIONALES:")
    
    # Analizar cara específica
    face_0 = analyze_single_face_in_image(image_path, 0)
    if face_0:
        print(f"Cara 0: {face_0['expression']}")
    
    # Filtrar caras ideales
    ideal_faces = filter_faces_by_expression(image_path, "ideal")
    print(f"Caras ideales encontradas: {len(ideal_faces)}")
    
    # Encontrar la mejor cara
    best = get_best_face(image_path)
    if best:
        print(f"Mejor cara: ID {best['face_id']} con calidad {best['quality_score']:.2f}")

# Ejecutar demo si se descomenta
# demo_additional_functions()