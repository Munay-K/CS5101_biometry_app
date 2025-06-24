import cv2
import numpy as np
import os
from scipy.spatial import Delaunay
from smile_eye_detector import MultiFaceSmileEyeDetector
from utils.landmarks import detect_landmarks

class BestFaceSwapper:
    def __init__(self):
        self.smile_detector = MultiFaceSmileEyeDetector()
        
    def detect_and_analyze_faces(self, image):
        """
        Detecta y analiza las expresiones de las caras en una imagen.
        Asume que hay exactamente 2 caras en cada imagen.
        """
        # Detectar landmarks de ambas caras
        all_landmarks = self.smile_detector.detect_multiple_faces(image)
        
        if len(all_landmarks) != 2:
            raise ValueError(f"Se esperaban 2 caras, pero se detectaron {len(all_landmarks)}")
        
        # Analizar cada cara
        analyses = []
        for i, landmarks in enumerate(all_landmarks):
            analysis = self.smile_detector.analyze_expression(landmarks)
            analysis['face_id'] = i
            analysis['landmarks'] = landmarks
            analyses.append(analysis)
            
        return analyses
    
    def warp_triangle(self, img_src, img_dst, t_src, t_dst):
        """
        Warps a triangle from source image to destination image.
        """
        # Find bounding boxes
        r1 = cv2.boundingRect(np.float32([t_src]))
        r2 = cv2.boundingRect(np.float32([t_dst]))
        
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        
        # Offset points by left top corner of respective rectangles
        src_rect = np.array([[pt[0]-x1, pt[1]-y1] for pt in t_src], np.float32)
        dst_rect = np.array([[pt[0]-x2, pt[1]-y2] for pt in t_dst], np.float32)
        
        # Crop input image
        crop = img_src[y1:y1+h1, x1:x1+w1]
        
        # Apply affine transformation
        M = cv2.getAffineTransform(src_rect, dst_rect)
        warped = cv2.warpAffine(crop, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        # Create mask
        mask = np.zeros((h2, w2), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_rect), 255)
        
        # Apply warped triangle to destination
        dst_roi = img_dst[y2:y2+h2, x2:x2+w2]
        dst_roi[mask==255] = warped[mask==255]
    
    def swap_face(self, source_img, target_img, source_landmarks, target_landmarks):
        """
        Swaps a face from source image to target image using Delaunay triangulation.
        """
        # Convert landmarks to numpy arrays
        source_lm = np.array(source_landmarks, np.int32)
        target_lm = np.array(target_landmarks, np.int32)
        
        # Create Delaunay triangulation on source landmarks
        tri = Delaunay(source_lm)
        triangles = tri.simplices
        
        # Create output image
        output = target_img.copy()
        
        # Warp each triangle
        for tri_idx in triangles:
            # Get triangle points for source and target
            t_src = source_lm[tri_idx]
            t_dst = target_lm[tri_idx]
            
            # Warp triangle
            self.warp_triangle(source_img, output, t_src, t_dst)
        
        # Seamless cloning for better blending
        # Get convex hull of target landmarks
        hull = cv2.convexHull(target_lm)
        
        # Create mask
        mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Find center of the face
        rect = cv2.boundingRect(hull)
        center = (rect[0] + rect[2]//2, rect[1] + rect[3]//2)
        
        # Apply seamless clone
        output = cv2.seamlessClone(output, target_img, mask, center, cv2.NORMAL_CLONE)
        
        return output
    
    def match_faces_between_images(self, faces1, faces2):
        """
        Encuentra la correspondencia entre caras de dos imágenes basándose en su posición.
        Asume que las personas están aproximadamente en las mismas posiciones relativas.
        """
        # Extraer centros de las caras
        centers1 = [np.array(face['landmarks'][1]) for face in faces1]  # Usando punta de nariz como centro
        centers2 = [np.array(face['landmarks'][1]) for face in faces2]  # landmarks[1] es nose_tip
        
        # Ordenar caras por posición X (de izquierda a derecha)
        sorted_indices1 = sorted(range(len(centers1)), key=lambda i: centers1[i][0])
        sorted_indices2 = sorted(range(len(centers2)), key=lambda i: centers2[i][0])
        
        # Crear mapeo
        face_mapping = {}
        for i in range(min(len(sorted_indices1), len(sorted_indices2))):
            face_mapping[sorted_indices1[i]] = sorted_indices2[i]
        
        return face_mapping
    
    def create_best_combination(self, img1_path, img2_path, output_path=None):
        """
        Crea una imagen combinando las mejores caras de dos imágenes.
        IMPORTANTE: Mantiene cada persona en su posición original, solo usa su mejor versión.
        
        Args:
            img1_path: Ruta de la primera imagen (con 2 personas)
            img2_path: Ruta de la segunda imagen (con las mismas 2 personas)
            output_path: Ruta donde guardar el resultado (opcional)
            
        Returns:
            combined_image: La imagen resultante con las mejores caras
            swap_info: Información sobre qué caras se intercambiaron
        """
        # Cargar imágenes
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("No se pudieron cargar las imágenes")
        
        print("🔍 Analizando caras en la primera imagen...")
        faces1 = self.detect_and_analyze_faces(img1)
        
        print("🔍 Analizando caras en la segunda imagen...")
        faces2 = self.detect_and_analyze_faces(img2)
        
        # Mostrar análisis de cada cara
        print("\n📊 ANÁLISIS DE CARAS:")
        print("Imagen 1:")
        for i, face in enumerate(faces1):
            position = "izquierda" if i == 0 else "derecha"
            print(f"  Persona {position}: {face['expression']} (Calidad: {face['quality_score']:.2f})")
        
        print("Imagen 2:")
        for i, face in enumerate(faces2):
            position = "izquierda" if i == 0 else "derecha"
            print(f"  Persona {position}: {face['expression']} (Calidad: {face['quality_score']:.2f})")
        
        # Encontrar correspondencia entre caras basándose en posición
        face_mapping = self.match_faces_between_images(faces1, faces2)
        
        # Determinar qué cara es mejor para cada persona
        swap_info = {}
        needs_swap = []
        
        for idx1, idx2 in face_mapping.items():
            position = "izquierda" if idx1 == 0 else "derecha"
            
            if faces1[idx1]['quality_score'] > faces2[idx2]['quality_score']:
                swap_info[f'persona_{position}'] = {
                    'best_from': 'img1',
                    'score': faces1[idx1]['quality_score'],
                    'idx_img1': idx1,
                    'idx_img2': idx2
                }
            else:
                swap_info[f'persona_{position}'] = {
                    'best_from': 'img2',
                    'score': faces2[idx2]['quality_score'],
                    'idx_img1': idx1,
                    'idx_img2': idx2
                }
                needs_swap.append((idx1, idx2))
        
        print(f"\n🎯 MEJOR SELECCIÓN:")
        for person_id, info in sorted(swap_info.items()):
            print(f"  {person_id}: Mejor cara de {info['best_from']} (score: {info['score']:.2f})")
        
        # Determinar qué hacer
        all_from_img1 = all(info['best_from'] == 'img1' for info in swap_info.values())
        all_from_img2 = all(info['best_from'] == 'img2' for info in swap_info.values())
        
        if all_from_img1:
            print("\n✅ Usando imagen 1 como resultado (todas las mejores caras están ahí)")
            result_img = img1.copy()
        elif all_from_img2:
            print("\n✅ Usando imagen 2 como resultado (todas las mejores caras están ahí)")
            result_img = img2.copy()
        else:
            # Necesitamos combinar caras
            print("\n🔄 Combinando las mejores caras de ambas imágenes...")
            
            # IMPORTANTE: Usar imagen 1 como base y reemplazar solo las caras que son mejores en imagen 2
            result_img = img1.copy()
            
            for person_id, info in swap_info.items():
                if info['best_from'] == 'img2':
                    idx1 = info['idx_img1']
                    idx2 = info['idx_img2']
                    print(f"  - Reemplazando {person_id} con su mejor versión de imagen 2...")
                    
                    # Aquí está la clave: transferimos la cara de la misma persona desde img2 a img1
                    # NO intercambiamos entre personas diferentes
                    result_img = self.swap_face(
                        img2,  # source (de donde tomamos la cara)
                        result_img,  # target (donde la ponemos)
                        faces2[idx2]['landmarks'],  # landmarks de la persona en img2
                        faces1[idx1]['landmarks']   # landmarks de LA MISMA persona en img1
                    )
        """
        Crea una imagen combinando las mejores caras de dos imágenes.
        
        Args:
            img1_path: Ruta de la primera imagen (con 2 personas)
            img2_path: Ruta de la segunda imagen (con las mismas 2 personas)
            output_path: Ruta donde guardar el resultado (opcional)
            
        Returns:
            combined_image: La imagen resultante con las mejores caras
            swap_info: Información sobre qué caras se intercambiaron
        """
        # Cargar imágenes
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("No se pudieron cargar las imágenes")
        
        print("🔍 Analizando caras en la primera imagen...")
        faces1 = self.detect_and_analyze_faces(img1)
        
        print("🔍 Analizando caras en la segunda imagen...")
        faces2 = self.detect_and_analyze_faces(img2)
        
        # Mostrar análisis de cada cara
        print("\n📊 ANÁLISIS DE CARAS:")
        print("Imagen 1:")
        for face in faces1:
            print(f"  Cara {face['face_id']+1}: {face['expression']} (Calidad: {face['quality_score']:.2f})")
        
        print("Imagen 2:")
        for face in faces2:
            print(f"  Cara {face['face_id']+1}: {face['expression']} (Calidad: {face['quality_score']:.2f})")
        
        # Encontrar correspondencia entre caras
        face_mapping = self.match_faces_between_images(faces1, faces2)
        print("\n🔗 Correspondencia de caras detectada:")
        for idx1, idx2 in face_mapping.items():
            print(f"  Cara {idx1+1} en img1 ↔ Cara {idx2+1} en img2")
        
        # Determinar qué cara es mejor para cada persona
        swap_info = {}
        best_combination = {}  # Para rastrear qué índices usar
        
        for idx1, idx2 in face_mapping.items():
            person_id = f'person{idx1+1}'
            
            if faces1[idx1]['quality_score'] > faces2[idx2]['quality_score']:
                swap_info[person_id] = {
                    'best_from': 'img1',
                    'score': faces1[idx1]['quality_score'],
                    'idx_img1': idx1,
                    'idx_img2': idx2
                }
                best_combination[person_id] = ('img1', idx1)
            else:
                swap_info[person_id] = {
                    'best_from': 'img2',
                    'score': faces2[idx2]['quality_score'],
                    'idx_img1': idx1,
                    'idx_img2': idx2
                }
                best_combination[person_id] = ('img2', idx2)
        
        print(f"\n🎯 MEJOR SELECCIÓN:")
        for person_id, info in swap_info.items():
            print(f"  {person_id}: Mejor cara de {info['best_from']} (score: {info['score']:.2f})")
        
        # Determinar si necesitamos face swap
        all_from_img1 = all(info['best_from'] == 'img1' for info in swap_info.values())
        all_from_img2 = all(info['best_from'] == 'img2' for info in swap_info.values())
        
        if all_from_img1:
            print("\n✅ Usando imagen 1 como resultado (todas las mejores caras están ahí)")
            result_img = img1.copy()
        elif all_from_img2:
            print("\n✅ Usando imagen 2 como resultado (todas las mejores caras están ahí)")
            result_img = img2.copy()
        else:
            # Necesitamos hacer face swap
            print("\n🔄 Realizando face swap para combinar las mejores caras...")
            
            # Decidir qué imagen usar como base (la que tenga más caras buenas)
            img1_count = sum(1 for info in swap_info.values() if info['best_from'] == 'img1')
            img2_count = sum(1 for info in swap_info.values() if info['best_from'] == 'img2')
            
            if img1_count >= img2_count:
                # Usar img1 como base
                result_img = img1.copy()
                print(f"  - Usando imagen 1 como base ({img1_count} caras buenas)")
                
                # Transferir las caras que son mejores en img2
                for person_id, info in swap_info.items():
                    if info['best_from'] == 'img2':
                        idx1 = info['idx_img1']
                        idx2 = info['idx_img2']
                        print(f"  - Transfiriendo {person_id} desde imagen 2...")
                        result_img = self.swap_face(
                            img2,  # source
                            result_img,  # target
                            faces2[idx2]['landmarks'],  # source landmarks
                            faces1[idx1]['landmarks']   # target landmarks
                        )
            else:
                # Usar img2 como base
                result_img = img2.copy()
                print(f"  - Usando imagen 2 como base ({img2_count} caras buenas)")
                
                # Transferir las caras que son mejores en img1
                for person_id, info in swap_info.items():
                    if info['best_from'] == 'img1':
                        idx1 = info['idx_img1']
                        idx2 = info['idx_img2']
                        print(f"  - Transfiriendo {person_id} desde imagen 1...")
                        result_img = self.swap_face(
                            img1,  # source
                            result_img,  # target
                            faces1[idx1]['landmarks'],  # source landmarks
                            faces2[idx2]['landmarks']   # target landmarks
                        )

        
        # Agregar texto informativo a la imagen resultado
        self.add_info_to_image(result_img, swap_info)
        
        # Guardar resultado si se especificó
        if output_path:
            cv2.imwrite(output_path, result_img)
            print(f"\n💾 Imagen guardada en: {output_path}")
        
        return result_img, swap_info
    
    def add_info_to_image(self, image, swap_info):
        """
        Agrega información sobre la selección de caras a la imagen.
        """
        h, w = image.shape[:2]
        
        # Crear una barra de información en la parte superior
        info_height = 80
        info_bar = np.zeros((info_height, w, 3), dtype=np.uint8)
        info_bar[:] = (40, 40, 40)  # Gris oscuro
        
        # Combinar con la imagen original
        result = np.vstack([info_bar, image])
        
        # Agregar texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Título
        cv2.putText(result, "MEJOR COMBINACION DE CARAS", (20, 30), 
                   font, 0.8, (255, 255, 255), 2)
        
        # Información de cada persona
        y_pos = 55
        x_pos = 20
        for person_id, info in sorted(swap_info.items()):
            text = f"{person_id}: {info['best_from']} (Score: {info['score']:.2f})"
            cv2.putText(result, text, (x_pos, y_pos), 
                       font, 0.6, (0, 255, 0), 1)
            x_pos += w // 2
        
        # Actualizar la imagen original
        image[:] = result[info_height:, :]
    
    def create_comparison_grid(self, img1_path, img2_path, result_img, output_path="comparison_grid.jpg"):
        """
        Crea una grilla comparativa mostrando las dos imágenes originales y el resultado.
        """
        # Cargar imágenes
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # Redimensionar todas las imágenes al mismo tamaño
        target_height = 600
        
        def resize_image(img):
            h, w = img.shape[:2]
            aspect = w / h
            new_width = int(target_height * aspect)
            return cv2.resize(img, (new_width, target_height))
        
        img1_resized = resize_image(img1)
        img2_resized = resize_image(img2)
        result_resized = resize_image(result_img)
        
        # Crear grilla
        # Primera fila: imágenes originales
        # Segunda fila: resultado
        
        # Agregar etiquetas
        label_height = 40
        
        def add_label(img, text):
            h, w = img.shape[:2]
            labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
            labeled[label_height:, :] = img
            labeled[:label_height, :] = (100, 100, 100)
            cv2.putText(labeled, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 255, 255), 2)
            return labeled
        
        img1_labeled = add_label(img1_resized, "Imagen 1 (Original)")
        img2_labeled = add_label(img2_resized, "Imagen 2 (Original)")
        result_labeled = add_label(result_resized, "Resultado (Mejores Caras)")
        
        # Combinar en grilla
        top_row = np.hstack([img1_labeled, np.ones((img1_labeled.shape[0], 10, 3), dtype=np.uint8) * 255, img2_labeled])
        
        # Centrar la imagen resultado
        total_width = top_row.shape[1]
        result_width = result_labeled.shape[1]
        padding = (total_width - result_width) // 2
        
        bottom_row = np.hstack([
            np.zeros((result_labeled.shape[0], padding, 3), dtype=np.uint8),
            result_labeled,
            np.zeros((result_labeled.shape[0], total_width - result_width - padding, 3), dtype=np.uint8)
        ])
        
        # Combinar filas
        grid = np.vstack([
            top_row,
            np.ones((20, total_width, 3), dtype=np.uint8) * 255,
            bottom_row
        ])
        
        # Guardar
        cv2.imwrite(output_path, grid)
        print(f"📊 Grilla comparativa guardada en: {output_path}")
        
        return grid


def main():
    print("🚀 GENERADOR DE MEJOR COMBINACIÓN DE CARAS")
    print("=" * 50)
    
    # CONFIGURACIÓN - Modifica estas rutas según tus archivos
    image1_path = "assets/img22.jpeg"  # Primera foto con 2 personas
    image2_path = "assets/img20.jpeg"  # Segunda foto con las mismas 2 personas
    output_path = "resultado_mejores_caras.jpg"  # Archivo de salida
    generate_grid = True  # Si quieres generar la grilla comparativa
    
    # Verificar que los archivos existan
    if not os.path.exists(image1_path):
        print(f"❌ Error: No se encuentra el archivo {image1_path}")
        print("   Por favor, actualiza la ruta en el código.")
        return
    
    if not os.path.exists(image2_path):
        print(f"❌ Error: No se encuentra el archivo {image2_path}")
        print("   Por favor, actualiza la ruta en el código.")
        return
    
    try:
        # Crear el face swapper
        swapper = BestFaceSwapper()
        
        # Procesar imágenes
        result_img, swap_info = swapper.create_best_combination(
            image1_path, 
            image2_path, 
            output_path
        )
        
        # Crear grilla comparativa si se solicitó
        if generate_grid:
            grid_path = output_path.replace('.jpg', '_grid.jpg').replace('.png', '_grid.png')
            swapper.create_comparison_grid(image1_path, image2_path, result_img, grid_path)
        
        print("\n✅ ¡Proceso completado con éxito!")
        
        # Mostrar resultado
        cv2.imshow("Resultado - Mejores Caras", result_img)
        print("\n⌨️  Presiona cualquier tecla para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()