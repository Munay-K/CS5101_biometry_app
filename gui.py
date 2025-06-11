import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from utils.landmarks import detect_landmarks, extract_geometric_features, get_feature_labels
from utils.verification import calculate_biometric_vector, compare_vectors, verify_identity
from utils.visualization import visualize_landmarks, draw_face_mesh, FACE_CONNECTIONS

class BiometricAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Verificación Biométrica Facial - Versión Mejorada")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")

        # Variables
        self.master_image = None
        self.submitted_image = None
        self.similarity = None
        self.verification_results = None

        # Layout
        self.create_layout()
        # caches for landmarks and mesh triangles
        self.master_landmarks = None
        self.submitted_landmarks = None
        self.triangles_cache = None

    def create_layout(self):
        # Panel de imágenes
        self.image_frame = tk.Frame(self.root, bg="#ffffff", padx=10, pady=10)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.master_panel = tk.LabelFrame(self.image_frame, text="Subject 1", bg="#ffffff", font=("Arial", 12))
        self.master_panel.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.submitted_panel = tk.LabelFrame(self.image_frame, text="Subject 2", bg="#ffffff", font=("Arial", 12))
        self.submitted_panel.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.master_image_label = tk.Label(self.master_panel, bg="#d9d9d9")
        self.master_image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.submitted_image_label = tk.Label(self.submitted_panel, bg="#d9d9d9")
        self.submitted_image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Panel de resultados detallados
        self.results_panel = tk.LabelFrame(self.image_frame, text="Análisis Detallado", bg="#ffffff", font=("Arial", 12))
        self.results_panel.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Resultados principales
        self.result_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        self.result_frame.pack(side=tk.TOP, fill=tk.X)

        self.metrics_label = tk.Label(self.result_frame, text="Metrics: --", font=("Arial", 14, "bold"), bg="#f0f0f0")
        self.metrics_label.pack(side=tk.LEFT, padx=20)

        self.confidence_label = tk.Label(self.result_frame, text="Confianza: --", font=("Arial", 14), bg="#f0f0f0")
        self.confidence_label.pack(side=tk.LEFT, padx=20)

        self.timestamp_label = tk.Label(self.result_frame, text="Timestamp: --", font=("Arial", 12), bg="#f0f0f0")
        self.timestamp_label.pack(side=tk.LEFT, padx=20)

        # Etiqueta para mostrar comparación de características
        self.features_text = tk.Text(self.results_panel, font=("Courier", 9), bg="#ffffff",
                                    wrap=tk.WORD, height=20, width=50)
        self.features_text.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Crear scrollbar para el texto
        scrollbar = tk.Scrollbar(self.results_panel, command=self.features_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.features_text.config(yscrollcommand=scrollbar.set)

        # Botones de decisión
        self.button_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        # Dropdown menu for visualization options
        self.visualization_menu = tk.Menubutton(self.button_frame, text="Opciones de Visualización", bg="#2196F3", fg="white",
                                                font=("Arial", 12), relief=tk.RAISED)
        menu = tk.Menu(self.visualization_menu, tearoff=0)
        menu.add_command(label="Mostrar Imágenes Originales", command=self.show_original_images)
        menu.add_command(label="Mostrar Landmarks", command=self.show_landmarks)
        menu.add_command(label="Resaltar Discrepancias", command=self.highlight_discrepancies)
        menu.add_command(label="Mostrar Mesh", command=self.show_mesh)
        menu.add_command(label="Mostrar Solo Mesh", command=self.show_mesh_only)
        menu.add_command(label="Mostrar Análisis Comparativo", command=self.show_comparative_analysis)
        menu.add_command(label="Experimento", command=self.experiment)
        menu.add_command(label="Experimento 2", command=self.experiment_2)
        menu.add_command(label="Experimento 3", command=self.experiment_3)
        self.visualization_menu.configure(menu=menu)

        self.visualization_menu.pack(side=tk.LEFT, padx=20)

        self.compare_button = tk.Button(self.button_frame, text="Comparar Biometrías", bg="#FF5722", fg="white",
                                        font=("Arial", 12, "bold"), command=self.compare_images)
        self.compare_button.pack(side=tk.LEFT, padx=20)

        # Checkbox para modo estricto
        self.strict_mode_var = tk.BooleanVar(value=True)
        self.strict_mode_check = tk.Checkbutton(self.button_frame, text="Modo Estricto",
                                                variable=self.strict_mode_var, bg="#f0f0f0",
                                                font=("Arial", 11))
        self.strict_mode_check.pack(side=tk.LEFT, padx=10)

        # Botón de información sobre modos
        self.info_button = tk.Button(self.button_frame, text="ℹ", bg="#9E9E9E", fg="white",
                                     font=("Arial", 10), width=2, command=self.show_mode_info)
        self.info_button.pack(side=tk.LEFT, padx=5)

        # Botones para selección manual de imágenes
        self.select_master_button = tk.Button(self.button_frame, text="Seleccionar Subject 1", bg="#673AB7", fg="white",
                                              font=("Arial", 12), command=self.select_master_image)
        self.select_master_button.pack(side=tk.LEFT, padx=20)

        self.select_submitted_button = tk.Button(self.button_frame, text="Seleccionar Subject 2", bg="#3F51B5", fg="white",
                                                 font=("Arial", 12), command=self.select_submitted_image)
        self.select_submitted_button.pack(side=tk.LEFT, padx=20)

        # Zona de información adicional
        self.info_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.reviewer_label = tk.Label(self.info_frame, text="Reviewed by: System v2.0", font=("Arial", 12), bg="#f0f0f0")
        self.reviewer_label.pack(side=tk.LEFT, padx=20)

        self.review_date_label = tk.Label(self.info_frame, text="Review Date: --", font=("Arial", 12), bg="#f0f0f0")
        self.review_date_label.pack(side=tk.LEFT, padx=20)

        # Cargar imágenes automáticamente después de que la GUI esté lista
        self.root.after(100, self.load_master_image)
        self.root.after(200, self.load_submitted_image)

    def _get_master_landmarks(self):
        """Return cached landmarks for master image, compute once."""
        if self.master_landmarks is None and self.master_image is not None:
            self.master_landmarks = detect_landmarks(self.master_image)
        return self.master_landmarks

    def _get_submitted_landmarks(self):
        """Return cached landmarks for submitted image, compute once."""
        if self.submitted_landmarks is None and self.submitted_image is not None:
            self.submitted_landmarks = detect_landmarks(self.submitted_image)
        return self.submitted_landmarks

    def _get_triangles(self):
        """Return cached list of triangles from FACE_CONNECTIONS."""
        if self.triangles_cache is None:
            from collections import defaultdict
            adjacency = defaultdict(set)
            for v1, v2 in FACE_CONNECTIONS:
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)
            tris = set()
            for v, neighbors in adjacency.items():
                nbrs = list(neighbors)
                for i in range(len(nbrs)):
                    for j in range(i+1, len(nbrs)):
                        v2, v3 = nbrs[i], nbrs[j]
                        if v3 in adjacency[v2]:
                            tris.add(tuple(sorted([v, v2, v3])))
            self.triangles_cache = list(tris)
        return self.triangles_cache

    def load_master_image(self):
        """Cargar automáticamente la imagen de referencia desde la carpeta assets."""
        file_path = "assets/img1.png"
        try:
            self.master_image = cv2.imread(file_path)
            self.master_landmarks = None
            self.triangles_cache = None
            if self.master_image is None:
                messagebox.showerror("Error", f"No se pudo cargar la imagen de referencia desde {file_path}")
                return
            self.display_image(self.master_image, self.master_image_label)
            self.update_review_date()
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar imagen: {str(e)}")

    def load_submitted_image(self):
        """Cargar automáticamente la imagen a verificar desde la carpeta assets."""
        file_path = "assets/img2.png"
        try:
            self.submitted_image = cv2.imread(file_path)
            self.submitted_landmarks = None
            self.triangles_cache = None
            if self.submitted_image is None:
                messagebox.showerror("Error", f"No se pudo cargar la imagen a verificar desde {file_path}")
                return
            self.display_image(self.submitted_image, self.submitted_image_label)
            self.compare_images()
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar imagen: {str(e)}")

    def select_master_image(self):
        """Permite seleccionar manualmente la imagen de Subject 1."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.master_image = cv2.imread(file_path)
            self.master_landmarks = None
            self.triangles_cache = None
            if self.master_image is None:
                messagebox.showerror("Error", f"No se pudo cargar la imagen desde {file_path}")
                return
            self.display_image(self.master_image, self.master_image_label)
            if self.submitted_image is not None:
                self.compare_images()

    def select_submitted_image(self):
        """Permite seleccionar manualmente la imagen de Subject 2."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.submitted_image = cv2.imread(file_path)
            self.submitted_landmarks = None
            self.triangles_cache = None
            if self.submitted_image is None:
                messagebox.showerror("Error", f"No se pudo cargar la imagen desde {file_path}")
                return
            self.display_image(self.submitted_image, self.submitted_image_label)
            if self.master_image is not None:
                self.compare_images()

    def display_image(self, image, label):
        """Muestra una imagen en el label especificado."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Obtener el tamaño actual del label
        label.update()
        max_width = max(label.winfo_width(), 400)
        max_height = max(label.winfo_height(), 300)

        # Escalar proporcionalmente
        original_width, original_height = image_pil.size
        scale = min(max_width / original_width, max_height / original_height, 1.0)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        image_pil = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convertir a formato compatible con Tkinter
        image_tk = ImageTk.PhotoImage(image_pil)
        label.configure(image=image_tk)
        label.image = image_tk

    def compare_images(self):
        """Compara las imágenes cargadas usando el sistema mejorado."""
        # Validar que existan imágenes cargadas
        if self.master_image is None or self.submitted_image is None:
            return

        try:
            # Detectar landmarks
            landmarks1 = self._get_master_landmarks()
            landmarks2 = self._get_submitted_landmarks()

            # Verificar identidad usando el sistema avanzado
            strict_mode = self.strict_mode_var.get()
            self.verification_results = verify_identity(landmarks1, landmarks2, strict_mode=strict_mode)

            # Actualizar la interfaz con los resultados
            self.update_results_display()

        except Exception as e:
            messagebox.showerror("Error", f"Error en la verificación: {str(e)}")
            self.features_text.delete(1.0, tk.END)
            self.features_text.insert(tk.END, f"Error: {str(e)}\n")

    def show_mode_info(self):
        """Muestra información sobre los modos de operación."""
        info_text = """
MODOS DE OPERACIÓN:

Modo Estricto (Recomendado):
• Mayor precisión en la verificación
• Reduce falsos positivos
• Ideal para seguridad alta
• Puede requerir mejor calidad de imagen

Modo Normal:
• Más tolerante a variaciones
• Mejor para fotos con diferentes:
  - Iluminación
  - Expresiones faciales
  - Ángulos de cámara
• Mayor riesgo de falsos positivos

RECOMENDACIONES:
• Use Modo Estricto para verificación de seguridad
• Use Modo Normal si las fotos tienen condiciones diferentes
• Para mejores resultados, use fotos con:
  - Buena iluminación
  - Rostro frontal
  - Expresión neutral
"""
        messagebox.showinfo("Información de Modos", info_text)

    def experiment(self):
            """Realiza un experimento para evaluar el sistema."""
            # Paso 1: Construir un grafo de adyacencia desde las conexiones
            from collections import defaultdict
            import itertools

            # Crear grafo de adyacencia
            adjacency = defaultdict(set)
            for edge in FACE_CONNECTIONS:
                v1, v2 = edge
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)

            # Paso 2: Encontrar todos los triángulos (ciclos de 3 vértices)
            triangles = set()
            for v1 in adjacency:
                neighbors = list(adjacency[v1])
                # Buscar pares de vecinos que también estén conectados entre sí
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        v2, v3 = neighbors[i], neighbors[j]
                        # Si v2 y v3 están conectados, tenemos un triángulo
                        if v3 in adjacency[v2]:
                            # Ordenar vértices para evitar duplicados
                            triangle = tuple(sorted([v1, v2, v3]))
                            triangles.add(triangle)

            triangles = list(triangles)
            print(f"Número total de triángulos encontrados: {len(triangles)}")
            print(f"Primeros 10 triángulos: {triangles[:10]}")

            # Paso 3: Detectar landmarks en ambas imágenes
            landmarks1 = self._get_master_landmarks()
            landmarks2 = self._get_submitted_landmarks()

            # Paso 4: Función para mapear un triángulo de una cara a otra
            def map_triangle(triangle_indices, source_landmarks, target_landmarks):
                """
                Mapea un triángulo de los landmarks fuente a los landmarks objetivo.

                Args:
                    triangle_indices: tupla de 3 índices del triángulo
                    source_landmarks: lista de landmarks de la imagen fuente
                    target_landmarks: lista de landmarks de la imagen objetivo

                Returns:
                    tupla con las coordenadas del triángulo en ambas imágenes
                """
                # Obtener puntos del triángulo en imagen fuente
                src_points = np.array([source_landmarks[idx] for idx in triangle_indices], dtype=np.float32)

                # Obtener puntos correspondientes en imagen objetivo
                dst_points = np.array([target_landmarks[idx] for idx in triangle_indices], dtype=np.float32)

                return src_points, dst_points

            # Paso 5: Ejemplo de mapeo de triángulos
            if len(triangles) > 0 and len(landmarks1) > 0 and len(landmarks2) > 0:
                # Tomar el primer triángulo como ejemplo
                example_triangle = triangles[0]
                print(f"\nEjemplo de mapeo con triángulo {example_triangle}:")

                src_tri, dst_tri = map_triangle(example_triangle, landmarks1, landmarks2)

                print(f"Puntos en imagen 1: {src_tri}")
                print(f"Puntos en imagen 2: {dst_tri}")

                # Paso 6: Calcular transformación afín entre triángulos
                # Esto es útil para warping o análisis de deformación
                affine_matrix = cv2.getAffineTransform(src_tri, dst_tri)
                print(f"\nMatriz de transformación afín:\n{affine_matrix}")

                # Paso 7: Visualizar algunos triángulos en ambas imágenes
                img1_copy = self.master_image.copy()
                img2_copy = self.submitted_image.copy()

                # Dibujar los primeros 20 triángulos
                for i, triangle in enumerate(triangles[:20]):
                    # Color aleatorio para cada triángulo
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

                    # Verificar que todos los índices estén en rango
                    if all(idx < len(landmarks1) and idx < len(landmarks2) for idx in triangle):
                        # Dibujar en imagen 1
                        pts1 = np.array([landmarks1[idx] for idx in triangle], np.int32)
                        cv2.drawContours(img1_copy, [pts1], 0, color, 2)

                        # Dibujar en imagen 2
                        pts2 = np.array([landmarks2[idx] for idx in triangle], np.int32)
                        cv2.drawContours(img2_copy, [pts2], 0, color, 2)

                # Mostrar imágenes con triángulos
                cv2.imwrite("triangles_image1.png", img1_copy)
                cv2.imwrite("triangles_image2.png", img2_copy)
                print("Imágenes guardadas como triangles_image1.png y triangles_image2.png")

                # Paso 8: Análisis de deformación entre caras
                print("\nAnálisis de deformación:")
                deformations = []
                for triangle in triangles[:10]:  # Analizar primeros 10 triángulos
                    if all(idx < len(landmarks1) and idx < len(landmarks2) for idx in triangle):
                        src_tri, dst_tri = map_triangle(triangle, landmarks1, landmarks2)

                        # Calcular área de cada triángulo
                        area1 = cv2.contourArea(src_tri)
                        area2 = cv2.contourArea(dst_tri)

                        # Calcular cambio de área (deformación)
                        if area1 > 0:
                            deformation = abs(area2 - area1) / area1
                            deformations.append(deformation)
                            print(f"Triángulo {triangle}: deformación = {deformation:.3f}")

                if deformations:
                    avg_deformation = np.mean(deformations)
                    print(f"\nDeformación promedio: {avg_deformation:.3f}")

            # Paso 9: Transferir el primer triángulo de una cara a la otra
            if len(triangles) > 0:
                print("\n=== TRANSFERENCIA DE TRIÁNGULO ===")
                first_triangle = triangles[0]
                print(f"Transfiriendo triángulo {first_triangle}")

                # Obtener los puntos del triángulo en ambas caras
                src_tri, dst_tri = map_triangle(first_triangle, landmarks1, landmarks2)

                # Crear máscaras para el triángulo
                mask1 = np.zeros(self.master_image.shape[:2], dtype=np.uint8)
                mask2 = np.zeros(self.submitted_image.shape[:2], dtype=np.uint8)

                # Dibujar el triángulo en las máscaras
                cv2.fillPoly(mask1, [src_tri.astype(np.int32)], 255)
                cv2.fillPoly(mask2, [dst_tri.astype(np.int32)], 255)

                # Calcular el rectángulo delimitador para cada triángulo
                rect1 = cv2.boundingRect(src_tri.astype(np.int32))
                rect2 = cv2.boundingRect(dst_tri.astype(np.int32))

                # Extraer la región del triángulo de la imagen 1
                x1, y1, w1, h1 = rect1
                triangle_crop1 = self.master_image[y1:y1+h1, x1:x1+w1].copy()
                mask_crop1 = mask1[y1:y1+h1, x1:x1+w1]

                # Ajustar las coordenadas del triángulo al crop
                src_tri_cropped = src_tri - [x1, y1]
                dst_tri_cropped = dst_tri - [rect2[0], rect2[1]]

                # Calcular la transformación afín
                warp_mat = cv2.getAffineTransform(
                    src_tri_cropped.astype(np.float32),
                    dst_tri_cropped.astype(np.float32)
                )

                # Aplicar la transformación al triángulo recortado
                warped_triangle = cv2.warpAffine(
                    triangle_crop1,
                    warp_mat,
                    (rect2[2], rect2[3]),
                    flags=cv2.INTER_LINEAR
                )

                # Crear la máscara para el triángulo transformado
                warped_mask = cv2.warpAffine(
                    mask_crop1,
                    warp_mat,
                    (rect2[2], rect2[3])
                )

                # Crear una copia de la imagen 2 para mostrar el resultado
                result_image = self.submitted_image.copy()

                # Aplicar el triángulo transformado a la imagen 2
                x2, y2, w2, h2 = rect2

                # Asegurarse de que las coordenadas estén dentro de los límites
                for y in range(h2):
                    for x in range(w2):
                        if (y2 + y < result_image.shape[0] and
                            x2 + x < result_image.shape[1] and
                            y < warped_mask.shape[0] and
                            x < warped_mask.shape[1]):
                            if warped_mask[y, x] > 0:
                                result_image[y2 + y, x2 + x] = warped_triangle[y, x]

                # Visualizar el proceso
                # 1. Imagen original con el triángulo marcado
                img1_triangle = self.master_image.copy()
                cv2.drawContours(img1_triangle, [src_tri.astype(np.int32)], 0, (0, 255, 0), 3)

                # 2. Imagen destino con el triángulo original marcado
                img2_triangle = self.submitted_image.copy()
                cv2.drawContours(img2_triangle, [dst_tri.astype(np.int32)], 0, (0, 0, 255), 3)

                # 3. Resultado con el triángulo transferido
                cv2.drawContours(result_image, [dst_tri.astype(np.int32)], 0, (255, 0, 0), 2)

                # Guardar las imágenes
                cv2.imwrite("transfer_1_source.png", img1_triangle)
                cv2.imwrite("transfer_2_destination.png", img2_triangle)
                cv2.imwrite("transfer_3_result.png", result_image)

                print("\nImágenes guardadas:")
                print("- transfer_1_source.png: Imagen 1 con el triángulo fuente (verde)")
                print("- transfer_2_destination.png: Imagen 2 con el triángulo destino (rojo)")
                print("- transfer_3_result.png: Imagen 2 con el triángulo transferido desde imagen 1")

                # Información adicional
                print(f"\nÁrea del triángulo en imagen 1: {cv2.contourArea(src_tri):.2f} píxeles²")
                print(f"Área del triángulo en imagen 2: {cv2.contourArea(dst_tri):.2f} píxeles²")
                print(f"Factor de escala: {cv2.contourArea(dst_tri)/cv2.contourArea(src_tri):.3f}")

            # Paso 10: Transferir los 10 triángulos marcados
            if len(triangles) >= 10:
                print("\n=== TRANSFERENCIA DE LOS 10 TRIÁNGULOS MARCADOS ===")

                # Crear una copia de la imagen 2 para el resultado
                result_image = self.submitted_image.copy()

                # Procesar los primeros 10 triángulos (los mismos que se dibujaron)
                for i, triangle in enumerate(triangles[:10]):
                    if all(idx < len(landmarks1) and idx < len(landmarks2) for idx in triangle):
                        print(f"\nProcesando triángulo {i+1}/10: {triangle}")

                        # Obtener los puntos del triángulo en ambas caras
                        src_tri, dst_tri = map_triangle(triangle, landmarks1, landmarks2)

                        # Crear máscaras para el triángulo
                        mask1 = np.zeros(self.master_image.shape[:2], dtype=np.uint8)
                        mask2 = np.zeros(self.submitted_image.shape[:2], dtype=np.uint8)

                        # Dibujar el triángulo en las máscaras
                        cv2.fillPoly(mask1, [src_tri.astype(np.int32)], 255)
                        cv2.fillPoly(mask2, [dst_tri.astype(np.int32)], 255)

                        # Calcular el rectángulo delimitador para cada triángulo
                        rect1 = cv2.boundingRect(src_tri.astype(np.int32))
                        rect2 = cv2.boundingRect(dst_tri.astype(np.int32))

                        # Extraer la región del triángulo de la imagen 1
                        x1, y1, w1, h1 = rect1
                        triangle_crop1 = self.master_image[y1:y1+h1, x1:x1+w1].copy()
                        mask_crop1 = mask1[y1:y1+h1, x1:x1+w1]

                        # Ajustar las coordenadas del triángulo al crop
                        src_tri_cropped = src_tri - [x1, y1]
                        dst_tri_cropped = dst_tri - [rect2[0], rect2[1]]

                        # Calcular la transformación afín
                        warp_mat = cv2.getAffineTransform(
                            src_tri_cropped.astype(np.float32),
                            dst_tri_cropped.astype(np.float32)
                        )

                        # Aplicar la transformación al triángulo recortado
                        warped_triangle = cv2.warpAffine(
                            triangle_crop1,
                            warp_mat,
                            (rect2[2], rect2[3]),
                            flags=cv2.INTER_LINEAR
                        )

                        # Crear la máscara para el triángulo transformado
                        warped_mask = cv2.warpAffine(
                            mask_crop1,
                            warp_mat,
                            (rect2[2], rect2[3])
                        )

                        # Aplicar el triángulo transformado a la imagen resultado
                        x2, y2, w2, h2 = rect2

                        # Transferir píxel por píxel
                        for y in range(h2):
                            for x in range(w2):
                                if (y2 + y < result_image.shape[0] and
                                    x2 + x < result_image.shape[1] and
                                    y < warped_mask.shape[0] and
                                    x < warped_mask.shape[1]):
                                    if warped_mask[y, x] > 0:
                                        result_image[y2 + y, x2 + x] = warped_triangle[y, x]

                        # Calcular y mostrar información del triángulo
                        area1 = cv2.contourArea(src_tri)
                        area2 = cv2.contourArea(dst_tri)
                        if area1 > 0:
                            scale_factor = area2 / area1
                            print(f"  - Área en imagen 1: {area1:.1f} px²")
                            print(f"  - Área en imagen 2: {area2:.1f} px²")
                            print(f"  - Factor de escala: {scale_factor:.3f}")

                # Crear imagen de comparación con los triángulos marcados
                comparison_img1 = self.master_image.copy()
                comparison_img2 = result_image.copy()

                # Dibujar contornos de los 10 triángulos transferidos
                for i, triangle in enumerate(triangles[:10]):
                    if all(idx < len(landmarks1) and idx < len(landmarks2) for idx in triangle):
                        # Color para identificar cada triángulo
                        color = (0, 255, 0)  # Verde para source
                        pts1 = np.array([landmarks1[idx] for idx in triangle], np.int32)
                        cv2.drawContours(comparison_img1, [pts1], 0, color, 2)

                        # Azul para destino
                        color = (255, 0, 0)
                        pts2 = np.array([landmarks2[idx] for idx in triangle], np.int32)
                        cv2.drawContours(comparison_img2, [pts2], 0, color, 2)

                # Guardar las imágenes
                cv2.imwrite("transfer_10_triangles_source.png", comparison_img1)
                cv2.imwrite("transfer_10_triangles_result.png", comparison_img2)

                # También guardar el resultado sin contornos
                cv2.imwrite("transfer_10_triangles_clean.png", result_image)

                print("\n=== TRANSFERENCIA COMPLETADA ===")
                print("\nImágenes guardadas:")
                print("- transfer_10_triangles_source.png: Imagen 1 con los 10 triángulos fuente marcados")
                print("- transfer_10_triangles_result.png: Resultado con los triángulos transferidos y marcados")
                print("- transfer_10_triangles_clean.png: Resultado sin marcas (solo la transferencia)")

                # Crear una imagen lado a lado para comparación
                h1, w1 = self.master_image.shape[:2]
                h2, w2 = self.submitted_image.shape[:2]
                max_height = max(h1, h2)

                # Crear canvas para comparación
                comparison = np.zeros((max_height, w1 + w2 + 10, 3), dtype=np.uint8)
                comparison[:h1, :w1] = comparison_img1
                comparison[:h2, w1+10:] = comparison_img2

                # Añadir línea divisoria
                comparison[:, w1:w1+10] = [128, 128, 128]

                cv2.imwrite("transfer_10_triangles_comparison.png", comparison)
                print("- transfer_10_triangles_comparison.png: Comparación lado a lado")
            else:
                print("No se pudieron detectar landmarks o triángulos.")

    def experiment_2(self):
        if self.master_image is None or self.submitted_image is None:
            messagebox.showerror("Error", "Ambas imágenes deben estar cargadas.")
            return

        exp_window = tk.Toplevel(self.root)
        exp_window.title("Experiment 2: Selección de Triángulos")

        # Convertir imágenes para Tkinter
        image_rgb = cv2.cvtColor(self.master_image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        master_photo = ImageTk.PhotoImage(pil_image)

        image_rgb2 = cv2.cvtColor(self.submitted_image, cv2.COLOR_BGR2RGB)
        pil_image2 = Image.fromarray(image_rgb2)
        submitted_photo = ImageTk.PhotoImage(pil_image2)

        # Crear canvases para ambas imágenes
        canvas_master = tk.Canvas(exp_window, width=pil_image.width, height=pil_image.height)
        canvas_master.grid(row=0, column=0, padx=10, pady=10)
        canvas_master.create_image(0, 0, anchor=tk.NW, image=master_photo)
        canvas_master.image = master_photo

        canvas_submitted = tk.Canvas(exp_window, width=pil_image2.width, height=pil_image2.height)
        canvas_submitted.grid(row=0, column=1, padx=10, pady=10)
        canvas_submitted.create_image(0, 0, anchor=tk.NW, image=submitted_photo)
        canvas_submitted.image = submitted_photo

        # Extraer landmarks de ambas imágenes
        landmarks_master = detect_landmarks(self.master_image)
        landmarks_submitted = detect_landmarks(self.submitted_image)

        # Construir grafo de adyacencia a partir de FACE_CONNECTIONS (usando landmarks del master)
        from collections import defaultdict
        adjacency = defaultdict(set)
        for edge in FACE_CONNECTIONS:
            v1, v2 = edge
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
        triangles = set()
        for v in adjacency:
            neighbors = list(adjacency[v])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    v2, v3 = neighbors[i], neighbors[j]
                    if v3 in adjacency[v2]:
                        triangle = tuple(sorted([v, v2, v3]))
                        triangles.add(triangle)
        triangles = list(triangles)

        # Diccionario para almacenar selección de cada triángulo
        selected = {tri: False for tri in triangles}

        # Diccionario para mapear cada triángulo a sus polígonos en ambos canvases
        triangles_mapping = {}
        for tri in triangles:
            pts_master = [landmarks_master[idx] for idx in tri]
            pts_submitted = [landmarks_submitted[idx] for idx in tri]
            flat_pts_master = [coord for point in pts_master for coord in point]
            flat_pts_submitted = [coord for point in pts_submitted for coord in point]
            poly_master = canvas_master.create_polygon(flat_pts_master, outline="blue", fill="", width=1)
            poly_submitted = canvas_submitted.create_polygon(flat_pts_submitted, outline="blue", fill="", width=1)
            triangles_mapping[tri] = {"master": poly_master, "submitted": poly_submitted}

        # Funciones de manejo de eventos
        def on_enter(event, tri_key):
            if not selected[tri_key]:
                canvas_master.itemconfig(triangles_mapping[tri_key]["master"], fill="#d3d3d3")
                canvas_submitted.itemconfig(triangles_mapping[tri_key]["submitted"], fill="#d3d3d3")

        def on_leave(event, tri_key):
            if not selected[tri_key]:
                canvas_master.itemconfig(triangles_mapping[tri_key]["master"], fill="")
                canvas_submitted.itemconfig(triangles_mapping[tri_key]["submitted"], fill="")

        def on_click(event, tri_key):
            selected[tri_key] = not selected[tri_key]
            new_color = "#00ff00" if selected[tri_key] else ""
            canvas_master.itemconfig(triangles_mapping[tri_key]["master"], fill=new_color)
            canvas_submitted.itemconfig(triangles_mapping[tri_key]["submitted"], fill=new_color)
            if selected[tri_key]:
                print(f"Triangle {tri_key} selected")
            else:
                print(f"Triangle {tri_key} deselected")

        # Vincular eventos a cada triángulo en ambos canvases
        for tri_key, mapping in triangles_mapping.items():
            canvas_master.tag_bind(mapping["master"], "<Enter>", lambda e, key=tri_key: on_enter(e, key))
            canvas_master.tag_bind(mapping["master"], "<Leave>", lambda e, key=tri_key: on_leave(e, key))
            canvas_master.tag_bind(mapping["master"], "<Button-1>", lambda e, key=tri_key: on_click(e, key))
            canvas_submitted.tag_bind(mapping["submitted"], "<Enter>", lambda e, key=tri_key: on_enter(e, key))
            canvas_submitted.tag_bind(mapping["submitted"], "<Leave>", lambda e, key=tri_key: on_leave(e, key))
            canvas_submitted.tag_bind(mapping["submitted"], "<Button-1>", lambda e, key=tri_key: on_click(e, key))

        # Botones Aceptar y Cancelar
        btn_frame = tk.Frame(exp_window)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)

        def on_cancel():
            for tri_key in selected.keys():
                selected[tri_key] = False
                canvas_master.itemconfig(triangles_mapping[tri_key]["master"], fill="")
                canvas_submitted.itemconfig(triangles_mapping[tri_key]["submitted"], fill="")
            exp_window.destroy()
            print("Selección cancelada. Triángulos deseleccionados.")

        def on_accept():
            updated_master = self.master_image.copy()
            updated_submitted = self.submitted_image.copy()
            for tri_key, is_selected in selected.items():
                if is_selected:
                    pts_master = np.array([landmarks_master[idx] for idx in tri_key], np.int32)
                    pts_submitted = np.array([landmarks_submitted[idx] for idx in tri_key], np.int32)

                    rect_master = cv2.boundingRect(pts_master)
                    x1, y1, w1, h1 = rect_master
                    rect_submitted = cv2.boundingRect(pts_submitted)
                    x2, y2, w2, h2 = rect_submitted

                    triangle_crop_master = self.master_image[y1:y1+h1, x1:x1+w1].copy()
                    mask_master = np.zeros((h1, w1), dtype=np.uint8)
                    pts_master_shifted = pts_master - [x1, y1]
                    cv2.fillPoly(mask_master, [pts_master_shifted], 255)

                    triangle_crop_submitted = self.submitted_image[y2:y2+h2, x2:x2+w2].copy()
                    mask_submitted = np.zeros((h2, w2), dtype=np.uint8)
                    pts_submitted_shifted = pts_submitted - [x2, y2]
                    cv2.fillPoly(mask_submitted, [pts_submitted_shifted], 255)

                    pts_master_shifted = np.float32(pts_master_shifted)
                    pts_submitted_shifted = np.float32(pts_submitted_shifted)
                    if pts_master_shifted.shape[0] >= 3 and pts_submitted_shifted.shape[0] >= 3:
                        M_sub_to_master = cv2.getAffineTransform(pts_submitted_shifted[:3], pts_master_shifted[:3])
                        M_master_to_sub = cv2.getAffineTransform(pts_master_shifted[:3], pts_submitted_shifted[:3])

                        warped_submitted = cv2.warpAffine(triangle_crop_submitted, M_sub_to_master, (w1, h1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                        warped_master = cv2.warpAffine(triangle_crop_master, M_master_to_sub, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

                        for i in range(h1):
                            for j in range(w1):
                                if mask_master[i, j] > 0:
                                    updated_master[y1 + i, x1 + j] = warped_submitted[i, j]
                        for i in range(h2):
                            for j in range(w2):
                                if mask_submitted[i, j] > 0:
                                    updated_submitted[y2 + i, x2 + j] = warped_master[i, j]
            self.master_image = updated_master
            self.submitted_image = updated_submitted
            self.display_image(self.master_image, self.master_image_label)
            self.display_image(self.submitted_image, self.submitted_image_label)
            print("Transferencia completada para triángulos seleccionados.")
            exp_window.destroy()

        btn_accept = tk.Button(btn_frame, text="Aceptar", command=on_accept, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        btn_accept.pack(side=tk.LEFT, padx=20)
        btn_cancel = tk.Button(btn_frame, text="Cancelar", command=on_cancel, bg="#F44336", fg="white", font=("Arial", 12, "bold"))
        btn_cancel.pack(side=tk.RIGHT, padx=5)

    def experiment_3(self):
        """Experiment 3: Face swap using Delaunay triangulation and seamless cloning."""
        # Ensure both images are loaded
        if self.master_image is None or self.submitted_image is None:
            messagebox.showerror("Error", "Both images must be loaded for face swap.")
            return
        # Keep originals for warping
        orig_master = self.master_image.copy()
        orig_submitted = self.submitted_image.copy()
        # Get landmarks
        lm1 = np.array(self._get_master_landmarks(), np.int32)
        lm2 = np.array(self._get_submitted_landmarks(), np.int32)
        # Delaunay triangulation on master landmarks
        from scipy.spatial import Delaunay
        tri1 = Delaunay(lm1)
        triangles1 = tri1.simplices
        # separate triangulation for reverse warp
        tri2 = Delaunay(lm2)
        triangles2 = tri2.simplices
        # Prepare blank warped face images
        h1, w1 = orig_master.shape[:2]
        h2, w2 = orig_submitted.shape[:2]
        face_warp_to_submitted = np.zeros_like(orig_submitted)
        face_warp_to_master = np.zeros_like(orig_master)

        # Warp helper
        def warp_triangle(img_src, img_dst, t_src, t_dst):
            r1 = cv2.boundingRect(np.float32([t_src]))
            r2 = cv2.boundingRect(np.float32([t_dst]))
            x1, y1, w1_, h1_ = r1
            x2, y2, w2_, h2_ = r2
            src_rect = np.array([[pt[0]-x1, pt[1]-y1] for pt in t_src], np.float32)
            dst_rect = np.array([[pt[0]-x2, pt[1]-y2] for pt in t_dst], np.float32)
            crop = img_src[y1:y1+h1_, x1:x1+w1_]
            M = cv2.getAffineTransform(src_rect, dst_rect)
            warped = cv2.warpAffine(crop, M, (w2_, h2_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = np.zeros((h2_, w2_), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst_rect), 255)
            dst_roi = img_dst[y2:y2+h2_, x2:x2+w2_]
            dst_roi[mask==255] = warped[mask==255]

        # Warp master -> submitted
        for tri_idx in triangles1:
            warp_triangle(orig_master, face_warp_to_submitted, lm1[tri_idx], lm2[tri_idx])
        # Warp submitted -> master
        for tri_idx in triangles2:
            warp_triangle(orig_submitted, face_warp_to_master, lm2[tri_idx], lm1[tri_idx])

        # Seamless clone into submitted
        hull2 = cv2.convexHull(lm2)
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
        cv2.fillConvexPoly(mask2, hull2, 255)
        rc2 = cv2.boundingRect(hull2)
        center2 = (rc2[0] + rc2[2]//2, rc2[1] + rc2[3]//2)
        swapped_submitted = cv2.seamlessClone(face_warp_to_submitted, orig_submitted, mask2, center2, cv2.NORMAL_CLONE)
        print("Experiment 3: swapped_submitted applied")

        # Seamless clone into master
        hull1 = cv2.convexHull(lm1)
        mask1 = np.zeros((h1, w1), dtype=np.uint8)
        cv2.fillConvexPoly(mask1, hull1, 255)
        rc1 = cv2.boundingRect(hull1)
        center1 = (rc1[0] + rc1[2]//2, rc1[1] + rc1[3]//2)
        swapped_master = cv2.seamlessClone(face_warp_to_master, orig_master, mask1, center1, cv2.NORMAL_CLONE)
        print("Experiment 3: swapped_master applied")

        # Update images and display both faces
        self.master_image = swapped_master
        self.submitted_image = swapped_submitted
        # First update the submitted panel, then master panel
        self.display_image(self.submitted_image, self.submitted_image_label)
        self.display_image(self.master_image, self.master_image_label)
        # Clear caches
        self.master_landmarks = None
        self.submitted_landmarks = None
        messagebox.showinfo("Experiment 3", "Face swap completed in both images.")

    def update_results_display(self):
        """Actualiza la visualización de resultados en la interfaz."""
        if not self.verification_results:
            return

        # Extraer resultados
        is_same = self.verification_results['is_same_person']
        confidence = self.verification_results['confidence']
        similarity = self.verification_results['similarity_score']

        # Actualizar etiquetas principales con información adicional
        conclusion = "MISMA PERSONA" if is_same else "PERSONA DIFERENTE"
        color = "#4CAF50" if is_same else "#F44336"

        # Agregar indicador de confianza visual
        if confidence > 0.8:
            confidence_text = "Alta"
            conf_color = "#4CAF50"
        elif confidence > 0.6:
            confidence_text = "Media"
            conf_color = "#FF9800"
        else:
            confidence_text = "Baja"
            conf_color = "#F44336"

        self.metrics_label.config(
            text=f"Resultado: {conclusion} (Similitud: {similarity:.4f})",
            fg=color
        )

        self.confidence_label.config(
            text=f"Confianza: {confidence:.1%} ({confidence_text})",
            fg=conf_color
        )

        self.timestamp_label.config(
            text=f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Actualizar panel de análisis detallado
        self.update_detailed_analysis()

    def update_detailed_analysis(self):
        """Actualiza el panel de análisis detallado."""
        self.features_text.delete(1.0, tk.END)

        # Título
        self.features_text.insert(tk.END, "=== ANÁLISIS BIOMÉTRICO DETALLADO ===\n\n", "title")

        # Resultados por método
        self.features_text.insert(tk.END, "RESULTADOS POR MÉTODO:\n", "subtitle")
        methods = self.verification_results['methods']

        for method_name, results in methods.items():
            match_text = "✓ Match" if results['match'] else "✗ No Match"
            self.features_text.insert(tk.END,
                f"  {method_name.capitalize():12} Similitud: {results['similarity']:.4f} {match_text}\n")

        # Extraer características para comparación
        try:
            landmarks1 = self._get_master_landmarks()
            landmarks2 = self._get_submitted_landmarks()
            features1 = extract_geometric_features(landmarks1)
            features2 = extract_geometric_features(landmarks2)

            # Comparación de características
            self.features_text.insert(tk.END, "\n\nCOMPARACIÓN DE CARACTERÍSTICAS:\n", "subtitle")
            self.features_text.insert(tk.END, "-" * 60 + "\n")

            labels = get_feature_labels()
            differences = np.abs(features1 - features2)

            # Ordenar por diferencia para mostrar las más significativas primero
            sorted_indices = np.argsort(differences)[::-1]

            for idx in sorted_indices[:10]:  # Mostrar las 10 más significativas
                label = labels[idx]
                f1 = features1[idx]
                f2 = features2[idx]
                diff = differences[idx]

                # Colorear según la magnitud de la diferencia
                if diff > 0.1:
                    tag = "high_diff"
                elif diff > 0.05:
                    tag = "medium_diff"
                else:
                    tag = "low_diff"

                self.features_text.insert(tk.END,
                    f"{label:25} A: {f1:7.4f}  B: {f2:7.4f}  Δ: {diff:7.4f}\n", tag)

            # Métricas generales
            self.features_text.insert(tk.END, "\n\nMÉTRICAS GENERALES:\n", "subtitle")
            self.features_text.insert(tk.END, "-" * 60 + "\n")

            euclidean_distance = np.linalg.norm(features1 - features2)
            cosine_similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))

            self.features_text.insert(tk.END, f"Distancia Euclidiana total: {euclidean_distance:.4f}\n")
            self.features_text.insert(tk.END, f"Similitud por coseno: {cosine_similarity:.4f}\n")
            self.features_text.insert(tk.END, f"Diferencia promedio: {np.mean(differences):.4f}\n")
            self.features_text.insert(tk.END, f"Diferencia máxima: {np.max(differences):.4f}\n")

            # Recomendación
            self.features_text.insert(tk.END, "\n\nRECOMENDACIÓN:\n", "subtitle")
            if self.verification_results['is_same_person']:
                self.features_text.insert(tk.END,
                    "✓ Las características biométricas indican que es la MISMA persona.\n", "success")
            else:
                self.features_text.insert(tk.END,
                    "✗ Las características biométricas indican que son personas DIFERENTES.\n", "error")

            self.features_text.insert(tk.END,
                f"\nNivel de confianza: {self.verification_results['confidence']:.1%}\n")

            # Debug info para casos especiales
            if 'debug' in self.verification_results:
                debug = self.verification_results['debug']
                if debug['is_identical']:
                    self.features_text.insert(tk.END,
                        "\n[DEBUG] Imágenes idénticas detectadas.\n", "debug")

                self.features_text.insert(tk.END,
                    f"[DEBUG] Similitud bruta: {debug['raw_similarity']:.4f}\n", "debug")
                self.features_text.insert(tk.END,
                    f"[DEBUG] Varianza de características: {debug['feature_variance']:.6f}\n", "debug")

        except Exception as e:
            self.features_text.insert(tk.END, f"\nError al calcular características: {str(e)}\n", "error")

        # Configurar tags de color
        self.features_text.tag_config("title", font=("Arial", 12, "bold"), foreground="#1976D2")
        self.features_text.tag_config("subtitle", font=("Arial", 10, "bold"), foreground="#424242")
        self.features_text.tag_config("high_diff", foreground="#D32F2F")
        self.features_text.tag_config("medium_diff", foreground="#F57C00")
        self.features_text.tag_config("low_diff", foreground="#388E3C")
        self.features_text.tag_config("success", foreground="#2E7D32", font=("Arial", 11, "bold"))
        self.features_text.tag_config("error", foreground="#C62828", font=("Arial", 11, "bold"))
        self.features_text.tag_config("debug", foreground="#757575", font=("Courier", 9))

    def show_comparative_analysis(self):
        """Muestra un análisis comparativo visual de las características."""
        if not self.verification_results:
            messagebox.showwarning("Advertencia", "Primero debe realizar una comparación.")
            return

        # Crear nueva ventana para el análisis
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Análisis Comparativo Visual")
        analysis_window.geometry("800x600")

        try:
            # Extraer características
            landmarks1 = detect_landmarks(self.master_image)
            landmarks2 = detect_landmarks(self.submitted_image)
            features1 = extract_geometric_features(landmarks1)
            features2 = extract_geometric_features(landmarks2)

            # Crear gráfico
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Gráfico de barras comparativo
            labels = get_feature_labels()[:10]  # Primeras 10 características
            x = np.arange(len(labels))
            width = 0.35

            ax1.bar(x - width/2, features1[:10], width, label='Subject 1', alpha=0.8)
            ax1.bar(x + width/2, features2[:10], width, label='Subject 2', alpha=0.8)
            ax1.set_xlabel('Características')
            ax1.set_ylabel('Valor normalizado')
            ax1.set_title('Comparación de Características Biométricas')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Gráfico de diferencias
            differences = np.abs(features1 - features2)
            ax2.bar(range(len(differences)), differences, color=['red' if d > 0.1 else 'orange' if d > 0.05 else 'green' for d in differences])
            ax2.set_xlabel('Índice de característica')
            ax2.set_ylabel('Diferencia absoluta')
            ax2.set_title('Magnitud de Diferencias por Característica')
            ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Umbral alto')
            ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Umbral medio')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Mostrar en la ventana
            canvas = FigureCanvasTkAgg(fig, master=analysis_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Error al generar análisis visual: {str(e)}")

    def show_original_images(self):
        """Muestra las imágenes originales."""
        if self.master_image is not None:
            self.display_image(self.master_image, self.master_image_label)
        if self.submitted_image is not None:
            self.display_image(self.submitted_image, self.submitted_image_label)

    def show_landmarks(self):
        """Muestra las imágenes con landmarks."""
        try:
            if self.master_image is not None:
                landmarks1 = detect_landmarks(self.master_image)
                master_with_landmarks = visualize_landmarks(self.master_image.copy(), landmarks1)
                self.display_image(master_with_landmarks, self.master_image_label)

            if self.submitted_image is not None:
                landmarks2 = detect_landmarks(self.submitted_image)
                submitted_with_landmarks = visualize_landmarks(self.submitted_image.copy(), landmarks2)
                self.display_image(submitted_with_landmarks, self.submitted_image_label)
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar landmarks: {str(e)}")

    def highlight_discrepancies(self):
        """Resalta los landmarks con mayores discrepancias."""
        if self.master_image is None or self.submitted_image is None:
            messagebox.showerror("Error", "Ambas imágenes deben estar cargadas.")
            return

        try:
            landmarks1 = detect_landmarks(self.master_image)
            landmarks2 = detect_landmarks(self.submitted_image)

            # Calcular discrepancias
            discrepancies = []
            for l1, l2 in zip(landmarks1, landmarks2):
                dist = np.linalg.norm(np.array(l1) - np.array(l2))
                discrepancies.append(dist)

            # Encontrar los índices con mayores discrepancias
            discrepancies = np.array(discrepancies)
            threshold = np.percentile(discrepancies, 90)  # Top 10%

            # Visualizar
            master_highlighted = self.master_image.copy()
            submitted_highlighted = self.submitted_image.copy()

            for idx, (l1, l2, disc) in enumerate(zip(landmarks1, landmarks2, discrepancies)):
                if disc >= threshold:
                    # Rojo para alta discrepancia
                    cv2.circle(master_highlighted, l1, 5, (0, 0, 255), -1)
                    cv2.circle(submitted_highlighted, l2, 5, (0, 0, 255), -1)
                else:
                    # Verde para baja discrepancia
                    cv2.circle(master_highlighted, l1, 3, (0, 255, 0), -1)
                    cv2.circle(submitted_highlighted, l2, 3, (0, 255, 0), -1)

            self.display_image(master_highlighted, self.master_image_label)
            self.display_image(submitted_highlighted, self.submitted_image_label)

        except Exception as e:
            messagebox.showerror("Error", f"Error al resaltar discrepancias: {str(e)}")

    def show_mesh(self):
        """Muestra las imágenes con mesh facial."""
        try:
            if self.master_image is not None:
                landmarks1 = detect_landmarks(self.master_image)
                master_with_mesh = draw_face_mesh(self.master_image.copy(), landmarks1)
                self.display_image(master_with_mesh, self.master_image_label)

            if self.submitted_image is not None:
                landmarks2 = detect_landmarks(self.submitted_image)
                submitted_with_mesh = draw_face_mesh(self.submitted_image.copy(), landmarks2)
                self.display_image(submitted_with_mesh, self.submitted_image_label)
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar mesh: {str(e)}")

    def show_mesh_only(self):
        """Muestra solo el mesh sobre fondo negro."""
        try:
            if self.master_image is not None:
                landmarks1 = self._get_master_landmarks()
                black_bg1 = np.zeros_like(self.master_image)
                master_mesh_only = draw_face_mesh(black_bg1, landmarks1)
                self.display_image(master_mesh_only, self.master_image_label)

            if self.submitted_image is not None:
                landmarks2 = self._get_submitted_landmarks()
                black_bg2 = np.zeros_like(self.submitted_image)
                submitted_mesh_only = draw_face_mesh(black_bg2, landmarks2)
                self.display_image(submitted_mesh_only, self.submitted_image_label)
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar mesh: {str(e)}")

    def update_review_date(self):
        """Actualiza la fecha de revisión."""
        self.review_date_label.config(text=f"Review Date: {datetime.now().strftime('%Y-%m-%d')}")

def run_gui():
    root = tk.Tk()
    app = BiometricAppGUI(root)
    root.mainloop()
