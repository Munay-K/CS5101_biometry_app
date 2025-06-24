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
from smile_eye_detector import MultiFaceSmileEyeDetector
from face_swap import FaceSwapper

class BiometricAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Verificación Biométrica Facial - Versión Mejorada")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")
        self.smile_eye_detector = MultiFaceSmileEyeDetector()
        self.face_swapper = FaceSwapper()

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
        menu.add_separator()
        # Face swap options
        menu.add_command(label="Visualizar Triángulos", command=self.visualize_triangles)
        menu.add_command(label="Face Swap Básico", command=self.basic_face_swap)
        menu.add_command(label="Face Swap Mejorado", command=self.improved_face_swap)
        menu.add_command(label="Face Swap Interactivo", command=self.interactive_face_swap)
        menu.add_separator()
        # Expression analysis
        menu.add_command(label="Analizar Expresión Facial", command=self.analyze_expression)
        menu.add_command(label="Comparar Expresiones", command=self.compare_expressions)
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
            self.face_swapper.clear_cache()
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
            self.face_swapper.clear_cache()
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
            self.face_swapper.clear_cache()
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
            self.face_swapper.clear_cache()
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

    # =============================================================================
    # FACE SWAP METHODS - Now using the FaceSwapper module
    # =============================================================================

    def visualize_triangles(self):
        """Visualiza triángulos en ambas imágenes para debugging."""
        if self.master_image is None or self.submitted_image is None:
            messagebox.showerror("Error", "Ambas imágenes deben estar cargadas.")
            return

        try:
            img1_triangles, img2_triangles = self.face_swapper.visualize_triangles(
                self.master_image, self.submitted_image, num_triangles=20
            )
            
            self.display_image(img1_triangles, self.master_image_label)
            self.display_image(img2_triangles, self.submitted_image_label)
            
            messagebox.showinfo("Visualización", "Triángulos visualizados. Los primeros 20 triángulos están marcados con colores.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al visualizar triángulos: {str(e)}")

    def basic_face_swap(self):
        """Realiza face swap básico usando Delaunay triangulation."""
        if self.master_image is None or self.submitted_image is None:
            messagebox.showerror("Error", "Ambas imágenes deben estar cargadas.")
            return

        try:
            result1, result2 = self.face_swapper.basic_face_swap(
                self.master_image, self.submitted_image
            )
            
            # Actualizar las imágenes mostradas
            self.master_image = result1
            self.submitted_image = result2
            
            # Mostrar resultados
            self.display_image(self.master_image, self.master_image_label)
            self.display_image(self.submitted_image, self.submitted_image_label)
            
            # Limpiar caches
            self.master_landmarks = None
            self.submitted_landmarks = None
            
            messagebox.showinfo("Face Swap Básico", "Face swap básico completado!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en face swap básico: {str(e)}")

    def improved_face_swap(self):
        """Realiza face swap mejorado con manejo de expresiones."""
        if self.master_image is None or self.submitted_image is None:
            messagebox.showerror("Error", "Ambas imágenes deben estar cargadas.")
            return

        try:
            result1, result2 = self.face_swapper.improved_face_swap(
                self.master_image, self.submitted_image
            )
            
            # Actualizar las imágenes mostradas
            self.master_image = result1
            self.submitted_image = result2
            
            # Mostrar resultados
            self.display_image(self.master_image, self.master_image_label)
            self.display_image(self.submitted_image, self.submitted_image_label)
            
            # Limpiar caches
            self.master_landmarks = None
            self.submitted_landmarks = None
            
            messagebox.showinfo("Face Swap Mejorado", "Face swap mejorado completado con manejo de expresiones!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en face swap mejorado: {str(e)}")

    def interactive_face_swap(self):
        """Realiza face swap interactivo con selección de triángulos."""
        if self.master_image is None or self.submitted_image is None:
            messagebox.showerror("Error", "Ambas imágenes deben estar cargadas.")
            return

        try:
            result1, result2 = self.face_swapper.interactive_triangle_swap(
                self.master_image, self.submitted_image, parent_window=self.root
            )
            
            if result1 is not None and result2 is not None:
                # Actualizar las imágenes mostradas
                self.master_image = result1
                self.submitted_image = result2
                
                # Mostrar resultados
                self.display_image(self.master_image, self.master_image_label)
                self.display_image(self.submitted_image, self.submitted_image_label)
                
                # Limpiar caches
                self.master_landmarks = None
                self.submitted_landmarks = None
                
                messagebox.showinfo("Face Swap Interactivo", "Face swap interactivo completado!")
            else:
                messagebox.showinfo("Cancelado", "Face swap interactivo fue cancelado.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en face swap interactivo: {str(e)}")

    # =============================================================================
    # EXPRESSION ANALYSIS METHODS
    # =============================================================================

    def analyze_expression(self):
        """Analiza la expresión facial en ambas imágenes."""
        if self.master_image is None and self.submitted_image is None:
            messagebox.showwarning("Advertencia", "Debe cargar al menos una imagen.")
            return
        
        try:
            # Analizar Subject 1
            if self.master_image is not None:
                landmarks1 = self._get_master_landmarks()
                analysis1 = self.smile_eye_detector.analyze_expression(landmarks1)
                
                # Visualizar usando analyze_multiple_faces para una sola cara
                multi_result1 = self.smile_eye_detector.analyze_multiple_faces(self.master_image)
                if multi_result1['num_faces'] > 0:
                    result_img1 = self.smile_eye_detector.visualize_multiple_faces(
                        self.master_image, multi_result1
                    )
                    self.display_image(result_img1, self.master_image_label)
                
                # Mostrar resultados en el panel de texto
                self.features_text.delete(1.0, tk.END)
                self.features_text.insert(tk.END, "=== ANÁLISIS DE EXPRESIÓN - SUBJECT 1 ===\n\n", "title")
                self._display_expression_results(analysis1, "Subject 1")
            
            # Analizar Subject 2
            if self.submitted_image is not None:
                landmarks2 = self._get_submitted_landmarks()
                analysis2 = self.smile_eye_detector.analyze_expression(landmarks2)
                
                # Visualizar usando analyze_multiple_faces para una sola cara
                multi_result2 = self.smile_eye_detector.analyze_multiple_faces(self.submitted_image)
                if multi_result2['num_faces'] > 0:
                    result_img2 = self.smile_eye_detector.visualize_multiple_faces(
                        self.submitted_image, multi_result2
                    )
                    self.display_image(result_img2, self.submitted_image_label)
                
                # Agregar resultados al panel
                if self.master_image is not None:
                    self.features_text.insert(tk.END, "\n\n=== ANÁLISIS DE EXPRESIÓN - SUBJECT 2 ===\n\n", "title")
                self._display_expression_results(analysis2, "Subject 2")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al analizar expresión: {str(e)}")
    
    def _display_expression_results(self, analysis, subject_name):
        """Muestra los resultados del análisis de expresión en el panel de texto."""
        # Resultado general
        self.features_text.insert(tk.END, f"Expresión detectada: {analysis['expression']}\n", "subtitle")
        self.features_text.insert(tk.END, f"Puntuación de calidad: {analysis['quality_score']:.2f}\n\n")
        
        # Análisis de sonrisa
        self.features_text.insert(tk.END, "ANÁLISIS DE SONRISA:\n", "subtitle")
        smile = analysis['smile']
        status = "✓ Sonriendo" if smile['is_smiling'] else "✗ No sonriendo"
        color_tag = "success" if smile['is_smiling'] else "error"
        self.features_text.insert(tk.END, f"  Estado: {status}\n", color_tag)
        self.features_text.insert(tk.END, f"  Confianza: {smile['confidence']:.2%}\n")
        self.features_text.insert(tk.END, f"  Métricas:\n")
        self.features_text.insert(tk.END, f"    - Ratio boca: {smile['metrics']['mouth_ratio']:.2f}\n")
        self.features_text.insert(tk.END, f"    - Curvatura: {smile['metrics']['mouth_curve_angle']:.1f}°\n")
        self.features_text.insert(tk.END, f"    - Separación labios: {smile['metrics']['lip_separation']:.3f}\n")
        
        # Análisis de ojos
        self.features_text.insert(tk.END, "\nANÁLISIS DE OJOS:\n", "subtitle")
        eyes = analysis['eyes']
        
        # Ojo izquierdo
        left_status = "✓ Abierto" if eyes['left_eye_open'] else "✗ Cerrado"
        left_color = "success" if eyes['left_eye_open'] else "error"
        self.features_text.insert(tk.END, f"  Ojo izquierdo: {left_status} (EAR: {eyes['metrics']['left_ear']:.3f})\n", left_color)
        
        # Ojo derecho
        right_status = "✓ Abierto" if eyes['right_eye_open'] else "✗ Cerrado"
        right_color = "success" if eyes['right_eye_open'] else "error"
        self.features_text.insert(tk.END, f"  Ojo derecho: {right_status} (EAR: {eyes['metrics']['right_ear']:.3f})\n", right_color)
        
        self.features_text.insert(tk.END, f"  Confianza general: {eyes['confidence']:.2%}\n")
        
        # Recomendación
        self.features_text.insert(tk.END, "\nRECOMENDACIÓN:\n", "subtitle")
        if analysis['is_ideal']:
            self.features_text.insert(tk.END, 
                "✓ Expresión ideal detectada (sonrisa con ojos abiertos)\n", "success")
        else:
            self.features_text.insert(tk.END, 
                "⚠ Se recomienda capturar con sonrisa y ojos completamente abiertos\n", "warning")
    
    def compare_expressions(self):
        """Compara las expresiones entre ambas imágenes."""
        if self.master_image is None or self.submitted_image is None:
            messagebox.showwarning("Advertencia", "Ambas imágenes deben estar cargadas.")
            return
        
        try:
            # Analizar ambas imágenes
            landmarks1 = self._get_master_landmarks()
            landmarks2 = self._get_submitted_landmarks()
            
            analysis1 = self.smile_eye_detector.analyze_expression(landmarks1)
            analysis2 = self.smile_eye_detector.analyze_expression(landmarks2)
            
            # Crear ventana de comparación
            comp_window = tk.Toplevel(self.root)
            comp_window.title("Comparación de Expresiones Faciales")
            comp_window.geometry("800x600")
            
            # Frame principal
            main_frame = tk.Frame(comp_window, bg="#f0f0f0")
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Título
            title_label = tk.Label(main_frame, text="COMPARACIÓN DE EXPRESIONES", 
                                 font=("Arial", 16, "bold"), bg="#f0f0f0")
            title_label.pack(pady=10)
            
            # Frame para las imágenes
            images_frame = tk.Frame(main_frame, bg="#f0f0f0")
            images_frame.pack(fill=tk.BOTH, expand=True)
            
            # Visualizar ambas imágenes con detecciones
            multi_result1 = self.smile_eye_detector.analyze_multiple_faces(self.master_image)
            multi_result2 = self.smile_eye_detector.analyze_multiple_faces(self.submitted_image)
            
            img1_viz = self.smile_eye_detector.visualize_multiple_faces(self.master_image, multi_result1)
            img2_viz = self.smile_eye_detector.visualize_multiple_faces(self.submitted_image, multi_result2)
            
            # Redimensionar para mostrar
            scale = 0.5
            h1, w1 = img1_viz.shape[:2]
            h2, w2 = img2_viz.shape[:2]
            
            img1_small = cv2.resize(img1_viz, (int(w1*scale), int(h1*scale)))
            img2_small = cv2.resize(img2_viz, (int(w2*scale), int(h2*scale)))
            
            # Convertir a formato Tkinter
            img1_rgb = cv2.cvtColor(img1_small, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2_small, cv2.COLOR_BGR2RGB)
            
            img1_pil = Image.fromarray(img1_rgb)
            img2_pil = Image.fromarray(img2_rgb)
            
            img1_tk = ImageTk.PhotoImage(img1_pil)
            img2_tk = ImageTk.PhotoImage(img2_pil)
            
            # Labels para las imágenes
            img1_label = tk.Label(images_frame, image=img1_tk)
            img1_label.image = img1_tk
            img1_label.pack(side=tk.LEFT, padx=10)
            
            img2_label = tk.Label(images_frame, image=img2_tk)
            img2_label.image = img2_tk
            img2_label.pack(side=tk.LEFT, padx=10)
            
            # Frame para resultados
            results_frame = tk.Frame(main_frame, bg="#ffffff", relief=tk.RIDGE, bd=2)
            results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            # Texto de comparación
            comp_text = tk.Text(results_frame, height=15, width=80, font=("Courier", 10))
            comp_text.pack(padx=10, pady=10)
            
            # Agregar resultados de comparación
            comp_text.insert(tk.END, "=== COMPARACIÓN DE EXPRESIONES ===\n\n", "title")
            
            # Comparar expresiones
            comp_text.insert(tk.END, f"Subject 1: {analysis1['expression']}\n")
            comp_text.insert(tk.END, f"Subject 2: {analysis2['expression']}\n\n")
            
            # Comparar sonrisas
            comp_text.insert(tk.END, "COMPARACIÓN DE SONRISAS:\n", "subtitle")
            smile_diff = abs(analysis1['smile']['confidence'] - analysis2['smile']['confidence'])
            comp_text.insert(tk.END, f"  Subject 1: {'Sí' if analysis1['smile']['is_smiling'] else 'No'} ({analysis1['smile']['confidence']:.2%})\n")
            comp_text.insert(tk.END, f"  Subject 2: {'Sí' if analysis2['smile']['is_smiling'] else 'No'} ({analysis2['smile']['confidence']:.2%})\n")
            comp_text.insert(tk.END, f"  Diferencia: {smile_diff:.2%}\n\n")
            
            # Comparar ojos
            comp_text.insert(tk.END, "COMPARACIÓN DE OJOS:\n", "subtitle")
            eyes_diff = abs(analysis1['eyes']['confidence'] - analysis2['eyes']['confidence'])
            comp_text.insert(tk.END, f"  Subject 1: {'Abiertos' if analysis1['eyes']['both_eyes_open'] else 'Cerrados'} ({analysis1['eyes']['confidence']:.2%})\n")
            comp_text.insert(tk.END, f"  Subject 2: {'Abiertos' if analysis2['eyes']['both_eyes_open'] else 'Cerrados'} ({analysis2['eyes']['confidence']:.2%})\n")
            comp_text.insert(tk.END, f"  Diferencia: {eyes_diff:.2%}\n\n")
            
            # Conclusión
            comp_text.insert(tk.END, "CONCLUSIÓN:\n", "subtitle")
            if analysis1['is_ideal'] and analysis2['is_ideal']:
                comp_text.insert(tk.END, "✓ Ambas imágenes tienen expresiones ideales\n", "success")
            elif analysis1['is_ideal'] or analysis2['is_ideal']:
                comp_text.insert(tk.END, "⚠ Solo una imagen tiene expresión ideal\n", "warning")
            else:
                comp_text.insert(tk.END, "✗ Ninguna imagen tiene expresión ideal\n", "error")
            
            # Configurar tags
            comp_text.tag_config("title", font=("Arial", 12, "bold"), foreground="#1976D2")
            comp_text.tag_config("subtitle", font=("Arial", 10, "bold"), foreground="#424242")
            comp_text.tag_config("success", foreground="#2E7D32", font=("Arial", 10, "bold"))
            comp_text.tag_config("warning", foreground="#F57C00", font=("Arial", 10, "bold"))
            comp_text.tag_config("error", foreground="#C62828", font=("Arial", 10, "bold"))
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al comparar expresiones: {str(e)}")

    # =============================================================================
    # BIOMETRIC ANALYSIS METHODS
    # =============================================================================

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