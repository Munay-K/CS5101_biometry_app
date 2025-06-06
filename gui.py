import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from utils.landmarks import detect_landmarks, extract_geometric_features
from utils.verification import calculate_biometric_vector, compare_vectors
from utils.visualization import visualize_landmarks, draw_face_mesh

class BiometricAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Verificación Biométrica Facial")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Variables
        self.master_image = None
        self.submitted_image = None
        self.similarity = None

        # Layout
        self.create_layout()

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

        # Resultados
        self.result_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        self.result_frame.pack(side=tk.TOP, fill=tk.X)

        self.metrics_label = tk.Label(self.result_frame, text="Metrics: --", font=("Arial", 14), bg="#f0f0f0")
        self.metrics_label.pack(side=tk.LEFT, padx=20)

        # Etiqueta para mostrar comparación de características
        self.features_label = tk.Label(self.image_frame, text="Comparación de Características:\n", font=("Arial", 10), bg="#ffffff", justify="left", anchor="nw", relief="solid", padx=10, pady=10)
        self.features_label.pack(side=tk.LEFT, padx=20, fill=tk.Y)

        self.landmark_info_label = tk.Label(self.result_frame, text="Landmark Info: --", font=("Arial", 14, "bold"), bg="#ffffff", fg="#333333", relief="solid", padx=10, pady=5)
        self.landmark_info_label.pack(side=tk.LEFT, padx=20)

        self.timestamp_label = tk.Label(self.result_frame, text="Timestamp: --", font=("Arial", 12), bg="#f0f0f0")
        self.timestamp_label.pack(side=tk.LEFT, padx=20)

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
        self.visualization_menu.configure(menu=menu)

        self.visualization_menu.pack(side=tk.LEFT, padx=20)

        self.compare_button = tk.Button(self.button_frame, text="Comparar Biometrías", bg="#FF5722", fg="white",
                                        font=("Arial", 12), command=self.compare_images)
        self.compare_button.pack(side=tk.LEFT, padx=20)



        # Zona de información adicional
        self.info_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.reviewer_label = tk.Label(self.info_frame, text="Reviewed by: System", font=("Arial", 12), bg="#f0f0f0")
        self.reviewer_label.pack(side=tk.LEFT, padx=20)

        self.review_date_label = tk.Label(self.info_frame, text="Review Date: --", font=("Arial", 12), bg="#f0f0f0")
        self.review_date_label.pack(side=tk.LEFT, padx=20)



        # Botones para selección manual de imágenes
        self.select_master_button = tk.Button(self.button_frame, text="Seleccionar Subject 1", bg="#673AB7", fg="white",
                                              font=("Arial", 12), command=self.select_master_image)
        self.select_master_button.pack(side=tk.LEFT, padx=20)

        self.select_submitted_button = tk.Button(self.button_frame, text="Seleccionar Subject 2", bg="#3F51B5", fg="white",
                                                 font=("Arial", 12), command=self.select_submitted_image)
        self.select_submitted_button.pack(side=tk.LEFT, padx=20)
        self.root.after(100, self.load_master_image)
        self.root.after(200, self.load_submitted_image)

    def load_master_image(self):
        # Cargar automáticamente la imagen de referencia desde la carpeta assets
        file_path = "assets/img1.png"
        if not file_path:
            messagebox.showerror("Error", "No se seleccionó ninguna imagen.")
            return
        self.master_image = cv2.imread(file_path)
        if self.master_image is None:
            messagebox.showerror("Error", f"No se pudo cargar la imagen de referencia desde {file_path}")
            return
        self.display_image(self.master_image, self.master_image_label)
        self.update_review_date()

    def load_submitted_image(self):
        # Cargar automáticamente la imagen a verificar desde la carpeta assets
        file_path = "assets/img2.png"
        if not file_path:
            messagebox.showerror("Error", "No se seleccionó ninguna imagen.")
            return
        self.submitted_image = cv2.imread(file_path)
        if self.submitted_image is None:
            messagebox.showerror("Error", f"No se pudo cargar la imagen a verificar desde {file_path}")
            return
        self.display_image(self.submitted_image, self.submitted_image_label)
        self.compare_images()

    def select_master_image(self):
        """Permite seleccionar manualmente la imagen de Subject 1."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.master_image = cv2.imread(file_path)
            if self.master_image is None:
                messagebox.showerror("Error", f"No se pudo cargar la imagen desde {file_path}")
                return
            self.display_image(self.master_image, self.master_image_label)
            self.compare_images()

    def display_image(self, image, label):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        # Escalar proporcionalmente para que encaje en el área de 400x300
        max_width, max_height = label.winfo_width(), label.winfo_height()
        original_width, original_height = image_pil.size
        scale = min(max_width / original_width, max_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        image_pil = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convertir a formato compatible con Tkinter
        image_tk = ImageTk.PhotoImage(image_pil)
        label.configure(image=image_tk)
        label.image = image_tk
        label.image = image_tk

    def highlight_landmark(self, image, landmarks, selected_landmark, label):
        """Highlight the selected landmark on the image."""
        highlighted_image = image.copy()
        for (x, y) in landmarks:
            if (x, y) == selected_landmark:
                color = (0, 0, 255)  # Red for the selected landmark
            else:
                color = (0, 255, 0)  # Green for other landmarks
            cv2.circle(highlighted_image, (int(x), int(y)), 5, color, -1)
        self.display_image(highlighted_image, label)

    def select_submitted_image(self):
        """Permite seleccionar manualmente la imagen de Subject 2."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.submitted_image = cv2.imread(file_path)
            if self.submitted_image is None:
                messagebox.showerror("Error", f"No se pudo cargar la imagen desde {file_path}")
                return
            self.display_image(self.submitted_image, self.submitted_image_label)
            self.compare_images()

    def compare_images(self):
        """Compara las imágenes cargadas y actualiza la interfaz con métricas y características."""
        # Validar que existan imágenes cargadas
        if self.master_image is None:
            messagebox.showerror("Error", "La imagen de referencia no está cargada.")
            return
        if self.submitted_image is None:
            messagebox.showerror("Error", "La imagen a verificar no está cargada.")
            return

        # Detectar landmarks
        try:
            landmarks1 = detect_landmarks(self.master_image)
            landmarks2 = detect_landmarks(self.submitted_image)
        except Exception as e:
            messagebox.showerror("Error", f"Error al detectar landmarks: {str(e)}")
            self.features_label.config(text="Comparación de Características:\nNo se detectaron landmarks en una o ambas imágenes.")
            return

        # Calcular vectores biométricos y comparar
        vector1 = calculate_biometric_vector(landmarks1)
        vector2 = calculate_biometric_vector(landmarks2)
        similarity, is_same_person = compare_vectors(vector1, vector2)
        self.similarity = similarity
        conclusion = "Misma persona" if is_same_person else "Persona diferente"
        self.metrics_label.config(text=f"Similarity: {similarity:.4f} ({conclusion})")

        # Inicializar y extraer características geométricas
        features1 = np.zeros(10)
        features2 = np.zeros(10)
        try:
            features1 = extract_geometric_features(landmarks1)
            features2 = extract_geometric_features(landmarks2)
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular características geométricas: {str(e)}")

        # Mostrar comparación de características y marca de tiempo
        self.features_label.config(text=self.generate_feature_comparison_text(features1, features2))
        self.timestamp_label.config(text=f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def generate_feature_comparison_text(self, features1, features2):
        """Genera el texto para mostrar la comparación de características geométricas."""
        labels = [
            "Distancia entre ojos", "Ancho de la boca", "Altura de la boca",
            "Altura de la nariz", "Razón nariz-boca", "Razón ojos-nariz",
            "Ángulo ojos-nariz-boca", "EAR ojo izquierdo", "EAR ojo derecho", "Simetría"
        ]
        differences = np.abs(features1 - features2)
        comparison_text = "Comparación de Características:\n"
        for label, f1, f2, diff in zip(labels, features1, features2, differences):
            comparison_text += f"{label}: A = {f1:.4f}, B = {f2:.4f}, Dif = {diff:.4f}\n"
        
        # Calcular métricas generales
        euclidean_distance = np.linalg.norm(features1 - features2)
        cosine_similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        comparison_text += f"\nDistancia Euclidiana total: {euclidean_distance:.4f}\n"
        comparison_text += f"Similitud por coseno: {cosine_similarity:.4f}\n"
        
        return comparison_text



    def draw_landmarks(self, image, landmarks, selected_landmark=None):
        """Draw landmarks on the image, highlighting the selected one if provided."""
        annotated_image = image.copy()
        for idx, (x, y) in enumerate(landmarks):
            color = (0, 0, 255) if selected_landmark == idx else (0, 255, 0)
            cv2.circle(annotated_image, (x, y), 5, color, -1)
        return annotated_image

    def highlight_discrepancies(self):
        """Highlight landmarks with the greatest discrepancies between the two images."""
        if self.master_image is None or self.submitted_image is None:
            messagebox.showerror("Error", "Ambas imágenes deben estar cargadas para resaltar discrepancias.")
            return

        try:
            # Detect landmarks
            landmarks1 = detect_landmarks(self.master_image)
            landmarks2 = detect_landmarks(self.submitted_image)

            if not landmarks1 or not landmarks2:
                self.features_label.config(text="Comparación de Características:\nNo se detectaron landmarks en una o ambas imágenes.")
                return

            # Calculate discrepancies
            discrepancies = [np.linalg.norm(np.array(l1) - np.array(l2)) for l1, l2 in zip(landmarks1, landmarks2)]
            max_discrepancy_indices = sorted(range(len(discrepancies)), key=lambda i: discrepancies[i], reverse=True)[:5]

            # Highlight discrepancies
            if self.master_image is None or self.submitted_image is None:
                messagebox.showerror("Error", "Las imágenes no están cargadas correctamente.")
                return
            if self.master_image is None or self.submitted_image is None:
                messagebox.showerror("Error", "Las imágenes no están cargadas correctamente.")
                return
            if self.master_image is not None and self.submitted_image is not None:
                master_with_discrepancies = self.master_image.copy()
                submitted_with_discrepancies = self.submitted_image.copy()
            else:
                messagebox.showerror("Error", "Las imágenes no están cargadas correctamente.")
                return
            for idx in max_discrepancy_indices:
                x1, y1 = landmarks1[idx]
                x2, y2 = landmarks2[idx]
                cv2.circle(master_with_discrepancies, (x1, y1), 5, (0, 0, 255), -1)  # Red for discrepancies
                cv2.circle(submitted_with_discrepancies, (x2, y2), 5, (0, 0, 255), -1)  # Red for discrepancies
                cv2.line(submitted_with_discrepancies, (x2, y2), (x1, y1), (255, 0, 0), 1)  # Blue line connecting

            # Display updated images
            self.display_image(master_with_discrepancies, self.master_image_label)
            self.display_image(submitted_with_discrepancies, self.submitted_image_label)

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al resaltar discrepancias: {str(e)}")

    def on_landmark_click(self, event, image_type):
        """Handle clicks on a landmark and highlight it."""
        if image_type == "master":
            landmarks = detect_landmarks(self.master_image)
            other_landmarks = detect_landmarks(self.submitted_image)
            label = self.master_image_label
            other_label = self.submitted_image_label
            image = self.master_image
            other_image = self.submitted_image
        else:
            landmarks = detect_landmarks(self.submitted_image)
            other_landmarks = detect_landmarks(self.master_image)
            label = self.submitted_image_label
            other_label = self.master_image_label
            image = self.submitted_image
            other_image = self.master_image

        if not landmarks or not other_landmarks:
            self.landmark_info_label.config(text="Landmark Info: No landmarks detected")
            return

        clicked_x, clicked_y = event.x, event.y
        if image is None:
            messagebox.showerror("Error", "La imagen no está cargada correctamente.")
            return
        if image is None:
            messagebox.showerror("Error", "La imagen no está cargada correctamente.")
            return
        if image is not None:
            scale_x = label.winfo_width() / image.shape[1]
            scale_y = label.winfo_height() / image.shape[0]
        else:
            messagebox.showerror("Error", "La imagen no está cargada correctamente.")
            return
        closest_idx = min(range(len(landmarks)), key=lambda i: ((landmarks[i][0] * scale_x - clicked_x) ** 2 + (landmarks[i][1] * scale_y - clicked_y) ** 2))

        # Update the images with the selected landmark highlighted
        updated_image = self.draw_landmarks(image, landmarks, selected_landmark=closest_idx)
        updated_other_image = self.draw_landmarks(other_image, other_landmarks, selected_landmark=closest_idx)
        self.display_image(updated_image, label)
        self.display_image(updated_other_image, other_label)

        # Update the landmark info
        self.landmark_info_label.config(
            text=f"Landmark Info: X={landmarks[closest_idx][0]}, Y={landmarks[closest_idx][1]}"
        )

        try:
            # Detectar landmarks y calcular vectores biométricos
            landmarks1 = detect_landmarks(self.master_image)
            landmarks2 = detect_landmarks(self.submitted_image)

            # Ajustar landmarks al escalado de las imágenes
            scale_master_x = 400 / self.master_image.shape[1] if self.master_image is not None else 1
            scale_master_y = 300 / self.master_image.shape[0] if self.master_image is not None else 1
            scale_submitted_x = 400 / self.submitted_image.shape[1] if self.submitted_image is not None else 1
            scale_submitted_y = 300 / self.submitted_image.shape[0] if self.submitted_image is not None else 1
            landmarks1 = [(int(x * scale_master_x), int(y * scale_master_y)) for x, y in landmarks1]
            landmarks2 = [(int(x * scale_submitted_x), int(y * scale_submitted_y)) for x, y in landmarks2]
            vector1 = calculate_biometric_vector(landmarks1)
            vector2 = calculate_biometric_vector(landmarks2)

            # Comparar vectores
            similarity, _ = compare_vectors(vector1, vector2)
            self.similarity = similarity * 100
            self.result_label.config(text=f"Photo Match: {self.similarity:.2f}%")
            self.timestamp_label.config(text=f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Visualizar landmarks
            master_landmarked = visualize_landmarks(self.master_image.copy(), landmarks1)
            submitted_landmarked = visualize_landmarks(self.submitted_image.copy(), landmarks2)
            self.display_image(master_landmarked, self.master_image_label)
            self.display_image(submitted_landmarked, self.submitted_image_label)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during comparison: {str(e)}")

    def show_original_images(self):
        """Muestra las imágenes originales."""
        self.display_image(self.master_image, self.master_image_label)
        self.display_image(self.submitted_image, self.submitted_image_label)

    def show_landmarks(self):
        """Muestra las imágenes con landmarks."""
        landmarks1 = detect_landmarks(self.master_image)
        landmarks2 = detect_landmarks(self.submitted_image)
        master_with_landmarks = visualize_landmarks(self.master_image.copy(), landmarks1)
        submitted_with_landmarks = visualize_landmarks(self.submitted_image.copy(), landmarks2)
        self.display_image(master_with_landmarks, self.master_image_label)
        self.display_image(submitted_with_landmarks, self.submitted_image_label)

    def show_mesh(self):
        """Muestra las imágenes con mesh."""
        landmarks1 = detect_landmarks(self.master_image)
        landmarks2 = detect_landmarks(self.submitted_image)
        if landmarks1:
            master_with_mesh = draw_face_mesh(self.master_image.copy(), landmarks1)
            self.display_image(master_with_mesh, self.master_image_label)
        else:
            print("No se detectaron landmarks en la imagen de referencia.")
        if landmarks2:
            submitted_with_mesh = draw_face_mesh(self.submitted_image.copy(), landmarks2)
            self.display_image(submitted_with_mesh, self.submitted_image_label)
        else:
            print("No se detectaron landmarks en la imagen a verificar.")

    def show_mesh_only(self):
        """Muestra solo el mesh sobre un fondo negro."""
        landmarks1 = detect_landmarks(self.master_image)
        landmarks2 = detect_landmarks(self.submitted_image)
        if self.master_image is not None and self.submitted_image is not None:
            master_mesh_only = np.zeros((self.master_image.shape[0], self.master_image.shape[1], 3), dtype=np.uint8)
            submitted_mesh_only = np.zeros((self.submitted_image.shape[0], self.submitted_image.shape[1], 3), dtype=np.uint8)
        else:
            messagebox.showerror("Error", "Las imágenes no están cargadas correctamente.")
            return
        if landmarks1:
            master_with_mesh = draw_face_mesh(master_mesh_only, landmarks1)
            self.display_image(master_with_mesh, self.master_image_label)
        else:
            print("No se detectaron landmarks en la imagen de referencia.")
        if landmarks2:
            submitted_with_mesh = draw_face_mesh(submitted_mesh_only, landmarks2)
            self.display_image(submitted_with_mesh, self.submitted_image_label)
        else:
            print("No se detectaron landmarks en la imagen a verificar.")

    def update_review_date(self):
        self.review_date_label.config(text=f"Review Date: {datetime.now().strftime('%Y-%m-%d')}")

def run_gui():
    root = tk.Tk()
    app = BiometricAppGUI(root)
    root.mainloop()
