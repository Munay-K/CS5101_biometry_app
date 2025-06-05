import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh

def detect_landmarks(image):
    """Detecta landmarks faciales en una imagen usando MediaPipe."""
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise ValueError("No se detectaron rostros en la imagen.")

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
