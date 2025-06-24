"""
Face Swap Module - Dedicated module for all face swapping functionality
Contains various face swap algorithms and utilities for better organization and debugging.
"""

import cv2
import numpy as np
from collections import defaultdict
from utils.landmarks import detect_landmarks
from utils.visualization import FACE_CONNECTIONS
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


class FaceSwapper:
    """
    Main class for face swapping operations with various algorithms and utilities.
    """
    
    def __init__(self):
        self.triangles_cache = None
        
    def clear_cache(self):
        """Clear cached triangulation data."""
        self.triangles_cache = None
        
    def get_consistent_triangulation(self):
        """Get consistent triangulation based on MediaPipe face mesh connections."""
        if self.triangles_cache is not None:
            return self.triangles_cache
        
        # Use MediaPipe's face mesh connections to create triangles
        adjacency = defaultdict(set)
        
        # Build adjacency from face connections
        for v1, v2 in FACE_CONNECTIONS:
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
        
        # Find triangles in the mesh
        triangles = set()
        for v in adjacency:
            neighbors = list(adjacency[v])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    v2, v3 = neighbors[i], neighbors[j]
                    if v3 in adjacency[v2]:
                        # Found a triangle - sort to avoid duplicates
                        triangle = tuple(sorted([v, v2, v3]))
                        triangles.add(triangle)
        
        self.triangles_cache = list(triangles)
        return self.triangles_cache

    def basic_face_swap(self, img1, img2):
        """
        Basic face swap using Delaunay triangulation (original experiment_3).
        
        Args:
            img1: First image (BGR format)
            img2: Second image (BGR format)
            
        Returns:
            tuple: (swapped_img1, swapped_img2) - Both images with faces swapped
        """
        print("Starting basic face swap...")
        
        # Get landmarks
        lm1 = np.array(detect_landmarks(img1), np.int32)
        lm2 = np.array(detect_landmarks(img2), np.int32)
        
        # Delaunay triangulation on each face separately
        from scipy.spatial import Delaunay
        tri1 = Delaunay(lm1)
        triangles1 = tri1.simplices
        tri2 = Delaunay(lm2)
        triangles2 = tri2.simplices
        
        # Prepare blank warped face images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        face_warp_to_img2 = np.zeros_like(img2)
        face_warp_to_img1 = np.zeros_like(img1)

        # Basic triangle warping
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

        # Warp img1 -> img2
        for tri_idx in triangles1:
            if all(idx < len(lm1) and idx < len(lm2) for idx in tri_idx):
                warp_triangle(img1, face_warp_to_img2, lm1[tri_idx], lm2[tri_idx])
                
        # Warp img2 -> img1
        for tri_idx in triangles2:
            if all(idx < len(lm1) and idx < len(lm2) for idx in tri_idx):
                warp_triangle(img2, face_warp_to_img1, lm2[tri_idx], lm1[tri_idx])

        # Seamless cloning
        hull1 = cv2.convexHull(lm1)
        hull2 = cv2.convexHull(lm2)
        
        mask1 = np.zeros((h1, w1), dtype=np.uint8)
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
        cv2.fillConvexPoly(mask1, hull1, 255)
        cv2.fillConvexPoly(mask2, hull2, 255)
        
        rc1 = cv2.boundingRect(hull1)
        rc2 = cv2.boundingRect(hull2)
        center1 = (rc1[0] + rc1[2]//2, rc1[1] + rc1[3]//2)
        center2 = (rc2[0] + rc2[2]//2, rc2[1] + rc2[3]//2)
        
        try:
            swapped_img2 = cv2.seamlessClone(face_warp_to_img2, img2, mask2, center2, cv2.NORMAL_CLONE)
            swapped_img1 = cv2.seamlessClone(face_warp_to_img1, img1, mask1, center1, cv2.NORMAL_CLONE)
        except Exception as e:
            print(f"Seamless cloning failed: {e}")
            # Fallback to alpha blending
            swapped_img2 = self._alpha_blend(img2, face_warp_to_img2, mask2)
            swapped_img1 = self._alpha_blend(img1, face_warp_to_img1, mask1)
        
        print("Basic face swap completed!")
        return swapped_img1, swapped_img2

    def improved_face_swap(self, img1, img2):
        """
        Improved face swap with expression handling and consistent triangulation.
        
        Args:
            img1: First image (BGR format)
            img2: Second image (BGR format)
            
        Returns:
            tuple: (swapped_img1, swapped_img2) - Both images with faces swapped
        """
        print("Starting improved face swap...")
        
        # Get landmarks
        lm1 = np.array(detect_landmarks(img1), np.int32)
        lm2 = np.array(detect_landmarks(img2), np.int32)
        
        # Use consistent triangulation
        triangles = self.get_consistent_triangulation()
        
        # Prepare blank warped face images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        face_warp_to_img2 = np.zeros_like(img2)
        face_warp_to_img1 = np.zeros_like(img1)

        # Expression-aware landmark adjustment
        lm1_adjusted, lm2_adjusted = self._adjust_for_expression_differences(lm1, lm2)
        
        print("Warping faces with consistent triangulation...")
        
        # Warp img1 -> img2 using adjusted landmarks
        for triangle in triangles:
            if all(idx < len(lm1_adjusted) and idx < len(lm2_adjusted) for idx in triangle):
                t_src = [lm1_adjusted[idx] for idx in triangle]
                t_dst = [lm2_adjusted[idx] for idx in triangle]
                self._warp_triangle_improved(img1, face_warp_to_img2, t_src, t_dst)
        
        # Warp img2 -> img1 using adjusted landmarks  
        for triangle in triangles:
            if all(idx < len(lm1_adjusted) and idx < len(lm2_adjusted) for idx in triangle):
                t_src = [lm2_adjusted[idx] for idx in triangle]
                t_dst = [lm1_adjusted[idx] for idx in triangle]
                self._warp_triangle_improved(img2, face_warp_to_img1, t_src, t_dst)

        # Create better face masks
        face_landmarks_1 = self._get_facial_boundary_landmarks(lm1)
        face_landmarks_2 = self._get_facial_boundary_landmarks(lm2)
        
        hull1 = cv2.convexHull(np.array(face_landmarks_1))
        hull2 = cv2.convexHull(np.array(face_landmarks_2))
        
        mask1 = np.zeros((h1, w1), dtype=np.uint8)
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
        cv2.fillConvexPoly(mask1, hull1, 255)
        cv2.fillConvexPoly(mask2, hull2, 255)
        
        # Apply Gaussian blur to masks for smoother blending
        mask1 = cv2.GaussianBlur(mask1, (5, 5), 0)
        mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)
        
        # Calculate centers for seamless cloning
        rc1 = cv2.boundingRect(hull1)
        rc2 = cv2.boundingRect(hull2)
        center1 = (rc1[0] + rc1[2]//2, rc1[1] + rc1[3]//2)
        center2 = (rc2[0] + rc2[2]//2, rc2[1] + rc2[3]//2)
        
        # Seamless clone with better blending
        try:
            swapped_img2 = cv2.seamlessClone(face_warp_to_img2, img2, mask2, center2, cv2.NORMAL_CLONE)
            swapped_img1 = cv2.seamlessClone(face_warp_to_img1, img1, mask1, center1, cv2.NORMAL_CLONE)
        except Exception as e:
            print(f"Seamless cloning failed, using alpha blending: {e}")
            swapped_img2 = self._alpha_blend(img2, face_warp_to_img2, mask2)
            swapped_img1 = self._alpha_blend(img1, face_warp_to_img1, mask1)
        
        print("Improved face swap completed!")
        return swapped_img1, swapped_img2

    def interactive_triangle_swap(self, img1, img2, parent_window=None):
        """
        Interactive triangle selection and swapping (original experiment_2).
        
        Args:
            img1: First image (BGR format)
            img2: Second image (BGR format)
            parent_window: Parent tkinter window for the dialog
            
        Returns:
            tuple: (modified_img1, modified_img2) or (None, None) if cancelled
        """
        print("Starting interactive triangle swap...")
        
        if parent_window is None:
            # Create a temporary root if none provided
            parent_window = tk.Tk()
            parent_window.withdraw()
        
        exp_window = tk.Toplevel(parent_window)
        exp_window.title("Interactive Triangle Selection")

        # Convert images for Tkinter
        image_rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        pil_image1 = Image.fromarray(image_rgb1)
        photo1 = ImageTk.PhotoImage(pil_image1)

        image_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        pil_image2 = Image.fromarray(image_rgb2)
        photo2 = ImageTk.PhotoImage(pil_image2)

        # Create canvases for both images
        canvas1 = tk.Canvas(exp_window, width=pil_image1.width, height=pil_image1.height)
        canvas1.grid(row=0, column=0, padx=10, pady=10)
        canvas1.create_image(0, 0, anchor=tk.NW, image=photo1)
        canvas1.image = photo1

        canvas2 = tk.Canvas(exp_window, width=pil_image2.width, height=pil_image2.height)
        canvas2.grid(row=0, column=1, padx=10, pady=10)
        canvas2.create_image(0, 0, anchor=tk.NW, image=photo2)
        canvas2.image = photo2

        # Get landmarks and triangles
        landmarks1 = detect_landmarks(img1)
        landmarks2 = detect_landmarks(img2)
        triangles = self.get_consistent_triangulation()

        # Selection state
        selected = {tri: False for tri in triangles}
        triangles_mapping = {}
        result = [None, None]  # Will store the result images

        # Create triangle polygons
        for tri in triangles:
            if all(idx < len(landmarks1) and idx < len(landmarks2) for idx in tri):
                pts1 = [landmarks1[idx] for idx in tri]
                pts2 = [landmarks2[idx] for idx in tri]
                flat_pts1 = [coord for point in pts1 for coord in point]
                flat_pts2 = [coord for point in pts2 for coord in point]
                poly1 = canvas1.create_polygon(flat_pts1, outline="blue", fill="", width=1)
                poly2 = canvas2.create_polygon(flat_pts2, outline="blue", fill="", width=1)
                triangles_mapping[tri] = {"canvas1": poly1, "canvas2": poly2}

        # Event handlers
        def on_enter(event, tri_key):
            if not selected[tri_key]:
                canvas1.itemconfig(triangles_mapping[tri_key]["canvas1"], fill="#d3d3d3")
                canvas2.itemconfig(triangles_mapping[tri_key]["canvas2"], fill="#d3d3d3")

        def on_leave(event, tri_key):
            if not selected[tri_key]:
                canvas1.itemconfig(triangles_mapping[tri_key]["canvas1"], fill="")
                canvas2.itemconfig(triangles_mapping[tri_key]["canvas2"], fill="")

        def on_click(event, tri_key):
            selected[tri_key] = not selected[tri_key]
            new_color = "#00ff00" if selected[tri_key] else ""
            canvas1.itemconfig(triangles_mapping[tri_key]["canvas1"], fill=new_color)
            canvas2.itemconfig(triangles_mapping[tri_key]["canvas2"], fill=new_color)

        # Bind events
        for tri_key, mapping in triangles_mapping.items():
            for canvas, item in [(canvas1, mapping["canvas1"]), (canvas2, mapping["canvas2"])]:
                canvas.tag_bind(item, "<Enter>", lambda e, key=tri_key: on_enter(e, key))
                canvas.tag_bind(item, "<Leave>", lambda e, key=tri_key: on_leave(e, key))
                canvas.tag_bind(item, "<Button-1>", lambda e, key=tri_key: on_click(e, key))

        # Buttons
        btn_frame = tk.Frame(exp_window)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)

        def on_accept():
            # Perform triangle swaps
            updated_img1 = img1.copy()
            updated_img2 = img2.copy()
            
            for tri_key, is_selected in selected.items():
                if is_selected and all(idx < len(landmarks1) and idx < len(landmarks2) for idx in tri_key):
                    self._swap_triangle_pair(updated_img1, updated_img2, landmarks1, landmarks2, tri_key)
            
            result[0] = updated_img1
            result[1] = updated_img2
            exp_window.destroy()

        def on_cancel():
            result[0] = None
            result[1] = None
            exp_window.destroy()

        tk.Button(btn_frame, text="Accept", command=on_accept, bg="#4CAF50", fg="white", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=20)
        tk.Button(btn_frame, text="Cancel", command=on_cancel, bg="#F44336", fg="white", font=("Arial", 12, "bold")).pack(side=tk.RIGHT, padx=5)

        # Wait for window to close
        exp_window.wait_window()
        
        print("Interactive triangle swap completed!")
        return result[0], result[1]

    def visualize_triangles(self, img1, img2, num_triangles=20):
        """
        Visualize triangles on both images for debugging.
        
        Args:
            img1: First image (BGR format)
            img2: Second image (BGR format)
            num_triangles: Number of triangles to draw
            
        Returns:
            tuple: (img1_with_triangles, img2_with_triangles)
        """
        landmarks1 = detect_landmarks(img1)
        landmarks2 = detect_landmarks(img2)
        triangles = self.get_consistent_triangulation()
        
        img1_copy = img1.copy()
        img2_copy = img2.copy()
        
        # Draw triangles with random colors
        for i, triangle in enumerate(triangles[:num_triangles]):
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            
            if all(idx < len(landmarks1) and idx < len(landmarks2) for idx in triangle):
                # Draw on image 1
                pts1 = np.array([landmarks1[idx] for idx in triangle], np.int32)
                cv2.drawContours(img1_copy, [pts1], 0, color, 2)
                
                # Draw on image 2
                pts2 = np.array([landmarks2[idx] for idx in triangle], np.int32)
                cv2.drawContours(img2_copy, [pts2], 0, color, 2)
        
        return img1_copy, img2_copy

    def _warp_triangle_improved(self, img_src, img_dst, t_src, t_dst):
        """Improved triangle warping with better boundary handling."""
        try:
            # Ensure triangles are valid
            if cv2.contourArea(np.float32(t_src)) < 1 or cv2.contourArea(np.float32(t_dst)) < 1:
                return
            
            r1 = cv2.boundingRect(np.float32([t_src]))
            r2 = cv2.boundingRect(np.float32([t_dst]))
            x1, y1, w1, h1 = r1
            x2, y2, w2, h2 = r2
            
            # Check bounds
            if x1 < 0 or y1 < 0 or x1+w1 > img_src.shape[1] or y1+h1 > img_src.shape[0]:
                return
            if x2 < 0 or y2 < 0 or x2+w2 > img_dst.shape[1] or y2+h2 > img_dst.shape[0]:
                return
            
            src_rect = np.array([[pt[0]-x1, pt[1]-y1] for pt in t_src], np.float32)
            dst_rect = np.array([[pt[0]-x2, pt[1]-y2] for pt in t_dst], np.float32)
            
            crop = img_src[y1:y1+h1, x1:x1+w1]
            if crop.size == 0:
                return
                
            M = cv2.getAffineTransform(src_rect, dst_rect)
            warped = cv2.warpAffine(crop, M, (w2, h2), 
                                  flags=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_REFLECT_101)
            
            # Create precise triangle mask
            mask = np.zeros((h2, w2), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst_rect), 255)
            
            # Apply warped triangle to destination
            dst_roi = img_dst[y2:y2+h2, x2:x2+w2]
            dst_roi[mask == 255] = warped[mask == 255]
            
        except Exception as e:
            print(f"Warning: Triangle warp failed: {e}")

    def _swap_triangle_pair(self, img1, img2, landmarks1, landmarks2, triangle_indices):
        """Swap a single triangle between two images."""
        pts1 = np.array([landmarks1[idx] for idx in triangle_indices], np.int32)
        pts2 = np.array([landmarks2[idx] for idx in triangle_indices], np.int32)

        rect1 = cv2.boundingRect(pts1)
        rect2 = cv2.boundingRect(pts2)
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Extract and swap triangles
        crop1 = img1[y1:y1+h1, x1:x1+w1].copy()
        crop2 = img2[y2:y2+h2, x2:x2+w2].copy()
        
        mask1 = np.zeros((h1, w1), dtype=np.uint8)
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
        
        pts1_shifted = pts1 - [x1, y1]
        pts2_shifted = pts2 - [x2, y2]
        
        cv2.fillPoly(mask1, [pts1_shifted], 255)
        cv2.fillPoly(mask2, [pts2_shifted], 255)

        # Transform and apply
        if len(pts1_shifted) >= 3 and len(pts2_shifted) >= 3:
            M1to2 = cv2.getAffineTransform(pts1_shifted[:3].astype(np.float32), pts2_shifted[:3].astype(np.float32))
            M2to1 = cv2.getAffineTransform(pts2_shifted[:3].astype(np.float32), pts1_shifted[:3].astype(np.float32))

            warped1to2 = cv2.warpAffine(crop1, M1to2, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            warped2to1 = cv2.warpAffine(crop2, M2to1, (w1, h1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # Apply warped triangles
            for i in range(h2):
                for j in range(w2):
                    if mask2[i, j] > 0:
                        img2[y2 + i, x2 + j] = warped1to2[i, j]
                        
            for i in range(h1):
                for j in range(w1):
                    if mask1[i, j] > 0:
                        img1[y1 + i, x1 + j] = warped2to1[i, j]

    def _adjust_for_expression_differences(self, lm1, lm2):
        """Adjust landmarks to handle expression differences between faces."""
        lm1_adj = lm1.copy()
        lm2_adj = lm2.copy()
        
        # Eye region adjustment
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Detect if eyes are in different states (open vs closed)
        left_eye_openness_1 = self._calculate_eye_openness(lm1, left_eye_indices)
        left_eye_openness_2 = self._calculate_eye_openness(lm2, left_eye_indices)
        right_eye_openness_1 = self._calculate_eye_openness(lm1, right_eye_indices)
        right_eye_openness_2 = self._calculate_eye_openness(lm2, right_eye_indices)
        
        # If eye states are very different, apply adjustment
        if abs(left_eye_openness_1 - left_eye_openness_2) > 0.3:
            lm1_adj, lm2_adj = self._normalize_eye_landmarks(lm1_adj, lm2_adj, left_eye_indices)
        
        if abs(right_eye_openness_1 - right_eye_openness_2) > 0.3:
            lm1_adj, lm2_adj = self._normalize_eye_landmarks(lm1_adj, lm2_adj, right_eye_indices)
        
        # Mouth region adjustment
        mouth_indices = [0, 17, 18, 200, 199, 175, 61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        
        mouth_openness_1 = self._calculate_mouth_openness(lm1, mouth_indices)
        mouth_openness_2 = self._calculate_mouth_openness(lm2, mouth_indices)
        
        if abs(mouth_openness_1 - mouth_openness_2) > 0.4:
            lm1_adj, lm2_adj = self._normalize_mouth_landmarks(lm1_adj, lm2_adj, mouth_indices)
        
        return lm1_adj, lm2_adj

    def _calculate_eye_openness(self, landmarks, eye_indices):
        """Calculate eye openness ratio."""
        try:
            if len(eye_indices) < 6:
                return 0.5
            
            # Get top and bottom eye points
            top_points = [landmarks[eye_indices[i]] for i in [2, 3]]
            bottom_points = [landmarks[eye_indices[i]] for i in [4, 5]]
            
            # Calculate average vertical distance
            vertical_dist = np.mean([
                np.linalg.norm(np.array(top_points[i]) - np.array(bottom_points[i])) 
                for i in range(min(len(top_points), len(bottom_points)))
            ])
            
            # Calculate horizontal distance for normalization
            horizontal_dist = np.linalg.norm(
                np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[1]])
            )
            
            return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0.5
        except:
            return 0.5

    def _calculate_mouth_openness(self, landmarks, mouth_indices):
        """Calculate mouth openness ratio."""
        try:
            # Top and bottom lip center points
            top_lip = landmarks[13]  # Upper lip center
            bottom_lip = landmarks[14]  # Lower lip center
            
            # Mouth corners
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            
            vertical_dist = np.linalg.norm(np.array(top_lip) - np.array(bottom_lip))
            horizontal_dist = np.linalg.norm(np.array(left_corner) - np.array(right_corner))
            
            return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0.1
        except:
            return 0.1

    def _normalize_eye_landmarks(self, lm1, lm2, eye_indices):
        """Normalize eye landmarks to handle open/closed differences."""
        # Calculate the center and average position
        center1 = np.mean([lm1[idx] for idx in eye_indices[:4]], axis=0)
        center2 = np.mean([lm2[idx] for idx in eye_indices[:4]], axis=0)
        
        # For closed eyes, move landmarks toward the center slightly
        openness1 = self._calculate_eye_openness(lm1, eye_indices)
        openness2 = self._calculate_eye_openness(lm2, eye_indices)
        
        target_openness = (openness1 + openness2) / 2  # Average openness
        
        # Adjust landmarks toward target openness
        for idx in eye_indices:
            if idx < len(lm1) and idx < len(lm2):
                # Move landmarks toward/away from center based on target openness
                direction1 = lm1[idx] - center1
                direction2 = lm2[idx] - center2
                
                # Apply adjustment factor
                adjustment_factor = 0.3  # How much to adjust
                if openness1 < target_openness:
                    lm1[idx] = center1 + direction1 * (1 + adjustment_factor)
                elif openness1 > target_openness:
                    lm1[idx] = center1 + direction1 * (1 - adjustment_factor)
                    
                if openness2 < target_openness:
                    lm2[idx] = center2 + direction2 * (1 + adjustment_factor)
                elif openness2 > target_openness:
                    lm2[idx] = center2 + direction2 * (1 - adjustment_factor)
        
        return lm1, lm2

    def _normalize_mouth_landmarks(self, lm1, lm2, mouth_indices):
        """Normalize mouth landmarks to handle open/closed differences."""
        # Similar to eye normalization but for mouth
        mouth_center_indices = [13, 14, 61, 291]  # Key mouth points
        center1 = np.mean([lm1[idx] for idx in mouth_center_indices], axis=0)
        center2 = np.mean([lm2[idx] for idx in mouth_center_indices], axis=0)
        
        openness1 = self._calculate_mouth_openness(lm1, mouth_indices)
        openness2 = self._calculate_mouth_openness(lm2, mouth_indices)
        
        target_openness = (openness1 + openness2) / 2
        
        # Adjust inner mouth landmarks more than outer ones
        inner_mouth_indices = [13, 14, 12, 15]  # Upper/lower lip centers
        
        for idx in inner_mouth_indices:
            if idx < len(lm1) and idx < len(lm2):
                direction1 = lm1[idx] - center1
                direction2 = lm2[idx] - center2
                
                adjustment_factor = 0.4
                if openness1 < target_openness:
                    lm1[idx] = center1 + direction1 * (1 + adjustment_factor)
                elif openness1 > target_openness:
                    lm1[idx] = center1 + direction1 * (1 - adjustment_factor)
                    
                if openness2 < target_openness:
                    lm2[idx] = center2 + direction2 * (1 + adjustment_factor)
                elif openness2 > target_openness:
                    lm2[idx] = center2 + direction2 * (1 - adjustment_factor)
        
        return lm1, lm2

    def _get_facial_boundary_landmarks(self, landmarks):
        """Get landmarks that define the facial boundary for better masking."""
        # Face oval landmarks from MediaPipe
        face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 340,
            346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410, 454,
            227, 116, 117, 118, 119, 120, 121, 128, 126, 142, 36, 205,
            206, 207, 213, 192, 147, 123, 116, 117, 118, 119, 120, 121
        ]
        
        # Filter valid indices
        valid_indices = [idx for idx in face_oval_indices if idx < len(landmarks)]
        
        # Return corresponding landmark points
        return [landmarks[idx] for idx in valid_indices]

    def _alpha_blend(self, background, foreground, mask):
        """Fallback alpha blending method."""
        mask_norm = mask.astype(np.float64) / 255.0
        mask_norm = np.stack([mask_norm] * 3, axis=2)
        
        blended = background.astype(np.float64) * (1 - mask_norm) + \
                  foreground.astype(np.float64) * mask_norm
        
        return blended.astype(np.uint8)


# Utility functions for standalone use
def quick_face_swap(img1_path, img2_path, output_dir="./", method="improved"):
    """
    Quick face swap utility function for standalone use.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image  
        output_dir: Directory to save results
        method: "basic" or "improved"
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: Could not load images")
        return
    
    swapper = FaceSwapper()
    
    if method == "improved":
        result1, result2 = swapper.improved_face_swap(img1, img2)
    else:
        result1, result2 = swapper.basic_face_swap(img1, img2)
    
    # Save results
    import os
    base1 = os.path.splitext(os.path.basename(img1_path))[0]
    base2 = os.path.splitext(os.path.basename(img2_path))[0]
    
    output1 = os.path.join(output_dir, f"{base1}_swapped_with_{base2}.jpg")
    output2 = os.path.join(output_dir, f"{base2}_swapped_with_{base1}.jpg")
    
    cv2.imwrite(output1, result1)
    cv2.imwrite(output2, result2)
    
    print(f"Face swap completed!")
    print(f"Results saved to: {output1}, {output2}")


if __name__ == "__main__":
    # Example usage
    print("Face Swap Module - Example Usage")
    
    # Example with file paths (uncomment to use)
    # quick_face_swap("assets/img1.png", "assets/img2.png", method="improved")
    
    # Example with direct images
    swapper = FaceSwapper()
    print("Face swapper initialized successfully!")
    print("Available methods:")
    print("- basic_face_swap(img1, img2)")
    print("- improved_face_swap(img1, img2)")  
    print("- interactive_triangle_swap(img1, img2, parent_window)")
    print("- visualize_triangles(img1, img2, num_triangles)")