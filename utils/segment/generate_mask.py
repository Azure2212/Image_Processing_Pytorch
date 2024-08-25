import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

def get_human_face_mask(image):
    # Validate image input
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input. Expected a non-empty NumPy array.")
    
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    
    try:
        # Convert the image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        raise RuntimeError(f"Error in cv2.cvtColor: {e}")

    # Process the image with FaceMesh
    results = face_mesh.process(rgb_image)

    # Create a blank mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw convex hull around face
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                points.append((x, y))

            # Convert points to numpy array and calculate convex hull
            points = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points)

            # Draw convex hull on mask
            cv2.fillConvexPoly(mask, hull, 255)

            # Optionally draw convex hull on the original image
            cv2.polylines(image, [hull], isClosed=True, color=(0, 255, 0), thickness=2)

    return mask