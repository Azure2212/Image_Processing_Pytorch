import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')
#create mask for each face part
class  mediapipe_tool():
    def __init__(self):
        self.image = None
    def draw_face_mask_segmentation(self, image_rgb, output_path=None):
        """
        Detects facial landmarks in an image and creates a mask where the face area is black and the rest is white.
        
        Args:
            image_rgb (numpy.ndarray): Input image in RGB format.
            output_path (str, optional): Path to save the output image with segmentation mask. If None, the image is not saved.
        """
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            # Perform face mesh detection
            results = face_mesh.process(image_rgb)

            # Convert image to uint8 if it's not already
            if image_rgb.dtype != np.uint8:
                image_rgb = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)

            # Initialize the mask to 255 (white image)
            ih, iw, _ = image_rgb.shape
            mask = np.ones((ih, iw), dtype=np.uint8)

            # Create a mask based on facial landmarks
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    points = []
                    for landmark in landmarks.landmark:
                        x = int(landmark.x * iw)
                        y = int(landmark.y * ih)
                        points.append((x, y))
                    
                    # Convert list of points to a NumPy array
                    points = np.array(points, dtype=np.int32)
                    
                    # Draw the convex hull of the face landmarks on the mask
                    convex_hull = cv2.convexHull(points)
                    cv2.fillConvexPoly(mask, convex_hull, 255)  # Fill the convex hull with black (0)

        # Display the mask using matplotlib
    #     plt.imshow(mask, cmap='gray')
    #     plt.axis('off')  # Hide the axis

    #     # Save the mask if output_path is provided
    #     if output_path:
    #         plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    #     else:
    #         plt.show()
            
        return mask

    def draw_eyes_mask_segmentation(self, image_rgb, output_path=None):
        """
        Detects facial landmarks in an image and creates a mask where the eye areas are black and the rest is white.
        
        Args:
            image_rgb (numpy.ndarray): Input image in RGB format.
            output_path (str, optional): Path to save the output image with segmentation mask. If None, the image is not saved.
        """
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            # Perform face mesh detection
            results = face_mesh.process(image_rgb)

            # Convert image to uint8 if it's not already
            if image_rgb.dtype != np.uint8:
                image_rgb = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)

            # Initialize the mask to 255 (white image)
            ih, iw, _ = image_rgb.shape
            mask = np.ones((ih, iw), dtype=np.uint8)

            # Define landmark indices for the eyes (using MediaPipe's face mesh index)
            left_eye_indices = [33, 160, 158, 133, 153, 144, 163, 7]
            right_eye_indices = [362, 385, 386, 263, 373, 374, 380, 249]

            # Create a mask based on eye landmarks
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    for eye_indices in [left_eye_indices, right_eye_indices]:
                        points = []
                        for index in eye_indices:
                            landmark = landmarks.landmark[index]
                            x = int(landmark.x * iw)
                            y = int(landmark.y * ih)
                            points.append((x, y))
                        
                        # Convert list of points to a NumPy array
                        points = np.array(points, dtype=np.int32)
                        
                        # Draw the convex hull of the eye landmarks on the mask
                        convex_hull = cv2.convexHull(points)
                        cv2.fillConvexPoly(mask, convex_hull, 255)  # Fill the convex hull with black (0)

        # Display the mask using matplotlib
    #     plt.imshow(mask, cmap='gray')
    #     plt.axis('off')  # Hide the axis

    #     # Save the mask if output_path is provided
    #     if output_path:
    #         plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    #     else:
    #         plt.show()
            
        return mask

    def draw_nose_mask_segmentation(self, image_rgb, output_path=None):
        """
        Detects facial landmarks in an image and creates a mask where the eye areas are black and the rest is white.
        
        Args:
            image_rgb (numpy.ndarray): Input image in RGB format.
            output_path (str, optional): Path to save the output image with segmentation mask. If None, the image is not saved.
        """
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            # Perform face mesh detection
            results = face_mesh.process(image_rgb)

            # Convert image to uint8 if it's not already
            if image_rgb.dtype != np.uint8:
                image_rgb = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)

            # Initialize the mask to 255 (white image)
            ih, iw, _ = image_rgb.shape
            mask = np.ones((ih, iw), dtype=np.uint8)

            # Define landmark indices for the eyes (using MediaPipe's face mesh index)
            nose_indices = [1,2,3,4,5,6,8,19,20,44,45,48,49,51,59,60,64,75,79,94,97,98,99,102,114,115,122,125,128,126,129
    ,131,134,141,166,168,174,188,193,195,196,197,198,209,217,218,219,220,235,236,237,238,239,240,241,242,245,248,250
    ,274,275,281,289,290,294,305,309,326,327,328,331,344,351,354,358,360,363,370,392,399,412,417,419,420,437,438,439,440  
    ,455,456,457,458,459,460,461,462]

            # Create a mask based on eye landmarks
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    points = []
                    for index in nose_indices:
                        landmark = landmarks.landmark[index]
                        x = int(landmark.x * iw)
                        y = int(landmark.y * ih)
                        points.append((x, y))

                    # Convert list of points to a NumPy array
                    points = np.array(points, dtype=np.int32)

                    # Draw the convex hull of the eye landmarks on the mask
                    convex_hull = cv2.convexHull(points)
                    cv2.fillConvexPoly(mask, convex_hull, 255)  # Fill the convex hull with black (0)

        # Display the mask using matplotlib
    #     plt.imshow(mask, cmap='gray')
    #     plt.axis('off')  # Hide the axis

    #     # Save the mask if output_path is provided
    #     if output_path:
    #         plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    #     else:
    #         plt.show()
            
        return mask

    def draw_mouth_mask_segmentation(self, image_rgb, output_path=None):
        """
        Detects facial landmarks in an image and creates a mask where the eye areas are black and the rest is white.
        
        Args:
            image_rgb (numpy.ndarray): Input image in RGB format.
            output_path (str, optional): Path to save the output image with segmentation mask. If None, the image is not saved.
        """
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            # Perform face mesh detection
            results = face_mesh.process(image_rgb)

            # Convert image to uint8 if it's not already
            if image_rgb.dtype != np.uint8:
                image_rgb = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)

            # Initialize the mask to 255 (white image)
            ih, iw, _ = image_rgb.shape
            mask = np.ones((ih, iw), dtype=np.uint8)

            # Define landmark indices for the eyes (using MediaPipe's face mesh index)
            mouth_indices = [11,12,13,14,15,16,17,37,38,39,40,41,42,61,62,72,73,74,76,77,78,78,80,81,82,84,85,86,87,88
    ,89,90,91,95,96,146,178,179,180,181,183,184,185,191,267,268,269,270,271,272,291,292,302,303,304,306,307,308
    ,310,311,312,314,315,316,317,318,319,320,321,324,325,375,402,403,404,405,407,408,409,415]

            # Create a mask based on eye landmarks
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    points = []
                    for index in mouth_indices:
                        landmark = landmarks.landmark[index]
                        x = int(landmark.x * iw)
                        y = int(landmark.y * ih)
                        points.append((x, y))

                    # Convert list of points to a NumPy array
                    points = np.array(points, dtype=np.int32)

                    # Draw the convex hull of the eye landmarks on the mask
                    convex_hull = cv2.convexHull(points)
                    cv2.fillConvexPoly(mask, convex_hull, 255)  # Fill the convex hull with black (0)

        # Display the mask using matplotlib
    #     plt.imshow(mask, cmap='gray')
    #     plt.axis('off')  # Hide the axis

    #     # Save the mask if output_path is provided
    #     if output_path:
    #         plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    #     else:
    #         plt.show()
            
        return mask


    def draw_eyebrows_mask_segmentation(self, image_rgb, output_path=None):
        """
        Detects facial landmarks in an image and creates a mask where the eye areas are black and the rest is white.
        
        Args:
            image_rgb (numpy.ndarray): Input image in RGB format.
            output_path (str, optional): Path to save the output image with segmentation mask. If None, the image is not saved.
        """
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            # Perform face mesh detection
            results = face_mesh.process(image_rgb)

            # Convert image to uint8 if it's not already
            if image_rgb.dtype != np.uint8:
                image_rgb = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)

            # Initialize the mask to 255 (white image)
            ih, iw, _ = image_rgb.shape
            mask = np.ones((ih, iw), dtype=np.uint8)

            # Define landmark indices for the eyes (using MediaPipe's face mesh index)
            left_eyebrow_indices = [46,52,53,63,65,66,68,70,71,105 ]
            right_eyebrow_indices = [282,283,293,296,298,300,301,334 ]

            # Create a mask based on eye landmarks
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    for eye_indices in [left_eyebrow_indices, right_eyebrow_indices]:
                        points = []
                        for index in eye_indices:
                            landmark = landmarks.landmark[index]
                            x = int(landmark.x * iw)
                            y = int(landmark.y * ih)
                            points.append((x, y))
                        
                        # Convert list of points to a NumPy array
                        points = np.array(points, dtype=np.int32)
                        
                        # Draw the convex hull of the eye landmarks on the mask
                        convex_hull = cv2.convexHull(points)
                        cv2.fillConvexPoly(mask, convex_hull, 255)  # Fill the convex hull with black (0)

        # Display the mask using matplotlib
    #     plt.imshow(mask, cmap='gray')
    #     plt.axis('off')  # Hide the axis

    #     # Save the mask if output_path is provided
    #     if output_path:
    #         plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    #     else:
    #         plt.show()
            
        return mask
