from imgaug import augmenters as iaa
import albumentations as A

from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import random
#
seg_fer = iaa.Sometimes(
        0.5,
	iaa.Sequential([iaa.Fliplr(p=0.6),iaa.Affine(rotate=(-30, 30))]),
        iaa.Sequential([iaa.Affine(scale=(1.0, 1.1))]),

)
seg_fertest2 = iaa.Sequential([iaa.Affine(scale=(1.0, 1.1))])
seg_fertest1 = iaa.Sequential([iaa.Fliplr(p=0.5),iaa.Affine(rotate=(-30, 30))])


#augmentaion for Image-net
seg_raf = iaa.Sometimes(
        0.5,
	iaa.Sequential([iaa.Fliplr(p=0.5), iaa.Affine(rotate=(-25, 25))]),
        iaa.Sequential([iaa.RemoveSaturation(1),iaa.Affine(scale=(1.0, 1.05)) ])
)
seg_raftest2 = iaa.Sequential([iaa.RemoveSaturation(1),iaa.Affine(scale=(1.0, 1.05))])
seg_raftest1 = iaa.Sequential([iaa.Fliplr(p=0.5), iaa.Affine(rotate=(-25, 25))])



# Define transformations for image
image_transform = A.Compose([
    A.Resize(width=224, height=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Define transformations for mask (no normalization)
mask_transform = A.Compose([
    A.Resize(width=224, height=224, always_apply=True),  # Ensure mask is resized
    ToTensorV2()
])

# Combine transformations into one with additional_targets
class CustomTransform:
    def __init__(self):
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, image, mask=None):
        # Apply image transformations
        transformed_image = self.image_transform(image=image)['image']
        
        if mask is not None:
            # Apply mask transformations if mask is provided
            transformed_mask = self.mask_transform(image=mask)['image']
            return {'image': transformed_image, 'mask': transformed_mask}
        return {'image': transformed_image}

#custome augmentation for landmark end point - end point
def make_augmentation_image_landmark_custom(image, face_landmarks): 
    import face_alignment
    def A_Horizontal_flip(image, landmarks):
        flipped_image = cv2.flip(image, 1)  # Lật ngang hình ảnh
        h, w, _ = image.shape

        # Điều chỉnh tọa độ landmarks
        flipped_landmarks = landmarks.copy()
        flipped_landmarks[:, 0] = w - landmarks[:, 0]  # Thay đổi tọa độ x

        return flipped_image, flipped_landmarks

    def A_Vertical_flip(image, landmarks):
        flipped_image = cv2.flip(image, 0)  # Lật dọc hình ảnh
        h, w, _ = image.shape

        # Điều chỉnh tọa độ landmarks
        flipped_landmarks = landmarks.copy()
        flipped_landmarks[:, 1] = h - landmarks[:, 1]  # Thay đổi tọa độ y

        return flipped_image, flipped_landmarks

    def A_Perspective(image, landmarks):
        h, w, _ = image.shape
        value = 30
        # Định nghĩa các điểm nguồn và điểm đích cho biến đổi phối cảnh
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst_points = np.float32([[random.uniform(-value, value), random.uniform(-value, value)],
                                  [w + random.uniform(-value, value), random.uniform(-value, value)],
                                  [random.uniform(-value, value), h + random.uniform(-value, value)],
                                  [w + random.uniform(-value, value), h + random.uniform(-value, value)]])

        # Tính toán ma trận biến đổi phối cảnh
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Áp dụng biến đổi cho hình ảnh
        transformed_image = cv2.warpPerspective(image, matrix, (w, h))

        # Áp dụng biến đổi cho các điểm landmark
        landmarks_h = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])  # Chuyển đổi sang dạng đồng nhất
        transformed_landmarks = matrix.dot(landmarks_h.T).T  # Áp dụng ma trận biến đổi

        # Chia cho phần tử thứ 3 để đưa về hệ tọa độ 2D
        transformed_landmarks = transformed_landmarks[:, :2] / transformed_landmarks[:, 2][:, np.newaxis]

        return transformed_image, transformed_landmarks
    
    def A_Rotate(my_image, landmarks):
        # Chọn góc quay ngẫu nhiên từ -45 đến 45 độ
        angle = np.random.uniform(-45, 45)
     
        # Quay hình ảnh
        (h, w) = my_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(my_image, M, (w, h))

        # Cập nhật landmarks
        theta = np.radians(-angle)  # Lấy âm của góc
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta), np.cos(theta)]])

        # Cập nhật tọa độ landmarks
        rotated_landmarks = landmarks - center
        rotated_landmarks = rotated_landmarks @ rotation_matrix.T
        rotated_landmarks += center

        return rotated_image, rotated_landmarks
    
#     if random.random() < 0.5:
#         image, face_landmarks = A_Horizontal_flip(image, face_landmarks)
#     if random.random() < 0.5:
#         random_number = random.choice([0, 1])
#         if random_number == 0:
#             image, face_landmarks = A_Vertical_flip(image, face_landmarks)
#         else:
#             image, face_landmarks = A_Rotate(image, face_landmarks)
#     if random.random() < 0.5:
#         image, face_landmarks = A_Perspective(image, face_landmarks)

    if random.random() < 0.9:
        random_number = random.choice([0, 1, 2])
        if random_number == 0:
            transform = A.Compose([A.CLAHE(p=1.0, clip_limit=2.0, tile_grid_size=(8, 8))])
        elif random_number == 1:
            transform = A.Compose([A.RandomBrightnessContrast(p=1)])
        elif random_number == 2:
            transform = A.Compose([A.RandomGamma(p=1)])
        augmented = transform(image=image)
        image = augmented['image']
        
    if random.random() < 0.9:
        random_number = random.choice([0, 1, 2])
        if random_number == 0:
            transform = A.Compose([A.Sharpen(p=1)])
        elif random_number == 1:
            transform = A.Compose([A.Blur(blur_limit=3, p=1)])
        elif random_number == 2:
            transform = A.Compose([A.MotionBlur(blur_limit=3, p=1)])
        augmented = transform(image=image)
        image = augmented['image']
        
    if random.random() < 0.9:
        random_number = random.choice([0, 1])
        if random_number == 0:
            transform = A.Compose([A.RandomBrightnessContrast(p=1)])
        else:
            transform = A.Compose([A.HueSaturationValue(p=1)])
        augmented = transform(image=image)
        image = augmented['image']
            
            
    return image, face_landmarks


#custome augmentation for landmark feature - (bounding box)
def make_augmentation_image_landmark_boundingbox_custom(image, task='resize', device='cpu'): 

    def A_Vertical_Flip_image_boundingbox(image, detected_faces):
        flipped_image = cv2.flip(image, 0)
        original_height = image.shape[0]
        
        adjusted_bbox = detected_faces[0]
        x_min, y_min, x_max, y_max = adjusted_bbox[:4].astype(int)
        
        adjusted_bbox[1] = original_height - y_max
        adjusted_bbox[3] = original_height - y_min
        detected_faces[0] = adjusted_bbox
        return flipped_image, detected_faces

    def A_Horizontal_Flip_image_boundingbox(image, detected_faces):
        flipped_image = cv2.flip(image, 1)
        original_height = image.shape[0]
        
        adjusted_bbox = detected_faces[0]
        x_min, y_min, x_max, y_max = adjusted_bbox[:4].astype(int)
        
        adjusted_bbox[0] = original_height - x_max
        adjusted_bbox[2] = original_height - x_min
        
        detected_faces[0] = adjusted_bbox
        return flipped_image, detected_faces

    def A_Rotate_image_boundingbox(image, detected_faces):
        
        adjusted_bbox = detected_faces[0]
        bbox_value = adjusted_bbox[:4].astype(int)  # [x_min, y_min, x_max, y_max]
        
        angle = np.random.uniform(-45, 45)

        # Tính toán kích thước mới của hình ảnh
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Tạo ma trận xoay
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Xoay hình ảnh
        rotated_image = cv2.warpAffine(image, M, (w, h))

        # Xoay bounding box
        bbox_corners = np.array([
            [bbox_value[0], bbox_value[1]],  # Top-left
            [bbox_value[2], bbox_value[1]],  # Top-right
            [bbox_value[2], bbox_value[3]],  # Bottom-right
            [bbox_value[0], bbox_value[3]]   # Bottom-left
        ])

        # Chuyển đổi sang dạng đồng nhất
        ones = np.ones((bbox_corners.shape[0], 1))
        bbox_corners_homogeneous = np.hstack([bbox_corners, ones])

        # Áp dụng phép biến đổi xoay
        rotated_corners = M.dot(bbox_corners_homogeneous.T).T

        # Tìm tọa độ min và max mới cho bounding box
        #print(adjusted_bbox)
        adjusted_bbox[0] = np.min(rotated_corners[:, 0])
        adjusted_bbox[1] = np.min(rotated_corners[:, 1])
        adjusted_bbox[2] = np.max(rotated_corners[:, 0])
        adjusted_bbox[3] = np.max(rotated_corners[:, 1])
        
        detected_faces[0] = adjusted_bbox
        # Cập nhật bounding box
        return rotated_image, detected_faces

    def A_Perspective_image_boundingbox(image, detected_faces):
        adjusted_bbox = detected_faces[0]
        bbox_value = adjusted_bbox[:4].astype(int)

        # Tính toán các điểm góc của bounding box
        bbox_corners = np.array([
            [bbox_value[0], bbox_value[1]],  # Top-left
            [bbox_value[2], bbox_value[1]],  # Top-right
            [bbox_value[2], bbox_value[3]],  # Bottom-right
            [bbox_value[0], bbox_value[3]]   # Bottom-left
        ], dtype=np.float32)

        # Định nghĩa ma trận biến đổi phối cảnh
        height, width = image.shape[:2]

        # Tạo các điểm nguồn và điểm đích cho biến đổi phối cảnh
        src_points = bbox_corners
        value = 20
        dst_points = np.array([
            [bbox_value[0] + np.random.uniform(-value, value), bbox_value[1] + np.random.uniform(-value, value)],  # Biến đổi ngẫu nhiên
            [bbox_value[2] + np.random.uniform(-value, value), bbox_value[1] + np.random.uniform(-value, value)],
            [bbox_value[2] + np.random.uniform(-value, value), bbox_value[3] + np.random.uniform(-value, value)],
            [bbox_value[0] + np.random.uniform(-value, value), bbox_value[3] + np.random.uniform(-value, value)]
        ], dtype=np.float32)

        # Tính ma trận biến đổi phối cảnh
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Áp dụng biến đổi phối cảnh
        transformed_image = cv2.warpPerspective(image, M, (width, height))

        # Tính toán bounding box mới
        transformed_corners = cv2.perspectiveTransform(bbox_corners.reshape(-1, 1, 2), M).reshape(-1, 2)
        x_min, y_min = np.min(transformed_corners, axis=0)
        x_max, y_max = np.max(transformed_corners, axis=0)

        adjusted_bbox[0] = x_min
        adjusted_bbox[1] = y_min
        adjusted_bbox[2] = x_max
        adjusted_bbox[3] = y_max
        
        detected_faces[0] = adjusted_bbox
        # Cập nhật bounding box
        return transformed_image, detected_faces

    def get_face_box_image_resized(image, device='cpu'):
        fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)
        detected_faces = fa_model.face_detector.detect_from_image(image.copy())
        
        original_height, original_width = image.shape[:2]
        new_size = (256,256)

        scale_x = new_size[0] / original_width
        scale_y = new_size[1] / original_height
        
        adjusted_bbox = detected_faces[0].copy()
        adjusted_bbox[0] *= scale_x  # x_min
        adjusted_bbox[1] *= scale_y  # y_min
        adjusted_bbox[2] *= scale_x  # x_max
        adjusted_bbox[3] *= scale_y  # y_max
        detected_faces[0] = adjusted_bbox
            
        #print(detected_faces)
        resized_image = cv2.resize(image.copy(), new_size)
        #print(detected_faces[0].shape)
        # if random.random() < 0.5: not suitable for landmark
        #     resized_image, detected_faces = A_Vertical_Flip_image_boundingbox(resized_image.copy(), detected_faces.copy())
        if random.random() < 0.5:
            resized_image, detected_faces = A_Horizontal_Flip_image_boundingbox(resized_image.copy(), detected_faces.copy())
        if random.random() < 0.5:
            resized_image, detected_faces = A_Rotate_image_boundingbox(resized_image.copy(), detected_faces.copy())
        if random.random() < 0.5:
            resized_image, detected_faces = A_Perspective_image_boundingbox(resized_image.copy(), detected_faces.copy())
        return resized_image, detected_faces

    
    
    
    if task == 'resize':
        resized_image, detected_faces = get_face_box_image_resized(image.copy(), device='cpu')
        return resized_image, detected_faces
    else:
        if random.random() < 0.9:
            random_number = random.choice([0, 1, 2])
            if random_number == 0:
                transform = A.Compose([A.CLAHE(p=1.0, clip_limit=2.0, tile_grid_size=(8, 8))])
            elif random_number == 1:
                transform = A.Compose([A.RandomBrightnessContrast(p=1)])
            elif random_number == 2:
                transform = A.Compose([A.RandomGamma(p=1)])
            augmented = transform(image=image)
            image = augmented['image']
            
        if random.random() < 0.9:
            random_number = random.choice([0, 1, 2])
            if random_number == 0:
                transform = A.Compose([A.Sharpen(p=1)])
            elif random_number == 1:
                transform = A.Compose([A.Blur(blur_limit=3, p=1)])
            elif random_number == 2:
                transform = A.Compose([A.MotionBlur(blur_limit=3, p=1)])
            augmented = transform(image=image)
            image = augmented['image']
            
        if random.random() < 0.9:
            random_number = random.choice([0, 1])
            if random_number == 0:
                transform = A.Compose([A.RandomBrightnessContrast(p=1)])
            else:
                transform = A.Compose([A.HueSaturationValue(p=1)])
            augmented = transform(image=image)
            image = augmented['image']
                
                
        return image

