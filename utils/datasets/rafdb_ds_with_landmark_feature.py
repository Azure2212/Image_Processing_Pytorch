from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
import pandas as pd
import os
import numpy as np
import cv2

def make_augmentation_image_landmark_custom(image, face_landmarks): 
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

class image_with_landmark_RafDataSet(Dataset):
    def __init__(self, data_type, configs, ttau=False, device='cpu'):
        self.device = device
        self.configs = configs
        self.data_type = data_type
        self.shape = (configs["image_size"], configs["image_size"])

        df = pd.read_csv(os.path.join(self.configs["raf_path"], configs["label_path"]), sep=' ', header=None, names=['name', 'label'])

        if data_type == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = np.array(self.data.loc[:, 'name'].values)
        self.labels = np.array(self.data.loc[:, 'label'].values - 1)  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        # Initialize an empty list to store the file names
        path_current = os.path.abspath(globals().get("__file__","."))
        path_current = os.path.abspath(f"{path_current}/..")
        path_images_without_landmarks = []

        # Open the input file in read mode
        file2read = os.path.join(path_current,'rafdb_image_path_without_landmarks.txt')
        with open(file2read, 'r') as f:
            # Read each line in the file
            for line in f:
                # Strip any whitespace characters (like newline) and add to the list
                path_images_without_landmarks.append(line.strip())

        check_in = ~np.isin(file_names, path_images_without_landmarks)
        file_names = file_names[check_in]
        self.labels = self.labels[check_in]


        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.configs["raf_path"], self.configs["image_path"], f)
            self.file_paths.append(path)
                
    def __len__(self):
        return len(self.file_paths)
    def get_landmarks(self, image):
        detected_faces = self.fa_model.face_detector.detect_from_image(image.copy())
        landmarks = []
        landmarks_scores = []
        if(detected_faces == None):
            return None
        
        for i, d in enumerate(detected_faces):
            center = np.array(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / self.fa_model.face_detector.reference_scale
            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()
            inp = inp.to(device =self.device , dtype=torch.float32)
            inp.div_(255.0).unsqueeze_(0)
            out = self.fa_model.face_alignment_net(inp).detach()
            # out = out.to(device='cpu', dtype=torch.float32).numpy()
            # pts, pts_img, scores = get_preds_fromhm(out, center, scale)
            # pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
            # pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
            # scores = scores.squeeze(0)
            # landmarks.append(pts_img.numpy())
            # landmarks_scores.append(scores)
            return out #landmarks
        
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)[:,:,::-1]
        landmarks  = self.landmarks_by_images[idx]
        
        image = cv2.resize(image, self.shape)

        if self.data_type == 'train':
            image, feature_landmark = make_augmentation_image_landmark_custom(image, landmarks)
            #landmarks = landmarks[0]
        return image.transpose(2, 0, 1), feature_landmark