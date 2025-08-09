import numpy as np
import cv2
import os
from PIL import Image
import sys
import openpyxl  # Import thư viện openpyxl
from openpyxl import Workbook

sys.stdout.reconfigure(encoding='utf-8')

def read_image(image_path):
    try:
        img = Image.open(image_path).convert('L')
        img = np.array(img)
        return img
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

def resize_image(image, new_size=(100, 100)):
    h, w = image.shape
    new_h, new_w = new_size
    resized = np.zeros(new_size, dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            x = int(i * h / new_h)
            y = int(j * w / new_w)
            resized[i, j] = image[x, y]
    return resized

def augment_image(image):
    angle = np.random.uniform(-10, 10)
    h, w = image.shape
    M = np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
    ])
    rotated = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            coords = np.dot(M, [i, j, 1])
            x, y = int(coords[0]), int(coords[1])
            if 0 <= x < h and 0 <= y < w:
                rotated[i, j] = image[x, y]
    brightness = np.random.uniform(0.8, 1.2)
    augmented = np.clip(rotated * brightness, 0, 255).astype(np.uint8)
    return augmented

def load_dataset(dataset_path):
    face_images = []
    face_labels = []
    non_face_images = []
    label_map = {}
    current_label = 0
    
    face_path = os.path.join(dataset_path, 'face')
    non_face_path = os.path.join(dataset_path, 'non_face')
    if not os.path.exists(face_path):
        raise FileNotFoundError(f"Face directory not found: {face_path}")
    if not os.path.exists(non_face_path):
        raise FileNotFoundError(f"Non-face directory not found: {non_face_path}")
    
    for person_folder in os.listdir(face_path):
        person_path = os.path.join(face_path, person_folder)
        if os.path.isdir(person_path):
            label_map[person_folder] = current_label
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                image = read_image(image_path)
                if image is not None:
                    image = resize_image(image, new_size=(100, 100))
                    face_images.append(image)
                    face_labels.append(current_label)
                    for _ in range(1):  # 1 ảnh tăng cường
                        augmented = augment_image(image)
                        augmented = resize_image(augmented, new_size=(100, 100))
                        face_images.append(augmented)
                        face_labels.append(current_label)
            current_label += 1
    
    for image_file in os.listdir(non_face_path):
        image_path = os.path.join(non_face_path, image_file)
        image = read_image(image_path)
        if image is not None:
            image = resize_image(image, new_size=(100, 100))
            non_face_images.append(image)
    
    if not face_images or not non_face_images:
        raise ValueError("No valid images found in dataset")
    
    face_labels_one_hot = np.eye(len(label_map))[face_labels]
    return face_images, face_labels, face_labels_one_hot, non_face_images, label_map

def detect_and_crop_face(image, adaboost_features=None, adaboost_weights=None, adaboost_thresholds=None):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Lỗi: Không thể tải Haar Cascade classifier!")
        return None
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.01, minNeighbors=1, minSize=(15, 15))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        margin = int(0.2 * max(w, h))
        x, y = max(0, x-margin), max(0, y-margin)
        w, h = min(image.shape[1]-x, w+2*margin), min(image.shape[0]-y, h+2*margin)
        return image[y:y+h, x:x+w]
    return None

def clahe(image, clip_limit=2.0, grid_size=8):
    h, w = image.shape
    tile_h, tile_w = h // grid_size, w // grid_size
    equalized = np.zeros_like(image)
    for i in range(0, h, tile_h):
        for j in range(0, w, tile_w):
            tile = image[i:i+tile_h, j:j+tile_w]
            hist, bins = np.histogram(tile.flatten(), bins=256, range=(0, 255))
            hist = np.clip(hist, 0, clip_limit * tile.size / 256)
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]
            equalized[i:i+tile_h, j:j+tile_w] = np.interp(tile, bins[:-1], cdf_normalized)
    return equalized.astype(np.uint8)

def lbp(image):
    h, w = image.shape
    lbp_image = np.zeros_like(image)
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = image[i, j]
            code = (image[i-1, j-1] > center) << 7 | (image[i-1, j] > center) << 6 | \
                   (image[i-1, j+1] > center) << 5 | (image[i, j+1] > center) << 4 | \
                   (image[i+1, j+1] > center) << 3 | (image[i+1, j] > center) << 2 | \
                   (image[i+1, j-1] > center) << 1 | (image[i, j-1] > center)
            lbp_image[i, j] = code
    return lbp_image.flatten()

def preprocess_and_extract_features(image, adaboost_features=None, adaboost_weights=None, adaboost_thresholds=None):
    cropped = detect_and_crop_face(image)
    if cropped is None:
        return None
    cropped = clahe(cropped)
    cropped = resize_image(cropped, new_size=(100, 100))
    features = lbp(cropped)
    return features

def pca_with_whitening(X, n_components=50):
    if X.size == 0 or X.shape[0] == 0:
        raise ValueError("Empty feature array, cannot perform PCA")
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov_matrix = np.dot(X_centered.T, X_centered) / X.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components]
    X_reduced = np.dot(X_centered, eigenvectors)
    X_reduced = X_reduced / np.sqrt(eigenvalues + 1e-6)
    return X_reduced, eigenvectors, mean

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred, W, lambda_reg=0.01):
    return -np.sum(y_true * np.log(y_pred + 1e-10)) / y_true.shape[0] + lambda_reg * np.sum(W**2)

def train_softmax_minibatch(X, y, n_classes, batch_size=32, learning_rate=0.01, n_epochs=100):
    W = np.random.randn(n_classes, X.shape[1]) * 0.01
    b = np.zeros((1, n_classes))
    n_samples = X.shape[0]
    for epoch in range(n_epochs):
        lr = learning_rate / (1 + 0.01 * epoch)
        indices = np.random.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:min(i+batch_size, n_samples)]
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            scores = np.dot(X_batch, W.T) + b
            y_pred = softmax(scores)
            grad = y_pred - y_batch
            grad_W = np.dot(grad.T, X_batch) / X_batch.shape[0] + 2 * 0.01 * W
            grad_b = np.sum(grad, axis=0, keepdims=True) / X_batch.shape[0]
            W -= lr * grad_W
            b -= lr * grad_b
        if epoch % 10 == 0:
            loss = cross_entropy_loss(y, softmax(np.dot(X, W.T) + b), W)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return W, b

def predict_softmax(X, W, b):
    scores = np.dot(X, W.T) + b
    y_pred = softmax(scores)
    return np.argmax(y_pred, axis=1)

def train_model(dataset_path, n_classes):
    print("Bắt đầu huấn luyện mô hình...")
    try:
        face_images, face_labels, face_labels_one_hot, non_face_images, label_map = load_dataset(dataset_path)
        print(f"Loaded {len(face_images)} face images, {len(non_face_images)} non-face images")
        
        X_train = []
        valid_labels = []
        for idx, (img, label) in enumerate(zip(face_images, face_labels)):
            features = preprocess_and_extract_features(img)
            if features is not None:
                X_train.append(features)
                valid_labels.append(label)
            else:
                person_name = [name for name, l in label_map.items() if l == label][0]
                print(f"Failed to detect face in image {idx} for person {person_name}")
        X_train = np.array(X_train)
        valid_labels_one_hot = np.eye(n_classes)[valid_labels]
        print(f"Number of successfully processed face images: {len(X_train)}")
        
        if len(X_train) == 0:
            raise ValueError("No face images were successfully processed")
        
        print("Bắt đầu PCA và huấn luyện softmax...")
        X_reduced, components, mean = pca_with_whitening(X_train, n_components=50)
        W, b = train_softmax_minibatch(X_reduced, valid_labels_one_hot, n_classes)
        print("Huấn luyện hoàn tất!")
        
        return W, b, components, mean, label_map, None, None, None
    except Exception as e:
        print(f"Lỗi trong train_model: {e}")
        raise

def predict_from_camera(W, b, components, mean, label_map, adaboost_features=None, adaboost_weights=None, adaboost_thresholds=None):
    print("Bắt đầu nhận diện qua camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera")
        return
    
    # --- Tích hợp Excel ---
    wb = Workbook()
    ws = wb.active
    ws.title = "Nhan Dien Khuon Mat"
    ws.append(["Ten Nguoi Duoc Nhan Dien"])
    recorded_names = set()
    excel_filename = "ket_qua_nhan_dien.xlsx"
    # --- Kết thúc tích hợp Excel ---

    predictions = []
    smoothed_name = "Unknown"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể chụp khung hình")
                break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = resize_image(frame_gray, new_size=(300, 300))
            features = preprocess_and_extract_features(frame_gray)

            if features is not None:
                features_reduced = np.dot(features - mean, components)
                predicted_label = predict_softmax(features_reduced[np.newaxis, :], W, b)[0]
                predicted_name = [name for name, idx in label_map.items() if idx == predicted_label][0]
                predictions.append(predicted_label)
                
                if len(predictions) >= 5:
                    smoothed_label = np.bincount(predictions[-5:]).argmax()
                    smoothed_name = [name for name, idx in label_map.items() if idx == smoothed_label][0]
                    print(f"Nhận diện được: {smoothed_name}")
                    
                    # --- Ghi vào file Excel nếu chưa có ---
                    if smoothed_name != "Unknown" and smoothed_name not in recorded_names:
                        ws.append([smoothed_name])
                        recorded_names.add(smoothed_name)
                        wb.save(excel_filename)
                        print(f"Đã ghi tên '{smoothed_name}' vào file {excel_filename}")
                    # --- Kết thúc ghi vào file Excel ---

            else:
                smoothed_name = "Unknown"
                print("Không phát hiện khuôn mặt")
                
            cv2.putText(frame, smoothed_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Đã tắt camera")
        # --- Lưu file Excel lần cuối khi thoát ---
        wb.save(excel_filename)
        print(f"File Excel '{excel_filename}' đã được lưu.")
        # --- Kết thúc lưu file Excel ---

if __name__ == "__main__":
    dataset_path = "D:/PYCHARM - Copy/data"
    n_classes = 10 
    print("Khởi chạy chương trình...")
    try:
        W, b, components, mean, label_map, _, _, _ = train_model(dataset_path, n_classes)
        predict_from_camera(W, b, components, mean, label_map)
    except Exception as e:
        print(f"Lỗi trong main: {e}")