import numpy as np
import cv2
import os
from PIL import Image
import sys
import openpyxl
from openpyxl import Workbook
import time
import random

# Đảm bảo console có thể in ra ký tự UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# --- CÁC HÀM TIỆN ÍCH ---
def read_image(image_path):
    try:
        img = Image.open(image_path).convert('L')
        return np.array(img)
    except Exception as e:
        print(f"Lỗi đọc ảnh {image_path}: {e}")
        return None

# CLAHE tối ưu
clahe_processor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def apply_clahe_optimized(image):
    return clahe_processor.apply(image)

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
            # Thêm uniform LBP: chỉ giữ patterns có <=2 transitions
            transitions = 0
            for k in range(8):
                if (code & (1 << k)) != (code & (1 << ((k+1) % 8))):
                    transitions += 1
            lbp_image[i, j] = code if transitions <= 2 else 0
    return lbp_image.flatten()

def augment_image(image):
    # Xoay ngẫu nhiên
    angle = np.random.uniform(-15, 15)
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Thay đổi độ sáng và contrast
    brightness = np.random.uniform(0.7, 1.3)
    contrast = np.random.uniform(0.8, 1.2)
    augmented = np.clip((rotated * brightness - 127 * (1 - contrast)) + 127, 0, 255).astype(np.uint8)

    # Flip ngang với xác suất 50%
    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)

    # Thêm Gaussian noise
    noise = np.random.normal(0, np.random.uniform(5, 15), augmented.shape).astype(np.uint8)
    augmented = np.clip(augmented + noise, 0, 255).astype(np.uint8)

    return augmented

def load_dataset(dataset_path):
    face_images = []
    face_labels = []
    label_map = {}
    current_label = 0
    
    face_path = os.path.join(dataset_path, 'face')
    if not os.path.exists(face_path):
        raise FileNotFoundError(f"Thư mục khuôn mặt không tìm thấy: {face_path}")
    
    for person_folder in os.listdir(face_path):
        person_path = os.path.join(face_path, person_folder)
        if os.path.isdir(person_path):
            label_map[person_folder] = current_label
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                image = read_image(image_path)
                if image is not None:
                    image_resized = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
                    face_images.append(image_resized)
                    face_labels.append(current_label)
                    for _ in range(3):
                        augmented = augment_image(image_resized)
                        face_images.append(augmented)
                        face_labels.append(current_label)
            current_label += 1
    
    if not face_images:
        raise ValueError("Không tìm thấy ảnh hợp lệ trong tập dữ liệu")
    
    face_labels_one_hot = np.eye(len(label_map))[face_labels]
    return face_images, face_labels, face_labels_one_hot, label_map

# Tải Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_and_crop_face(image):
    # Preprocess toàn frame trước detect
    image_eq = cv2.equalizeHist(image)
    image_clahe = apply_clahe_optimized(image_eq)
    
    # Detect frontal trước
    faces = face_cascade.detectMultiScale(image_clahe, scaleFactor=1.03, minNeighbors=6, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) == 0:
        # Thử profile nếu không detect frontal
        faces = profile_cascade.detectMultiScale(image_clahe, scaleFactor=1.03, minNeighbors=6, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) > 0:
        # Lấy face lớn nhất
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Confirm bằng eye detection
        face_roi = image_clahe[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
        if len(eyes) >= 1:
            margin = int(0.2 * max(w, h))
            x_m, y_m = max(0, x - margin), max(0, y - margin)
            w_m, h_m = min(image.shape[1] - x_m, w + 2*margin), min(image.shape[0] - y_m, h + 2*margin)
            cropped_face = image[y_m:y_m + h_m, x_m:x_m + w_m]
            return cropped_face, (x, y, w, h)
    
    return None, None

def preprocess_and_extract_features(image):
    cropped, coords = detect_and_crop_face(image)
    if cropped is None:
        return None, None
    cropped_eq = cv2.equalizeHist(cropped)
    cropped_clahe = apply_clahe_optimized(cropped_eq)
    cropped_resized = cv2.resize(cropped_clahe, (100, 100), interpolation=cv2.INTER_AREA)
    features = lbp(cropped_resized)
    return features, coords

def pca_with_whitening(X, explained_variance=0.95):
    if X.size == 0: raise ValueError("Mảng đặc trưng rỗng")
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Tính cumulative variance và chọn n_components
    cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    n_components = np.argmax(cum_var >= explained_variance) + 1
    print(f"Sử dụng {n_components} components để giữ {explained_variance*100}% variance.")
    
    X_reduced = np.dot(X_centered, eigenvectors[:, :n_components])
    X_whitened = X_reduced / np.sqrt(eigenvalues[:n_components] + 1e-6)
    return X_whitened, eigenvectors[:, :n_components], mean

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred, W, lambda_reg=0.01):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
    loss = np.sum(log_likelihood) / m
    reg_loss = 0.5 * lambda_reg * np.sum(W * W)
    return loss + reg_loss

def train_softmax_minibatch(X_train, y_train, X_val, y_val, n_classes, batch_size=32, learning_rate=0.01, n_epochs=200, patience=10):
    W = np.random.randn(X_train.shape[1], n_classes) * 0.01
    b = np.zeros((1, n_classes))
    n_samples = X_train.shape[0]
    best_loss = float('inf')
    patience_counter = 0
    lr = learning_rate
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, n_samples)]
            X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]
            scores = np.dot(X_batch, W) + b
            y_pred = softmax(scores)
            
            grad = y_pred - y_batch
            grad_W = np.dot(X_batch.T, grad) / batch_size + 0.01 * W
            grad_b = np.sum(grad, axis=0, keepdims=True) / batch_size
            
            W -= lr * grad_W
            b -= lr * grad_b
            
        if epoch % 10 == 0:
            full_scores = np.dot(X_train, W) + b
            loss = cross_entropy_loss(y_train, softmax(full_scores), W)
            val_scores = np.dot(X_val, W) + b
            val_loss = cross_entropy_loss(y_val, softmax(val_scores), W)
            val_pred = np.argmax(softmax(val_scores), axis=1)
            val_true = np.argmax(y_val, axis=1)
            val_acc = np.mean(val_pred == val_true)
            print(f"Epoch {epoch}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # Early stopping dựa trên val_loss
        val_scores = np.dot(X_val, W) + b
        val_loss = cross_entropy_loss(y_val, softmax(val_scores), W)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping...")
                break
        
        # Learning rate decay
        if epoch % 20 == 0 and epoch > 0:
            lr *= 0.9
    
    return W, b

# --- HÀM PREDICT TRẢ VỀ XÁC SUẤT ---
def predict_softmax(X, W, b):
    scores = np.dot(X, W.T) + b
    y_pred_probs = softmax(scores)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    max_probs = np.max(y_pred_probs, axis=1)
    return y_pred_labels, max_probs

def train_model(dataset_path, n_classes):
    print("Bắt đầu huấn luyện mô hình...")
    face_images, face_labels, face_labels_one_hot, label_map = load_dataset(dataset_path)
    print(f"Đã tải {len(face_images)} ảnh khuôn mặt.")
    
    X_train = np.array([lbp(img) for img in face_images])
    y_train = face_labels_one_hot
    
    print("Bắt đầu PCA và huấn luyện softmax...")
    X_reduced, components, mean = pca_with_whitening(X_train)

    # Split thủ công
    indices = np.arange(X_reduced.shape[0])
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    X_train_split = X_reduced[train_idx]
    y_train_split = y_train[train_idx]
    X_val = X_reduced[val_idx]
    y_val = y_train[val_idx]

    W, b_row = train_softmax_minibatch(X_train_split, y_train_split, X_val, y_val, n_classes)
    W_T = W.T
    print("Huấn luyện hoàn tất!")
    return W_T, b_row, components, mean, label_map

def predict_from_camera(W, b, components, mean, label_map):
    print("Bắt đầu nhận diện qua camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera")
        return

    SKIP_FRAMES = 3
    frame_counter = 0
    last_coords = None
    smoothed_name = "Unknown"
    predictions = []

    CONFIDENCE_THRESHOLD = 0.65

    wb = Workbook()
    ws = wb.active
    ws.title = "Nhan Dien Khuon Mat"
    ws.append(["Ten Nguoi Duoc Nhan Dien"])
    recorded_names = set()
    excel_filename = "ket_qua_nhan_dien.xlsx"

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_counter += 1
        if frame_counter % SKIP_FRAMES == 0:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_frame = cv2.resize(frame_gray, (0, 0), fx=0.4, fy=0.4)  # Giảm fx để tăng FPS
            features, coords_small = preprocess_and_extract_features(small_frame)

            if features is not None and coords_small is not None:
                x, y, w, h = [int(v * 2.5) for v in coords_small]  # Tỷ lệ với fx=0.4
                last_coords = (x, y, w, h)

                features_centered = features.reshape(1, -1) - mean
                features_reduced = np.dot(features_centered, components)

                predicted_label, max_prob = predict_softmax(features_reduced, W, b)
                predicted_label = predicted_label[0]
                max_prob = max_prob[0]

                if max_prob < CONFIDENCE_THRESHOLD:
                    smoothed_name = "Unknown"
                else:
                    predictions.append(predicted_label)
                    if len(predictions) > 10: predictions.pop(0)
                    smoothed_label = np.bincount(predictions).argmax()
                    smoothed_name = [name for name, idx in label_map.items() if idx == smoothed_label][0]

                    if smoothed_name not in recorded_names:
                        ws.append([smoothed_name])
                        recorded_names.add(smoothed_name)
                        wb.save(excel_filename)
            else:
                last_coords = None
        
        if last_coords is not None:
            x, y, w, h = last_coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, smoothed_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Camera - Nhan Q de thoat', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    try:
        wb.save(excel_filename)
        print(f"File Excel '{excel_filename}' đã được lưu.")
    except Exception as e:
        print(f"Lỗi khi lưu file Excel: {e}")

if __name__ == "__main__":
    dataset_path = "C:/AI PROJECT/data/data"
    try:
        face_dir = os.path.join(dataset_path, 'face')
        n_classes = len([name for name in os.listdir(face_dir) if os.path.isdir(os.path.join(face_dir, name))])
        if n_classes == 0: raise ValueError("Không tìm thấy thư mục người nào trong 'data/face'")
        print(f"Tìm thấy {n_classes} lớp (người) để huấn luyện.")
    except Exception as e:
        print(f"Lỗi khi xác định số lớp: {e}")
        sys.exit(1)

    print("Khởi chạy chương trình...")
    try:
        W, b, components, mean, label_map = train_model(dataset_path, n_classes)
        predict_from_camera(W, b, components, mean, label_map)
    except Exception as e:
        print(f"Lỗi nghiêm trọng trong luồng chính: {e}")
