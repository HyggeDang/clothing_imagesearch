import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model, Model
from PIL import Image

# Tải mô hình đã huấn luyện từ file
model_path = 'models/vgg16_model.h5'  # Đảm bảo mô hình đã được huấn luyện và lưu ở đây
model = load_model(model_path)

# Định nghĩa hàm trích xuất đặc trưng
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Hàm tìm kiếm ảnh tương tự
def find_similar_images(query_image, top_k=5):
    query_features = extract_features(query_image)
    query_features = query_features / np.linalg.norm(query_features)  # Chuẩn hóa

    # Tải các đặc trưng đã lưu từ pickle
    with open('features/vectors.pkl', 'rb') as f:
        features = pickle.load(f)

    with open('features/paths.pkl', 'rb') as f:
        paths = pickle.load(f)

    # Tính toán khoảng cách Euclid giữa ảnh truy vấn và tất cả ảnh trong tập dữ liệu
    distances = euclidean_distances([query_features], features)[0]

    # Lấy top 5 ảnh tương tự nhất
    top_k_indices = np.argsort(distances)[:top_k]
    similar_images = [(paths[i], distances[i]) for i in top_k_indices]

    return similar_images

# Hàm hiển thị ảnh với matplotlib
def display_images_with_similarity(query_image, similar_images):
    # Vẽ ảnh truy vấn
    fig, axes = plt.subplots(1, len(similar_images) + 1, figsize=(15, 5))
    
    # Hiển thị ảnh truy vấn
    query_img = Image.open(query_image)
    axes[0].imshow(query_img)
    axes[0].set_title("Ảnh đầu vào")
    axes[0].axis('off')
    
    # Hiển thị các ảnh tương tự
    for i, (img_path, dist) in enumerate(similar_images):
        similar_img = Image.open(img_path)
        axes[i + 1].imshow(similar_img)
        axes[i + 1].set_title(f"Độ tương đồng: {dist:.4f}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()

# Test tìm kiếm ảnh tương tự với một ảnh truy vấn
query_image = 'data/test_images/test_34.jpg'  # Thay thế với đường dẫn ảnh truy vấn
similar_images = find_similar_images(query_image)

# Hiển thị ảnh và độ tương đồng
display_images_with_similarity(query_image, similar_images)
