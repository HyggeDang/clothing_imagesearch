import numpy as np
from tensorflow.keras.models import load_model, Model  
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import os

# Tải mô hình đã huấn luyện
model_path = 'models/vgg16_product_model.h5'  # Đường dẫn tới mô hình đã huấn luyện
model = load_model(model_path)

# Tạo model VGG16 để trích xuất đặc trưng
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)  # Dòng này đã sửa

# Đọc các đặc trưng đã được lưu (sử dụng pickle)
FEATURES_FILE = 'image_features.pkl'
with open(FEATURES_FILE, 'rb') as f:
    image_features = pickle.load(f)  # Dữ liệu đặc trưng từ file pickle

# Kiểm tra các đường dẫn ảnh trong pickle
print("Các đường dẫn ảnh trong pickle:")
for image_path in list(image_features.keys())[:5]:
    print(image_path)

# Hàm tiền xử lý hình ảnh (dùng cho VGG16)
def preprocess_image(img_path):
    img = Image.open(img_path)  # Đọc ảnh từ file
    img = img.resize((224, 224))  # Đảm bảo ảnh có kích thước 224x224
    img_array = np.array(img)  # Chuyển ảnh thành mảng numpy
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch (1, 224, 224, 3)
    img_array = preprocess_input(img_array)  # Tiền xử lý (chuẩn hóa cho VGG16)
    return img_array

# Hàm trích xuất đặc trưng từ ảnh
def extract_features(img_path):
    img_array = preprocess_image(img_path)  # Tiền xử lý ảnh
    features = feature_extractor.predict(img_array)  # Dự đoán đặc trưng
    features = features.flatten()  # Chuyển thành vector 1D
    return features

# Hàm so sánh độ tương đồng giữa các đặc trưng
def compare_features(input_features):
    similarities = {}
    for image_path, stored_features in image_features.items():
        stored_features = np.array(stored_features)  # Chuyển đặc trưng đã lưu thành numpy array
        similarity = cosine_similarity([input_features], [stored_features])[0][0]
        similarities[image_path] = similarity
    
    # In ra kết quả độ tương đồng
    print("Similarities:", similarities)

    # Sắp xếp các sản phẩm theo độ tương đồng giảm dần
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities

# Hàm để hiển thị ảnh
def show_image(img_path):
    try:
        img = Image.open(img_path)
        plt.figure()  # Tạo hình vẽ mới mỗi lần
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error displaying image {img_path}: {e}")

# Hàm in kết quả dự đoán
def print_predictions(predictions):
    class_labels = ['aobalo', 'aokhoac', 'aokieu', 'aolen', 'aoni', 'aosomi', 'aothun', 
                    'chanvaydai', 'chanvayngan', 'quandai', 'quanshort', 'vaydai', 'vayngan']
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_idx]
    print(f"Predicted Class: {predicted_class}")
    print(f"Prediction Probability: {predictions[0][predicted_class_idx]:.4f}")

# Hàm hiển thị các sản phẩm tương tự
def show_similar_products(img_path):
    input_features = extract_features(img_path)  # Trích xuất đặc trưng của ảnh thử nghiệm
    similar_products = compare_features(input_features)  # So sánh với các sản phẩm đã lưu

    if not similar_products:
        print("No similar products found.")
        return

    print("Top 5 similar products:")
    for i, (image_path, similarity) in enumerate(similar_products[:5]):
        print(f"{i+1}. Image: {image_path}, Similarity: {similarity:.4f}")
        show_image(image_path)  # Hiển thị hình ảnh sản phẩm tương tự

# Kiểm tra với một ảnh mẫu
if __name__ == '__main__':
    img_path = 'data/test_images/test_2.jpg'  # Cập nhật đường dẫn tới ảnh thử nghiệm
    print("Testing with image:", img_path)
    show_image(img_path)  # Hiển thị ảnh thử nghiệm
    predictions = model.predict(preprocess_image(img_path))  # Dự đoán từ mô hình
    print_predictions(predictions)  # In kết quả dự đoán

    # Hiển thị các sản phẩm tương tự
    show_similar_products(img_path)
