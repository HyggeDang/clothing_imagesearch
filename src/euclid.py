import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# Tải mô hình VGG16 đã được huấn luyện sẵn, không cần phân loại đầu ra
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

# Hàm trích xuất đặc trưng của ảnh
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Hàm tìm kiếm ảnh tương tự dựa trên khoảng cách Euclid
def find_similar_images(query_image, top_k=5):
    # Trích xuất đặc trưng của ảnh truy vấn
    query_features = extract_features(query_image)
    query_features = query_features / np.linalg.norm(query_features)  # Chuẩn hóa

    # Tải các đặc trưng đã lưu từ pickle
    with open('features/vectors1.pkl', 'rb') as f:
        features = pickle.load(f)
        
    with open('features/paths1.pkl', 'rb') as f:
        paths = pickle.load(f)
    
    # Tính toán khoảng cách Euclid giữa ảnh truy vấn và tất cả ảnh trong tập dữ liệu
    distances = euclidean_distances([query_features], features)[0]
    
    # Lấy top 5 ảnh tương tự nhất
    top_k_indices = np.argsort(distances)[:top_k]
    similar_images = [(paths[i], distances[i]) for i in top_k_indices]
    
    return similar_images

# Test hàm tìm kiếm với một ảnh truy vấn
query_image = 'data/test_images/test_33.jpg'  # Thay thế với đường dẫn ảnh truy vấn
similar_images = find_similar_images(query_image)

# In kết quả
for img, dist in similar_images:
    print(f"Image: {img}, Distance: {dist}")

# Lưu mô hình đã huấn luyện (VGG16 là mô hình tiền huấn luyện)
base_model.save('models/vgg16_model.h5')
