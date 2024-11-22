import numpy as np
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import pickle
from sklearn.preprocessing import normalize

# Tải mô hình VGG16 đã được huấn luyện sẵn, không cần phân loại đầu ra
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

# Hàm trích xuất đặc trưng cho một batch ảnh
def extract_features_batch(image_paths):
    images = []
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        images.append(img_array)
    
    images = np.array(images)
    images = preprocess_input(images)
    features = model.predict(images)
    
    # Làm phẳng đặc trưng cho mỗi ảnh
    flattened_features = features.reshape(features.shape[0], -1)  # Chuyển từ (batch_size, h, w, c) thành (batch_size, h*w*c)
    
    return flattened_features

# Đọc ảnh và trích xuất đặc trưng
def process_images(image_dir):
    features_list = []
    image_paths = []
    
    # Kiểm tra các thư mục con
    for subdir in os.listdir(image_dir):
        subdir_path = os.path.join(image_dir, subdir)
        if os.path.isdir(subdir_path):
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                if file_path.endswith('.jpg') or file_path.endswith('.png'):
                    image_paths.append(file_path)
    
    # Kiểm tra nếu không có ảnh nào được tìm thấy
    if len(image_paths) == 0:
        print(f"No images found in {image_dir}. Please check the folder.")
        return None, None

    # Xử lý ảnh theo batch
    batch_size = 32
    all_features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_features = extract_features_batch(batch_paths)
        
        # Kiểm tra nếu batch_features không rỗng
        if batch_features is not None and len(batch_features) > 0:
            all_features.append(batch_features)
        else:
            print(f"Warning: Empty feature batch encountered for {batch_paths}")
    
    # Kiểm tra nếu all_features không có dữ liệu
    if len(all_features) == 0:
        print("No features were extracted. Please check the image files.")
        return None, None
    
    features = np.vstack(all_features)
    return features, image_paths

# Đọc và trích xuất đặc trưng cho toàn bộ bộ dữ liệu
image_dir = 'data/images'
features, paths = process_images(image_dir)

# Kiểm tra nếu không có đặc trưng nào được trích xuất
if features is None or paths is None:
    print("Exiting due to errors in feature extraction.")
else:
    # Chuẩn hóa vector đặc trưng theo độ chuẩn norm
    normalized_features = normalize(features)

    # Lưu đặc trưng và đường dẫn ảnh vào 2 file pickle riêng biệt
    if not os.path.exists('features'):
        os.makedirs('features')

    with open('features/vectors1.pkl', 'wb') as f:
        pickle.dump(normalized_features, f)

    with open('features/paths1.pkl', 'wb') as f:
        pickle.dump(paths, f)

    print("Feature extraction and saving completed successfully.")
