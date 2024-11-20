from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io

app = Flask(__name__)

# Bật CORS cho toàn bộ ứng dụng Flask
CORS(app)

# Tải mô hình VGG16 đã huấn luyện
try:
    model = load_model('models/vgg16_product_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")

# Tạo model VGG16 để trích xuất đặc trưng (sử dụng VGG16 mà không có các lớp phân loại cuối cùng)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Định nghĩa các lớp sản phẩm
class_labels = ['aobalo', 'aokhoac', 'aokieu', 'aolen', 'aoni', 'aosomi', 'aothun', 
                'chanvaydai', 'chanvayngan', 'quandai', 'quanshort', 'vaydai', 'vayngan']

# Đọc các đặc trưng đã được lưu
FEATURES_FILE = 'image_features.pkl'
try:
    with open(FEATURES_FILE, 'rb') as f:
        image_features = pickle.load(f)
except Exception as e:
    print(f"Error loading image features: {e}")
    image_features = {}

# Hàm tiền xử lý hình ảnh
def preprocess_image(file):
    try:
        img = Image.open(io.BytesIO(file.read()))  # Đọc ảnh từ request
        img = img.resize((224, 224))  # Resize ảnh về kích thước 224x224
        img_array = np.array(img)  # Chuyển ảnh thành mảng numpy
        img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
        img_array = img_array / 255.0  # Chuẩn hóa ảnh
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

# Hàm trích xuất đặc trưng của hình ảnh đầu vào
def extract_features(img_array):
    try:
        features = feature_extractor.predict(img_array)
        features = features.flatten()  # Chuyển đặc trưng thành vector 1D
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        raise

# Hàm so sánh đặc trưng hình ảnh đầu vào với các đặc trưng đã lưu
def compare_features(input_features):
    similarities = {}
    
    # Đảm bảo input_features là mảng 2D (1 sample x n features)
    input_features = input_features.reshape(1, -1)  # Chuyển về mảng 2D (1, n)

    for image_path, stored_features in image_features.items():
        stored_features = np.array(stored_features).reshape(1, -1)  # Đảm bảo stored_features là 2D (1, n)
        
        similarity = cosine_similarity(input_features, stored_features)[0][0]
        similarities[image_path] = similarity
    
    # Sắp xếp các sản phẩm theo độ tương đồng giảm dần
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities

@app.route("/predict/", methods=["POST"])
def predict_image():
    try:
        # Kiểm tra xem có file hình ảnh không
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Tiền xử lý hình ảnh đầu vào
        img_array = preprocess_image(file)
        
        # Trích xuất đặc trưng của hình ảnh đầu vào
        input_features = extract_features(img_array)
        
        # So sánh đặc trưng và lấy các sản phẩm tương đồng
        similar_products = compare_features(input_features)
        
        # Trả về kết quả (danh sách các sản phẩm tương đồng)
        result = [{"product_name": image_path, "similarity": round(similarity, 4)} for image_path, similarity in similar_products[:5]]  # Trả về top 5 sản phẩm tương đồng
        
        return jsonify({"similar_products": result})
    
    except Exception as e:
        print(f"Error in predict_image: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
