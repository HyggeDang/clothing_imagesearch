from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import pickle
import sys
import os

# Đường dẫn tới dữ liệu train và validation
train_dir = 'data/train'
validation_dir = 'data/validation'

# Kích thước ảnh đầu vào cho VGG16
image_size = (224, 224)

# Tạo ImageDataGenerator cho huấn luyện và kiểm tra
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Chuẩn hóa ảnh (chia giá trị pixel cho 255)
    rotation_range=40,  # Xoay ảnh ngẫu nhiên
    width_shift_range=0.2,  # Dịch chuyển ngang ngẫu nhiên
    height_shift_range=0.2,  # Dịch chuyển dọc ngẫu nhiên
    shear_range=0.2,  # Biến dạng
    zoom_range=0.2,  # Phóng to/thu nhỏ ngẫu nhiên
    horizontal_flip=True,  # Lật ngang ảnh
    fill_mode='nearest'  # Điền các pixel thiếu
)

validation_datagen = ImageDataGenerator(rescale=1./255)  # Chỉ chuẩn hóa cho validation

# Tạo generator cho huấn luyện và kiểm tra
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Đường dẫn tới dữ liệu huấn luyện
    target_size=image_size,  # Thay đổi kích thước ảnh
    batch_size=32,  # Kích thước batch
    class_mode='categorical'  # Phân loại nhiều lớp
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,  # Đường dẫn tới dữ liệu kiểm tra
    target_size=image_size,  # Thay đổi kích thước ảnh
    batch_size=32,  # Kích thước batch
    class_mode='categorical'  # Phân loại nhiều lớp
)

# Tạo mô hình VGG16 với các lớp phân loại cuối
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Đóng băng các lớp của mô hình VGG16
base_model.trainable = False

# Đường dẫn file pickle để lưu đặc trưng
FEATURES_FILE = 'image_features.pkl'

# Kiểm tra xem file pickle đã tồn tại chưa
if not os.path.exists(FEATURES_FILE):
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump({}, f)  # Khởi tạo file pickle với một dict trống

# Hàm trích xuất đặc trưng và lưu vào pickle
def extract_and_save_features(generator, model, features_file):
    # Lấy đặc trưng từ tất cả các batch trong generator
    features_data = {}
    for inputs, labels in generator:
        # Trích xuất đặc trưng từ các lớp của base_model
        features = model.predict(inputs)
        features = features.flatten()  # Chuyển đổi thành vector 1D
        
        # Lấy đường dẫn ảnh từ generator
        image_paths = generator.filepaths
        
        # Lưu đặc trưng và đường dẫn ảnh vào dictionary
        for i, image_path in enumerate(image_paths):
            features_data[image_path] = features[i].tolist()  # Lưu đặc trưng dưới dạng list để dễ lưu vào pickle

        # Nếu tất cả các batch đã được xử lý, thoát khỏi vòng lặp
        if generator.batch_index == 0:  # Khi hoàn thành 1 vòng lặp (epoch)
            break

    # Lưu đặc trưng vào file pickle
    if os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            data = pickle.load(f)  # Đọc dữ liệu cũ từ file pickle
    else:
        data = {}  # Khởi tạo data nếu file pickle chưa tồn tại
    
    # Thêm dữ liệu mới vào dictionary
    data.update(features_data)
    
    # Lưu lại vào file pickle
    with open(features_file, 'wb') as f:
        pickle.dump(data, f)  # Lưu lại dictionary vào file pickle

# Trích xuất đặc trưng cho dữ liệu huấn luyện
extract_and_save_features(train_generator, base_model, FEATURES_FILE)
# Trích xuất đặc trưng cho dữ liệu kiểm tra
extract_and_save_features(validation_generator, base_model, FEATURES_FILE)

# Xây dựng mô hình
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Dropout giúp giảm overfitting
    layers.Dense(train_generator.num_classes, activation='softmax')  # Số lớp bằng số lớp trong dữ liệu của bạn
])

# Biên dịch mô hình
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Lưu mô hình đã huấn luyện
model.save('vgg16_product_model.h5')

print("Mô hình đã được huấn luyện và lưu thành công!")
# In ra kết quả của quá trình huấn luyện
print("Training History:")
print(history.history)

