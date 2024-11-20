from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Đường dẫn tới thư mục chứa dữ liệu train và validation
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
