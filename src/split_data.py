import os
import shutil
import random

def split_data(data_dir, train_dir, val_dir, val_split=0.2):
    # Kiểm tra xem thư mục train và val có tồn tại chưa, nếu chưa thì tạo
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Duyệt qua tất cả các thư mục con trong thư mục data_dir (từng lớp)
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        # Nếu là thư mục con, tiếp tục
        if os.path.isdir(class_path):
            # Tạo thư mục train và val cho lớp này
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)

            if not os.path.exists(train_class_dir):
                os.makedirs(train_class_dir)
            if not os.path.exists(val_class_dir):
                os.makedirs(val_class_dir)

            # Lấy danh sách tất cả các ảnh trong lớp này
            all_images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

            # Tính toán số lượng ảnh cho validation (theo tỉ lệ val_split)
            val_size = int(len(all_images) * val_split)
            
            # Chia ngẫu nhiên danh sách ảnh thành train và validation
            val_images = random.sample(all_images, val_size)
            train_images = [img for img in all_images if img not in val_images]

            # Di chuyển các ảnh vào thư mục train và val tương ứng
            for img in train_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(train_class_dir, img)
                shutil.copy(src, dst)

            for img in val_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(val_class_dir, img)
                shutil.copy(src, dst)

            print(f"Class '{class_name}': {len(train_images)} images for training, {len(val_images)} images for validation")

# Đường dẫn thư mục gốc chứa dữ liệu
data_dir = 'data/images'  # Thay bằng đường dẫn thực tế tới dữ liệu của bạn

# Đường dẫn tới thư mục train và val sẽ được tạo ra
train_dir = 'data/train'
val_dir = 'data/val'

# Chia dữ liệu
split_data(data_dir, train_dir, val_dir, val_split=0.2)  # 80% train, 20% val
