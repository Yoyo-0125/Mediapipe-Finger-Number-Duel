import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms

# 保存增强后的图片目录
save_dir = 'transformed_data2'
os.makedirs(save_dir, exist_ok=True)

# 定义数据增强变换
data_transforms = transforms.Compose([
    transforms.RandomRotation(20),                    # 旋转
    transforms.RandomResizedCrop(size=(640, 640), scale=(0.85, 1.15)),  # 缩放+裁剪
    transforms.RandomHorizontalFlip(),                # 水平翻转
    transforms.RandomAffine(degrees=0, shear=20),    # 剪切
])

# 读取原始图片目录
img_dir = "Chinese-number-gestures-recognition"
dirs = os.listdir(img_dir)
print(f"Found {len(dirs)} images.")

cnt = [0] * 10
for filename in dirs:
    prefix = filename.split('.')[0]
    img_path = os.path.join(img_dir, filename)
    img = Image.open(img_path).convert('RGB')  # 转为RGB
    
    for i in range(2):
        cnt[int(prefix[0])] += 1
        img_aug = data_transforms(img)
        save_path = os.path.join(save_dir, f"{prefix[0]}_{cnt[int(prefix[0])]}.jpg")
        img_aug.save(save_path)

    print(f"Processed {prefix}")
