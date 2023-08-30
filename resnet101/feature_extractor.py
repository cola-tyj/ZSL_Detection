import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# 设置文件夹路径和模型
folder_path = "test_img//people"
model = models.resnet101(pretrained=True)
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 用于存储特征的列表
features_list = []

# 遍历文件夹中的图像文件
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)

        # 加载图像并进行预处理
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        # 使用ResNet-101提取特征
        with torch.no_grad():
            features = model(image_tensor)

        # 将特征转换为NumPy数组并添加到列表中
        features_np = features.numpy()
        features_list.append(features_np)

        print(f"Processed image: {filename}")

# 将特征列表转换为NumPy数组
all_features = np.concatenate(features_list, axis=0)

# 保存特征数组为NumPy文件
output_file_path = "resnet101//features.npy"
np.save(output_file_path, all_features)

print("特征保存完成")
