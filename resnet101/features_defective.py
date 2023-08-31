import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import xml.etree.ElementTree as ET

# 文件夹路径
image_folder = "data//defective"
label_folder = "data//defective_label"
output_folder = "resnet101//defective_features"

# 加载预训练的ResNet-101模型
model = models.resnet101(pretrained=True)
model.eval()

# 图像预处理


class CustomResize(object):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image):
        image_width, image_height = image.size
        new_short = self.target_size[0]
        if image_width > 0 and image_height > 0:
            new_long = int(new_short * max(image_width,
                           image_height) / min(image_width, image_height))
            return image.resize((new_long, new_short))
        else:
            return image


preprocess = transforms.Compose([
    CustomResize(target_size=(256, 256)),  # 调整大小时避免除以零
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 遍历图像文件夹和标注文件夹
for image_filename in os.listdir(image_folder):
    if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
        image_path = os.path.join(image_folder, image_filename)

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 加载对应的标注文件
        label_filename = image_filename.replace(
            ".jpg", ".xml").replace(".png", ".xml")
        label_path = os.path.join(label_folder, label_filename)

        # 解析标注文件，提取缺陷信息
        tree = ET.parse(label_path)
        root = tree.getroot()

        # 遍历标注的缺陷信息
        for obj in root.findall("object"):
            defect_name = obj.find("name").text

            # 根据缺陷的位置信息进行裁剪
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            defect_image = image.crop((xmin, ymin, xmax, ymax))

            # 进行预处理和特征提取
            defect_image_tensor = preprocess(defect_image)
            defect_image_tensor = torch.unsqueeze(defect_image_tensor, 0)
            with torch.no_grad():
                defect_features = model(defect_image_tensor)

            # 保存特征到对应的缺陷类型的npy文件
            output_path = os.path.join(output_folder, f"{defect_name}.npy")
            np.save(output_path, defect_features.numpy())

            print(
                f"Processed defect: {defect_name} in image: {image_filename}")
