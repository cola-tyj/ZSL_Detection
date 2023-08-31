import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import xml.etree.ElementTree as ET


image_folder = "data//defective"
label_folder = "data//defective_label"
output_folder = "resnet101//defective_features"


model = models.resnet101(pretrained=True)
model.eval()


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
    CustomResize(target_size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

defect_features_dict = {}


for image_filename in os.listdir(image_folder):
    if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
        image_path = os.path.join(image_folder, image_filename)

        image = Image.open(image_path).convert("RGB")

        if image.size[0] < 7 or image.size[1] < 7:
            print(f"Skipped image: {image_filename} due to small size")
            continue

        try:

            label_filename = image_filename.replace(
                ".jpg", ".xml").replace(".png", ".xml")
            label_path = os.path.join(label_folder, label_filename)

            tree = ET.parse(label_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                defect_name = obj.find("name").text

                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                defect_image = image.crop((xmin, ymin, xmax, ymax))

                defect_image_tensor = preprocess(defect_image)
                defect_image_tensor = torch.unsqueeze(defect_image_tensor, 0)
                with torch.no_grad():
                    defect_features = model(defect_image_tensor).numpy()

                if defect_name not in defect_features_dict:
                    defect_features_dict[defect_name] = []
                defect_features_dict[defect_name].append(defect_features)

                print(
                    f"Processed defect: {defect_name} in image: {image_filename}")
        except RuntimeError as e:
            print(
                f"Error processing image: {image_filename}, skipping. Error message: {str(e)}")


for defect_name, features_list in defect_features_dict.items():
    output_path = os.path.join(output_folder, f"{defect_name}.npy")
    all_features = np.concatenate(features_list, axis=0)
    np.save(output_path, all_features)

print("All defect features have been saved.")
