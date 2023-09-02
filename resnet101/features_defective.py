import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import xml.etree.ElementTree as ET


resnet = models.resnet101(pretrained=True)


feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])


feature_extractor.eval()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


image_folder = "/root/autodl-tmp/ZSL_Detection/data/defective"
label_folder = "/root/autodl-tmp/ZSL_Detection/data/defective_label"
output_folder = "/root/autodl-tmp/ZSL_Detection/resnet101/defective_features"


def extract_features(image_path):
 
    img = Image.open(image_path).convert('RGB')

    img_tensor = transform(img)

    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        features = feature_extractor(img_tensor)

    feature_vector = torch.flatten(features, start_dim=1)

    return feature_vector.numpy()


def main():
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

                    defect_features = extract_features(image_path)

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


if __name__ == "__main__":
    main()
