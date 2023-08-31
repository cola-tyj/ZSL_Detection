import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

folder_path = "data//no_defective"
model = models.resnet101(pretrained=True)
model.eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


features_list = []

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)

        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        with torch.no_grad():
            features = model(image_tensor)

        features_np = features.numpy()
        features_list.append(features_np)

        print(f"Processed image: {filename}")


all_features = np.concatenate(features_list, axis=0)


output_file_path = "resnet101//no_defective_features.npy"
np.save(output_file_path, all_features)
