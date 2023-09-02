import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load pre-trained ResNet-101 model
resnet = models.resnet101(pretrained=True)

# Remove the final fully connected layer
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# Set model to evaluation mode
feature_extractor.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load image folder
folder_path = "/root/autodl-tmp/ZSL_Detection/data/no_defective"
output_folder = "/root/autodl-tmp/ZSL_Detection/resnet101/no_defective_features"

def extract_features(image_path):
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Apply transforms to image
    img_tensor = transform(img)

    # Add batch dimension to tensor
    img_tensor = img_tensor.unsqueeze(0)

    # Pass image through ResNet-101 model
    with torch.no_grad():
        features = feature_extractor(img_tensor)

    # Flatten the features into a 1D vector
    feature_vector = torch.flatten(features, start_dim=1)

    return feature_vector.numpy()

def main():
    features_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)

            feature_vector = extract_features(image_path)
            features_list.append(feature_vector)

            print(f"Processed image: {filename}")

    all_features = np.concatenate(features_list, axis=0)

    output_file_path = os.path.join(output_folder, "no_defective_features.npy")
    np.save(output_file_path, all_features)

    print("Features for non-defective images have been saved.")

if __name__ == "__main__":
    main()
