import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x_hidden = self.fc3(x)
        x = x_hidden
        return x, x_hidden


model = CustomNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


defective_classes = ["tuoluo", "xiushi", "qipao", "kailie"]
no_defective_class = "no_defective"
data_folder = "/root/autodl-tmp/ZSL_Detection/resnet101/defective_features"
no_defective_folder = "/root/autodl-tmp/ZSL_Detection/resnet101/no_defective_features"


features = []
labels = []

for defect_name in defective_classes:
    file_path = os.path.join(data_folder, f"{defect_name}.npy")
    defect_features = np.load(file_path)
    num_samples = defect_features.shape[0]
    features.append(defect_features)
    labels.extend([defect_name] * num_samples)


no_defective_file_path = os.path.join(
    no_defective_folder, "no_defective_features.npy")
no_defective_features = np.load(no_defective_file_path)
num_no_defective_samples = no_defective_features.shape[0]
features.append(no_defective_features)
labels.extend([no_defective_class] * num_no_defective_samples)


features = torch.tensor(np.concatenate(features), dtype=torch.float32)


label_to_index = {label: index for index, label in enumerate(set(labels))}
labels = torch.tensor([label_to_index[label]
                      for label in labels], dtype=torch.long)


model.train()

epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs, hidden_outputs = model(features)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")


output_folder = "/root/autodl-tmp/ZSL_Detection/classify/feature_5"
os.makedirs(output_folder, exist_ok=True)

for defect_name in defective_classes:
    hidden_output = hidden_outputs[labels ==
                                   label_to_index[defect_name]].detach().numpy()
    np.save(os.path.join(output_folder,
            f"{defect_name}_hidden.npy"), hidden_output)


hidden_output = hidden_outputs[labels ==
                               label_to_index[no_defective_class]].detach().numpy()
np.save(os.path.join(output_folder,
        f"{no_defective_class}_hidden.npy"), hidden_output)
