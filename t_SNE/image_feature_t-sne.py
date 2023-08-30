import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


people_features = np.load("resnet101//people_features.npy")
scene_features = np.load("resnet101//scene_features.npy")

# 合并两个特征数组
all_features = np.concatenate([people_features, scene_features], axis=0)

# 创建对应于特征类别的标签
num_people_samples = people_features.shape[0]
num_scene_samples = scene_features.shape[0]
labels = np.array(["people features"] * num_people_samples +
                  ["scene features"] * num_scene_samples)

# 使用 t-SNE 进行降维和可视化
tsne = TSNE(n_components=2, perplexity=1, random_state=42)
embedded_features = tsne.fit_transform(all_features)

# 根据标签分割嵌入特征
people_embedded = embedded_features[:num_people_samples]
scene_embedded = embedded_features[num_people_samples:]

plt.figure(figsize=(10, 8))
plt.scatter(people_embedded[:, 0],
            people_embedded[:, 1], c='blue', label='people')
plt.scatter(scene_embedded[:, 0],
            scene_embedded[:, 1], c='green', label='scene')
plt.xlabel('t-SNE dim 1')
plt.ylabel('t-SNE dim2')
plt.title('t-SNE visualization')
plt.legend()
plt.show()
