import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载特征
xiushi_features = np.load("resnet101//defective_features//kailie.npy")
kailie_features = np.load("resnet101//defective_features//kailie.npy")
qipao_features = np.load("resnet101//defective_features//qipao.npy")
tuoluo_features = np.load("resnet101//defective_features//tuoluo.npy")

# 合并特征并生成标签
all_features = np.concatenate([xiushi_features, kailie_features,
                               qipao_features, tuoluo_features], axis=0)

num_samples = all_features.shape[0]
labels = np.array(["xiushi"] * xiushi_features.shape[0] +
                  ["kailie"] * kailie_features.shape[0] +
                  ["qipao"] * qipao_features.shape[0] +
                  ["tuoluo"] * tuoluo_features.shape[0])

# 使用t-SNE进行降维
perplexity = 1
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
embedded_features = tsne.fit_transform(all_features)

# 提取各类特征对应的降维结果
xiushi_embedded = embedded_features[:xiushi_features.shape[0]]
kailie_embedded = embedded_features[xiushi_features.shape[0]:xiushi_features.shape[0] + kailie_features.shape[0]]
qipao_embedded = embedded_features[xiushi_features.shape[0] + kailie_features.shape[0]:
                                   xiushi_features.shape[0] + kailie_features.shape[0] + qipao_features.shape[0]]
tuoluo_embedded = embedded_features[xiushi_features.shape[0] + kailie_features.shape[0] + qipao_features.shape[0]:
                                    xiushi_features.shape[0] + kailie_features.shape[0] + qipao_features.shape[0] + tuoluo_features.shape[0]]

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(xiushi_embedded[:, 0], xiushi_embedded[:, 1], c='blue', label='xiushi')
plt.scatter(kailie_embedded[:, 0], kailie_embedded[:, 1], c='green', label='kailie')
plt.scatter(qipao_embedded[:, 0], qipao_embedded[:, 1], c='red', label='qipao')
plt.scatter(tuoluo_embedded[:, 0], tuoluo_embedded[:, 1], c='purple', label='tuoluo')
plt.xlabel('t-SNE dim 1')
plt.ylabel('t-SNE dim 2')
plt.title('t-SNE visualization of Defect Types')
plt.legend()
plt.show()
