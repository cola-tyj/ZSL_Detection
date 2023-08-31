import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


xiushi_features = np.load("resnet101//defective_features//kailie.npy")
kailie_features = np.load("resnet101//defective_features//kailie.npy")
qipao_features = np.load("resnet101//defective_features//qipao.npy")
tuoluo_features = np.load("resnet101//defective_features//tuoluo.npy")


all_features = np.concatenate([xiushi_features, kailie_features,
                               qipao_features, tuoluo_features], axis=0)

num_samples = all_features.shape[0]
labels = np.array(["xiushi"] * xiushi_features.shape[0] +
                  ["kailie"] * kailie_features.shape[0] +
                  ["qipao"] * qipao_features.shape[0] +
                  ["tuoluo"] * tuoluo_features.shape[0])


perplexity = 1
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
embedded_features = tsne.fit_transform(all_features)


xiushi_embedded = embedded_features[:xiushi_features.shape[0]]
kailie_embedded = embedded_features[xiushi_features.shape[0]
    :xiushi_features.shape[0] + kailie_features.shape[0]]
qipao_embedded = embedded_features[xiushi_features.shape[0] + kailie_features.shape[0]:
                                   xiushi_features.shape[0] + kailie_features.shape[0] + qipao_features.shape[0]]
tuoluo_embedded = embedded_features[xiushi_features.shape[0] + kailie_features.shape[0] + qipao_features.shape[0]:
                                    xiushi_features.shape[0] + kailie_features.shape[0] + qipao_features.shape[0] + tuoluo_features.shape[0]]


plt.figure(figsize=(10, 8))
plt.scatter(xiushi_embedded[:, 0],
            xiushi_embedded[:, 1], c='blue', label='Rust')
plt.scatter(kailie_embedded[:, 0],
            kailie_embedded[:, 1], c='green', label='Crack')
plt.scatter(qipao_embedded[:, 0],
            qipao_embedded[:, 1], c='red', label='Bubbling')
plt.scatter(tuoluo_embedded[:, 0],
            tuoluo_embedded[:, 1], c='purple', label='Fall Off')
plt.xlabel('t-SNE dim 1')
plt.ylabel('t-SNE dim 2')
plt.title('t-SNE visualization of Defect Types')
plt.legend()
plt.show()
