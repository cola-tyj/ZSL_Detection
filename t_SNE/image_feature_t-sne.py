import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


xiushi_features = np.load("resnet101//defective_features//xiushi.npy")
kailie_features = np.load("resnet101//defective_features//kailie.npy")
qipao_features = np.load("resnet101//defective_features//qipao.npy")
tuoluo_features = np.load("resnet101//defective_features//tuoluo.npy")
no_defective_features = np.load("resnet101//no_defective_features.npy")


all_features = np.concatenate([xiushi_features, kailie_features,
                               qipao_features, tuoluo_features,
                               no_defective_features], axis=0)

num_samples = all_features.shape[0]
labels = np.array(["xiushi"] * xiushi_features.shape[0] +
                  ["kailie"] * kailie_features.shape[0] +
                  ["qipao"] * qipao_features.shape[0] +
                  ["tuoluo"] * tuoluo_features.shape[0] +
                  ["no_defective"] * no_defective_features.shape[0])


# perplexity = 1
tsne = TSNE(n_components=2, random_state=42)
embedded_features = tsne.fit_transform(all_features)


xiushi_embedded = embedded_features[:xiushi_features.shape[0]]
kailie_embedded = embedded_features[xiushi_features.shape[0]:xiushi_features.shape[0] + kailie_features.shape[0]]
qipao_embedded = embedded_features[xiushi_features.shape[0] + kailie_features.shape[0]:
                                   xiushi_features.shape[0] + kailie_features.shape[0] + qipao_features.shape[0]]
tuoluo_embedded = embedded_features[xiushi_features.shape[0] + kailie_features.shape[0] + qipao_features.shape[0]:
                                    xiushi_features.shape[0] + kailie_features.shape[0] + qipao_features.shape[0] + tuoluo_features.shape[0]]
no_defective_embedded = embedded_features[-no_defective_features.shape[0]:]


output_dir = "t_SNE//t-sne__result"
os.makedirs(output_dir, exist_ok=True)


def visualize_and_save(embedded_features, color, label, title, filename):
    plt.figure(figsize=(10, 8))
    plt.scatter(embedded_features[:, 0],
                embedded_features[:, 1], c=color, label=label)
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


visualize_and_save(xiushi_embedded, 'blue', 'Rust',
                   't-SNE visualization of Rust Defects', 'rust_t_sne.png')
visualize_and_save(kailie_embedded, 'green', 'Crack',
                   't-SNE visualization of Crack Defects', 'crack_t_sne.png')
visualize_and_save(qipao_embedded, 'red', 'Bubbling',
                   't-SNE visualization of Bubbling Defects', 'bubbling_t_sne.png')
visualize_and_save(tuoluo_embedded, 'purple', 'Fall Off',
                   't-SNE visualization of Fall Off Defects', 'fall_off_t_sne.png')
visualize_and_save(no_defective_embedded, 'orange', 'No Defective',
                   't-SNE visualization of No Defects', 'no_defective_t_sne.png')


plt.figure(figsize=(10, 8))
plt.scatter(xiushi_embedded[:, 0],
            xiushi_embedded[:, 1], c='blue', label='Rust')
plt.scatter(kailie_embedded[:, 0],
            kailie_embedded[:, 1], c='green', label='Crack')
plt.scatter(qipao_embedded[:, 0],
            qipao_embedded[:, 1], c='red', label='Bubbling')
plt.scatter(tuoluo_embedded[:, 0],
            tuoluo_embedded[:, 1], c='purple', label='Fall Off')
plt.scatter(no_defective_embedded[:, 0],
            no_defective_embedded[:, 1], c='orange', label='No Defective')
plt.xlabel('t-SNE dim 1')
plt.ylabel('t-SNE dim 2')
plt.title('t-SNE visualization of Defect Types')
plt.legend()
plt.savefig(os.path.join(output_dir, 'all_defects_t_sne.png'))
plt.close()
