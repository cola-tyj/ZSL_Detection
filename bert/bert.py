from transformers import BertTokenizer, BertModel
import torch
import json

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Attribute descriptions
attributes = [
    {
        "name": "rust",
        "description": (
            "生锈通常表现为金属表面颜色的变化，呈现出赤褐色、橙色、红色等与周围金属不同的色调。生锈区域的表面往往变得不平整，"
            "可能出现凹凸不平的斑块，与周围平滑的金属表面形成鲜明对比。生锈区域的纹理也会发生变化，可能呈现出不规则的颗粒状、裂缝或凹陷。"
            "生锈现象通常有薄覆盖层。这种覆盖层可能呈现出磨砂状、粗糙状甚至颗粒状的外观。生锈的特征在边界和边缘处尤为明显，"
            "可能在不同材料接触点处形成锈迹。"
        )
    },
    {
        "name": "bubbling",
        "description": (
            "起泡是一种常见的金属表面故障，通常表现为在金属表面出现气泡状的凸起。这些气泡可能具有不同大小和形状，"
            "使得金属表面的原本平滑和均匀的外观被打破。起泡区域的颜色往往会呈现出明显的变化，可能表现为光亮与暗淡之间的对比。"
            "起泡区能感觉到明显的凹凸不平的纹理，这与周围平滑的金属表面形成明显的对比。起泡可能导致金属表面的纹理发生变化，"
            "呈现出气泡状、裂缝或不规则的颗粒。"
        )
    },
    {
        "name": "crack",
        "description": (
            "金属表面的开裂现象通常表现为在金属材料中出现裂纹或裂纹网络。这些裂纹可能在不同方向上延伸，形成明显的线状或网状图案。"
            "开裂区域的颜色和纹理可能发生变化，与周围金属表面产生对比。裂纹可能呈现出不规则的形状，有时会伴随着细小的颗粒状结构。"
        )
    },
    {
        "name": "falloff",
        "description": (
            "金属表面的脱落现象通常表现为金属材料部分或整体脱离基础表面。脱落可能导致金属表面的区域变得凹凸不平，"
            "与周围平滑的表面形成明显的分界。脱落区域可能呈现出不同的颜色，由于暴露在空气中的时间较长，可能出现氧化和变色。"
            "脱落的边界可能会形成不规则的形状，表面可能呈现出颗粒状、粗糙状甚至裂纹。脱落区域可能伴随明显的不平整和松动。"
        )
    }
]

# Process each attribute description and save word vectors to JSON files
for i, attribute in enumerate(attributes):
    description = attribute["description"]
    tokens = tokenizer.tokenize(description)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    input_ids = torch.tensor([input_ids])
    attention_mask = torch.tensor([attention_mask])

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    word_vectors = outputs.last_hidden_state

    # Save word vectors to a JSON file
    file_name = f'word_vectors_{attribute["name"]}.json'
    word_vectors_json = word_vectors.tolist()
    with open(file_name, 'w') as f:
        json.dump(word_vectors_json, f)
