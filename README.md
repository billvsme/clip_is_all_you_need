CLIP Is All You Need
=========
# 模型
```
ViT -> CLIP -> vilt -> blip -> stable diffusion
```
任务|模型|Colab
---|---|---
图片分类|ViT|[![Open Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/billvsme/clip_is_all_you_need/blob/master/jupyter/vit.ipynb)  
文本->图片搜索|CLIP|
图片->图片搜索|CLIP|
图片->图片描述生成|blip|
文本->图片生成|stable diffusion|

## ViT
### Paper
[https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

### 结构
<a href="https://sm.ms/image/3RMuvNXxJgDOP25" target="_blank"><img src="https://s2.loli.net/2023/12/24/3RMuvNXxJgDOP25.png" width="70%"></a>
### 模型
[https://huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)

### 使用
下载模型
```shell
mkdir -p /content/models
git clone https://huggingface.co/google/vit-base-patch16-224 /content/models/vit-base-patch16-224
```
ImageNet分类任务demo
```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('/content/models/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('/content/models/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# ImageNet分类
predicted_class_idx = logits.argmax(-1).item()
print("分类:", model.config.id2label[predicted_class_idx])
```

## CLIP
### Paper
[https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
### 结构
<a href="https://sm.ms/image/PWbqDaCXK9cJIYo" target="_blank"><img src="https://s2.loli.net/2023/12/24/PWbqDaCXK9cJIYo.png" width="70%"></a>
### 模型
[https://huggingface.co/openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)

## vilt

## blip

