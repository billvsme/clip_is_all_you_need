Clip Is All You Need
=========
# 模型
```
vit -> clip -> vilt -> blip -> stable diffusion
```
任务|模型|Colab
---|---|---
图片分类|vit|
文本->图片搜索|clip|
图片->图片搜索|clip|
图片->图片描述生成|blip|
文本->图片生成|stable diffusion|

## vit
### Paper
[https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

### 结构
<a href="https://sm.ms/image/3RMuvNXxJgDOP25" target="_blank"><img src="https://s2.loli.net/2023/12/24/3RMuvNXxJgDOP25.png" width="60%"></a>
### 模型
[https://huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)

### 使用
```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

## clip

## vilt

## blip

