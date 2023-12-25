CLIP Is All You Need
=========
多模态检索、生成。提供示例代码，一步一步理解从transformer到stable diffusion的发展。
```
ViT -> CLIP -> ViLT -> BLIP -> stable diffusion
        \
         \-> Qwen-VL
          \
           \-> CogVLM
```
任务|模型|Colab
---|---|---
图片分类|ViT|[![Open Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/billvsme/clip_is_all_you_need/blob/master/jupyter/vit.ipynb)  
文本/图片搜索|CLIP|[![Open Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/billvsme/clip_is_all_you_need/blob/master/jupyter/clip.ipynb)  
图片描述生成|BLIP|[![Open Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/billvsme/clip_is_all_you_need/blob/master/jupyter/blip.ipynb)  
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
<a href="https://sm.ms/image/PWbqDaCXK9cJIYo" target="_blank"><img src="https://s2.loli.net/2023/12/24/PWbqDaCXK9cJIYo.png" width="60%"></a>
<a href="https://sm.ms/image/hVJOSd43FDZaieR" target="_blank"><img src="https://s2.loli.net/2023/12/25/hVJOSd43FDZaieR.png" width="30%"></a>

### 模型
[https://huggingface.co/openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)

### 使用
安装依赖
```shell
pip install chromadb
pip install diffusers
```
下载模型
```shell
mkdir -p /content/models
git clone --depth=1 https://huggingface.co/openai/clip-vit-large-patch14  /content/models/clip-vit-large-patch14
git clone --depth=1 https://huggingface.co/stabilityai/sdxl-turbo  /content/models/sdxl-turbo
```
先用stable diffusion生成一些用于分类的图片，使用sdxl-turbo快速生成，只需要1-4个step⚡️
```python
import os
import torch
from diffusers import DiffusionPipeline

prompts = [
    ("cat", "a photo of cat"),
    ("dog", "a photo of dog"),
    ("pig", "a photo of pig"),
    ("chair", "a photo of chair"),
    ("table", "a photo of dining table")
]

model_id = "/content/models/sdxl-turbo"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

for category, prompt in prompts:
    os.makedirs(f"output/{category}", exist_ok=True)
    for index in range(2):
        image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        image.save(f"output/{category}/{category}-{index}.png")
```
初始化Chroma，注意相似度计算，按照CLIP的实现，需要使用点积，
```python
import chromadb

client = chromadb.PersistentClient(path="/content/chroma/my")
collection = client.get_or_create_collection(
    name="my_collection",
    metadata={"hnsw:space": "ip"}
)
```
图片向量存入Chroma，注意修改图片分辨率到对于VIT模型的分辨率，保证postion没有问题
```python
import os
from PIL import Image
import requests
import hashlib

from transformers import CLIPProcessor, CLIPModel

model_id = "/content/models/clip-vit-large-patch14"

def all_images():
    for root, ds, fs in os.walk("output"):
        for f in fs:
            if f.endswith('.png'):
                yield os.path.join(root, f)

model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

for image_path in all_images():
    image = Image.open(image_path)
    inputs = processor(images=image.resize((224, 224)), return_tensors="pt")
    image_feature = model.get_image_features(**inputs)[0]

    id_ = hashlib.md5(image.tobytes()).hexdigest()
    collection.add(
        embeddings=[image_feature.tolist()],
        metadatas=[{"source": image_path, "category": image_path.split("/")[1]}],
        ids=[id_]
    )
    
    print(image_path, id_)
```
文本/图片检索
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)


# 文本 -> 图片检索
queries = [
    "a photo of cat",
    "a photo of dog",
    "a photo of pig",
    "a photo of chair",
    "a photo of dining table"
]

print("="*20)
print(f"文本 -> 图片检索")
for query in queries:
    inputs = tokenizer([query], padding=True, return_tensors="pt")
    text_feature = model.get_text_features(**inputs)[0]
    result = collection.query(
        query_embeddings=[text_feature.tolist()],
        n_results=2
    )
    print(f"检索：{query}")
    for metadata in result["metadatas"][0]:
        print("image_path:", metadata["source"])

# 图片->图片检索
images = [
    "output/cat/cat-0.png",
    "output/dog/dog-0.png",
    "output/pig/pig-0.png",
    "output/chair/chair-0.png",
    "output/table/table-0.png"
]

print("="*20)
print(f"图片 -> 图片检索")
for image_path in images:
    image = Image.open(image_path)
    inputs = processor(images=image.resize((224, 224)), return_tensors="pt")
    image_feature = model.get_image_features(**inputs)[0]

    result = collection.query(
        query_embeddings=[image_feature.tolist()],
        n_results=2
    )

    print(f"检索：{image_path}")
    for metadata in result["metadatas"][0]:
        print("image_path:", metadata["source"])
```
可以看到，文本->图片，图片->图片 检索结果正确  
<a href="https://sm.ms/image/vcBWaGVjMe5u6sF" target="_blank"><img src="https://s2.loli.net/2023/12/25/vcBWaGVjMe5u6sF.png" width="30%"></a>

## ViLT

## BLIP
### Paper
[https://arxiv.org/pdf/2201.12086.pdf](https://arxiv.org/pdf/2201.12086.pdf)


### 结构
<a href="https://sm.ms/image/haKLzrHEOGslB1u" target="_blank"><img src="https://s2.loli.net/2023/12/25/haKLzrHEOGslB1u.png" width="70%"></a>  
  
<a href="https://sm.ms/image/PqLzRkvgYnwKFSx" target="_blank"><img src="https://s2.loli.net/2023/12/25/PqLzRkvgYnwKFSx.png" width="70%"></a>  

### 模型
[https://huggingface.co/Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large)

### 使用
下载模型
```shell
mkdir -p /content/modelsw
git clone --depth=1 https://huggingface.co/Salesforce/blip-image-captioning-large  /content/models/blip-image-captioning-large
```
生成图像描述
```python
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

model_id = "/content/models/blip-image-captioning-large/"

processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id)
model.to("cuda")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
# 有提示
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print("="*20)
print("描述：", processor.decode(out[0], skip_special_tokens=True))

# 没有提示
inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print("="*20)
print("描述：", processor.decode(out[0], skip_special_tokens=True))
```
