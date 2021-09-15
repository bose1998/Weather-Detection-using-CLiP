import clip
import os
import torch
import time
#from torchvision.datasets import CIFAR100
from tqdm import tqdm
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

li1=["sunny and clear sky","clear sky","rainy and cloudy","snowy and cloudy","cloudy","foggy","stormy","snowy and clear sky"]  ## this is the linear dataset that we will use.

text_inputs = torch.cat([clip.tokenize(f"a photo of a {c} day") for c in li1]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)

start=time.time()
for i in range(1,67):
    weather_li=[]
    weather_percent=[]
        
    img = Image.open('images/img ({}).jpg'.format(i))
        
    display(img.resize((500,500)))
    image_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        #text_features = model.encode_text(text_inputs)
    #image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(2)
    for value, index in zip(values, indices):
        print(f"{li1[index]:>16s}: {100 * value.item():.2f}%")
        
end=time.time()
print("average time for execution of one image is {}".format((end-start)/66))
