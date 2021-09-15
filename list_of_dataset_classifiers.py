li1=[["cloudy","clear sky"],["rainy","not rainy"],["sunny","not sunny"],["snowy","not snowy"],["foggy","not foggy"]]
text_feat=[]
for j in range(0,5):
  text_inputs = torch.cat([clip.tokenize(f"a photo of a {c} day") for c in li1[j]]).to(device)
  with torch.no_grad():
      text_features = model.encode_text(text_inputs)
      
  text_features /= text_features.norm(dim=-1, keepdim=True)
  text_feat.append(text_features.T)
  
weather_li=[]
weather_percent=[]
start=time.time()
for i in range(1,67):
        weather_li=[]
        weather_percent=[]
        img = Image.open('/img ({}).jpg'.format(i))
        display(img.resize((500,500)))
        image_input = preprocess(img).unsqueeze(0).to(device)
        for j in range(0,5):
            with torch.no_grad():
                image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_feat[j]).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            for value, index in zip(values, indices):
                weather_li.append(1) if index==1 else weather_li.append(0)
                weather_percent.append(value.item())
        
        for a in range(0,5):
            print(f"{li1[a][weather_li[a]]:>16s}: {100 * weather_percent[a]:.2f}%")
            
end=time.time()
print("average time for execution of one image is {}".format((end-start)/66))
