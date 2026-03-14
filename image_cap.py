from PIL import Image 
import requests 
from transformers import AutoProcessor, BlipForConditionalGeneration 


processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_path = "Imagee.jpg"
image = Image.open(image_path,).convert("RGB")
text = "the image of"

inputs = processor(images = image , text = text , return_tensors='pt')
outputs = model.generate(**inputs, max_length = 50)

caption = processor.decode(outputs[0], skip_special_tokens = True)
print(caption)


