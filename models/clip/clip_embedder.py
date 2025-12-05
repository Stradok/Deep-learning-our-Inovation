import torch
from transformers import CLIPModel, CLIPProcessor

class CLIPEncoder:
    def __init__(self, device, model_name="openai/clip-vit-base-patch32"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_images(self, images):
        images = (images + 1)/2  # [-1,1] -> [0,1]
        inputs = {"pixel_values": images}
        with torch.no_grad():
            out = self.model.vision_model(**inputs)
            return out.pooler_output

    def encode_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            return self.model.get_text_features(**inputs)
