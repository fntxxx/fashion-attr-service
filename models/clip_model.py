import torch
import open_clip
from PIL import Image

device = "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(device)
model.eval()


def predict(image: Image.Image, labels):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text = tokenizer(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(1)
    return labels[indices[0]], values[0].item()