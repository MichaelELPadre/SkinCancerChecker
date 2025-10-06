import torch
from torchvision import transforms
from PIL import Image
from model import get_model  # using model from training

# trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, _ = get_model(num_classes=2)
model.load_state_dict(torch.load("skin_model.pth", map_location=device))
model.eval()
model = model.to(device)

# Preprocess new image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # same size as training
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

img_path = "training.jpeg"
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)


with torch.no_grad():
    outputs = model(image)
    _, predicted = outputs.max(1)
    label = "benign" if predicted.item() == 0 else "malignant"

print(f"Prediction: {label}")
