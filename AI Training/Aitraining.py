import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from model import get_model


model, criterion, optimizer = get_model(num_classes=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
val_dataset   = datasets.ImageFolder("dataset/val", transform=transform)
test_dataset  = datasets.ImageFolder("dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# training loop
def train(num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "skin_model.pth")
    print("Model saved as skin_model.pth")

# Evaluation
def evaluate():
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    print("Test Results:")
    print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

if __name__ == "__main__":
    train(num_epochs=5)
    evaluate()
