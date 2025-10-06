import torch
import torch.nn as nn
import timm

def get_model(num_classes=2, lr=1e-4):
    # Efficentnet imported
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer
