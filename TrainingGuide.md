# CNN Training Guide — Benign vs Malignant Classifier

This guide explains how to train and test the **skin lesion classifier** built with PyTorch and EfficientNet-B0.

---


### Create a Virtual Environment
```
python3 -m venv .venv
source .venv/bin/activate
.\.venv\Scripts\activate 
```


---

### Install Dependencies
```
pip install -r requirements.txt
```


### Dataset Preparation

- Download images from ISIC Archive
- Filter to clinical overview images.
- Create metadata.csv with columns like:
    - image_id
    - diagnosis_1


- Python script to separate images:
```
import os, shutil, pandas as pd

df = pd.read_csv("metadata.csv")

for _, row in df.iterrows():
    img = row["image_id"] + ".jpg"
    label = row["diagnosis_1"]
    if label == "benign":
        shutil.copy(f"ISIC-images/{img}", f"dataset/train/benign/{img}")
    else:
        shutil.copy(f"ISIC-images/{img}", f"dataset/train/malignant/{img}")
```

### Define the Model (model.py)
```
import torch
import torch.nn as nn
import timm

model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### Train the Model (AItraining.py)
```
python3 AItraining.py
```

Loads dataset and trains for 5 epochs. Saves model as skin_model.pth

Output Example:
```
Epoch [1/5], Loss: 2.5902
Epoch [2/5], Loss: 0.4519
Training complete. Model saved to skin_model.pth
```

### Test Prediction (predict.py)
```
python3 predict.py
```

Example output:
```
Prediction: malignant
```