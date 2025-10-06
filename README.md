# Michael’s Benign vs Malignant Skin Lesion Classifier

This project focuses on building a **Convolutional Neural Network (CNN)** to classify skin lesion images as **benign (non-cancerous)** or **malignant (cancerous)** using **PyTorch** and the **ISIC dataset**.  
It represents the first stage of a larger medical web platform that later integrates with a **Skin Cancer Type Detector** and **AI Nutritionist Assistant**.



## Project Overview

1. **Virtual Environment Setup**
   - Python 3.10 / 3.11 used inside a virtual environment (`.venv`).

2. **Dataset Preparation**
   - Filtered ISIC dataset to include only **clinical overview** images.
   - About **400 images** split evenly between *benign* and *malignant*.
   - Metadata stored in `metadata.csv`.
   - Used **Pandas** to:
     - Read and filter metadata.
     - Copy each image into separate folders:  
       `dataset/train/benign` and `dataset/train/malignant`.

3. **Model Creation**
   - Used **EfficientNet-B0** CNN architecture from the `timm` library.
   - Defined in `model.py` with:
     - `CrossEntropyLoss()` for classification.
     - `Adam` optimizer with learning rate = `1e-4`.

4. **Model Training**
   - Implemented in `AItraining.py` using **PyTorch**.
   - Steps:
     - Load and preprocess data (resize, normalize, convert to tensors).
     - Forward pass → Predict.
     - Compare with true labels → Compute loss.
     - Backpropagate → Update model weights.
   - After several **epochs**, the model’s loss decreased steadily.
   - Saved trained model weights as `skin_model.pth`.

5. **Prediction Script**
   - `predict.py` allows testing the trained model with a new image **outside the dataset**.
   - The model loads the saved weights and predicts:
     ```
     Prediction: benign
     or
     Prediction: malignant
     ```



## Technologies Used

### Core AI & Training
| Library | Purpose |
|----------|----------|
| **torch (≥2.0.0)** | Core deep learning framework. Handles tensor operations, automatic differentiation, and model training. |
| **torchvision (≥0.15.0)** | Image preprocessing utilities (resize, normalize, augment). Simplifies dataset handling for PyTorch. |
| **timm (≥0.9.0)** | Provides pretrained models (like EfficientNet-B0). Saves time by leveraging ImageNet knowledge for transfer learning. |
| **Pandas** | Used to manage the metadata CSV, filter benign/malignant classes, and create structured training folders. |
| **scikit-learn** | For later evaluation — accuracy, precision, recall, confusion matrix. Helps measure model performance. |
| **matplotlib** | Visualizes training progress (loss vs epochs) and displays images during debugging. |
| **numpy** | Fast numerical operations, foundational for PyTorch and image preprocessing. |
| **Pillow (PIL)** | Handles image loading, resizing, and format conversions (JPEG/PNG). |
| **onnx** | Converts PyTorch model to ONNX format for deployment across different environments (e.g., browser, mobile). |
| **onnxruntime** | Runs ONNX models outside PyTorch; ideal for lightweight web or mobile deployment. |
| **tensorboard** | Visual dashboard for tracking accuracy/loss metrics during training. |



## How It Works (Simplified)

1. **Dataset →** Skin images labeled benign/malignant.
2. **CNN (EfficientNet-B0) →** Learns visual patterns like edges, textures, and color variations.
3. **Training →** Runs for several **epochs**, improving after each cycle.
4. **Model Output →** Classifies new, unseen images into benign or malignant.

