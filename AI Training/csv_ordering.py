import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


csv_file = "metadata.csv"          
image_dir = "ISIC-images"
output_dir = "dataset"


train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1


df = pd.read_csv(csv_file)

# split into image_id and diagnosis_1
df = df[["isic_id", "diagnosis_1"]]

train_df, temp_df = train_test_split(df, test_size=(1-train_ratio), stratify=df["diagnosis_1"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=(test_ratio/(test_ratio+val_ratio)), stratify=temp_df["diagnosis_1"], random_state=42)

splits = {"train": train_df, "val": val_df, "test": test_df}


for split in splits.keys():
    for label in ["benign", "malignant"]:
        os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)


for split, split_df in splits.items():
    for _, row in split_df.iterrows():
        #images are .jpg
        img_name = row["isic_id"] + ".jpg"
        label = row["diagnosis_1"].lower()   #"benign" or "malignant"
        
        src_path = os.path.join(image_dir, img_name)
        dst_path = os.path.join(output_dir, split, label, img_name)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"⚠️ Missing image: {img_name}")

print("Dataset organized into train/val/test folders!")
