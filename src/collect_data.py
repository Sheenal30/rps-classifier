# src/collect_data.py
import os
import requests
import zipfile
from tqdm import tqdm
import tensorflow as tf

# Define base raw data path
RAW_DATA_DIR = os.path.join("data", "raw")

def download_rps_dataset():
    """Download Rock Paper Scissors dataset from TensorFlow"""
    
    # Create raw data directory
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Download dataset using TensorFlow Datasets
    import tensorflow_datasets as tfds
    
    # Download and prepare the dataset
    ds, info = tfds.load('rock_paper_scissors', 
                         split=['train', 'test'], 
                         with_info=True, 
                         as_supervised=True)
    
    train_ds, test_ds = ds
    
    print(f"Dataset info: {info}")
    print(f"Number of classes: {info.features['label'].num_classes}")
    print(f"Class names: {info.features['label'].names}")
    
    return train_ds, test_ds, info

def save_dataset_locally(train_ds, test_ds, info):
    """Save downloaded dataset as images locally"""
    import matplotlib.pyplot as plt
    
    class_names = info.features['label'].names

    # Create directories for each class inside data/raw/train and data/raw/test
    for split in ['train', 'test']:
        for class_name in class_names:
            class_dir = os.path.join(RAW_DATA_DIR, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    # Save training images
    for i, (image, label) in enumerate(tqdm(train_ds, desc="Saving train images")):
        class_name = class_names[label.numpy()]
        path = os.path.join(RAW_DATA_DIR, "train", class_name, f"img_{i:04d}.jpg")
        plt.imsave(path, image.numpy())
    
    # Save test images
    for i, (image, label) in enumerate(tqdm(test_ds, desc="Saving test images")):
        class_name = class_names[label.numpy()]
        path = os.path.join(RAW_DATA_DIR, "test", class_name, f"img_{i:04d}.jpg")
        plt.imsave(path, image.numpy())

if __name__ == "__main__":
    train_ds, test_ds, info = download_rps_dataset()
    save_dataset_locally(train_ds, test_ds, info)
    print("✅ Dataset downloaded and saved successfully in data/raw")
