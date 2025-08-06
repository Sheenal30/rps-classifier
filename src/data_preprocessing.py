# src/data_preprocessing.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

class DataPreprocessor:
    def __init__(self, data_path='../data/raw', img_size=(224, 224), batch_size=32):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = ['rock', 'paper', 'scissors']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def load_and_preprocess_image(self, image_path, label):
        """Load and preprocess a single image"""
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.img_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image, label
    
    def load_dataset(self):
        """Load the complete dataset"""
        images = []
        labels = []
        
        # Check if we have custom data or downloaded data
        data_dirs = []
        if os.path.exists(f'{self.data_path}/custom'):
            data_dirs.append(f'{self.data_path}/custom')
        if os.path.exists(f'{self.data_path}/train'):
            data_dirs.append(f'{self.data_path}/train')
        if os.path.exists(f'{self.data_path}/test'):
            data_dirs.append(f'{self.data_path}/test')
        
        for data_dir in data_dirs:
            for class_name in self.classes:
                class_path = os.path.join(data_dir, class_name)
                if not os.path.exists(class_path):
                    continue
                    
                print(f"Loading images from {class_path}")
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_file)
                        try:
                            image, label = self.load_and_preprocess_image(
                                img_path, self.class_to_idx[class_name]
                            )
                            images.append(image)
                            labels.append(label)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def create_data_splits(self, test_size=0.2, val_size=0.2):
        """Create train/validation/test splits"""
        X, y = self.load_dataset()
        
        print(f"Total images loaded: {len(X)}")
        print(f"Images per class: {np.bincount(y)}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            stratify=y_temp, random_state=42
        )
        
        print(f"Train set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_data_generators(self):
        """Create data generators with augmentation"""
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation/test data generator (no augmentation)
        val_datagen = ImageDataGenerator()
        
        return train_datagen, val_datagen
    
    def visualize_samples(self, X, y, num_samples=9):
        """Visualize sample images from the dataset"""
        plt.figure(figsize=(12, 8))
        
        for i in range(min(num_samples, len(X))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(X[i])
            plt.title(f"Class: {self.classes[y[i]]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('data/processed/sample_images.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_processed_data(self, train_data, val_data, test_data):
        """Save processed data splits"""
        os.makedirs('data/processed', exist_ok=True)
        
        (X_train, y_train) = train_data
        (X_val, y_val) = val_data  
        (X_test, y_test) = test_data
        
        # Save as numpy arrays
        np.savez_compressed('data/processed/train_data.npz', 
                           images=X_train, labels=y_train)
        np.savez_compressed('data/processed/val_data.npz', 
                           images=X_val, labels=y_val)
        np.savez_compressed('data/processed/test_data.npz', 
                           images=X_test, labels=y_test)
        
        print("Processed data saved successfully!")

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Create data splits
    train_data, val_data, test_data = preprocessor.create_data_splits()
    
    # Visualize samples
    X_train, y_train = train_data
    preprocessor.visualize_samples(X_train, y_train)
    
    # Save processed data
    preprocessor.save_processed_data(train_data, val_data, test_data)
    
    # Create and test data generators
    train_gen, val_gen = preprocessor.create_data_generators()
    print("Data preprocessing completed!")