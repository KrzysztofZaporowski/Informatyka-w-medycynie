import numpy as np
import cv2
import os
import joblib
from tqdm import tqdm
from ml_processing import MLVesselSegmenter, prepare_dataset

# Konfiguracja - ścieżki absolutne względem skryptu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'data/images/')
MANUAL_DIR = os.path.join(BASE_DIR, 'data/manual/')
MASK_DIR = os.path.join(BASE_DIR, 'data/mask/')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'vessel_ml.joblib')

def train():
    print("🚀 Rozpoczynam przygotowanie danych dla ML...")
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.JPG', '.png'))])
    
    # Wybieramy kilka obrazów do treningu 
    train_images = image_files[:15]
    
    img_paths = [os.path.join(IMAGES_DIR, f) for f in train_images]
    man_paths = [os.path.join(MANUAL_DIR, os.path.splitext(f)[0] + '.tif') for f in train_images]
    mask_paths = [os.path.join(MASK_DIR, os.path.splitext(f)[0] + '_mask.tif') for f in train_images]

    # prepare_dataset musi być zaktualizowane o augmentację i nowe cechy
    X, y = prepare_dataset(img_paths, man_paths, mask_paths, augment=True)
    
    print(f"📊 Zebrano {len(X)} próbek. Rozpoczynam trening Random Forest...")
    
    segmenter = MLVesselSegmenter()
    # Mały trik: tqdm dla RF nie jest prosty, więc informujemy o starcie
    segmenter.train(X, y)
    
    print(f"💾 Zapisywanie modelu do {MODEL_SAVE_PATH}...")
    segmenter.save(MODEL_SAVE_PATH)
    print("✅ Trening ML zakończony pomyślnie!")

if __name__ == "__main__":
    train()
