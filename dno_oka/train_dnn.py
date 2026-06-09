import os
import numpy as np
from dnn_processing import DNNVesselSegmenter, prepare_dnn_dataset
from tqdm import tqdm

# Konfiguracja
IMAGES_DIR = 'dno_oka/data/images/'
MANUAL_DIR = 'dno_oka/data/manual/'
MASK_DIR = 'dno_oka/data/mask/'
MODEL_SAVE_PATH = 'dno_oka/vessel_cnn.h5'

def train():
    print("🚀 Przygotowanie danych dla CNN (wycinki 5x5 + augmentacja)...")
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.JPG', '.png'))])
    
    train_images = image_files[:15]
    
    img_paths = [os.path.join(IMAGES_DIR, f) for f in train_images]
    man_paths = [os.path.join(MANUAL_DIR, os.path.splitext(f)[0] + '.tif') for f in train_images]
    mask_paths = [os.path.join(MASK_DIR, os.path.splitext(f)[0] + '_mask.tif') for f in train_images]

    # prepare_dnn_dataset z augmentacją
    X, y = prepare_dnn_dataset(img_paths, man_paths, mask_paths, augment=True)
    
    print(f"📊 Zebrano {len(X)} wycinków. Rozpoczynam trening sieci...")
    
    segmenter = DNNVesselSegmenter()
    
    # Callback do paska postępu w Keras
    from tensorflow.keras.callbacks import Callback
    class TqdmProgress(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f" Epoch {epoch+1}: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}")

    segmenter.train(X, y, epochs=10, batch_size=256)
    
    print(f"💾 Zapisywanie modelu do {MODEL_SAVE_PATH}...")
    segmenter.save(MODEL_SAVE_PATH)
    print("✅ Trening CNN zakończony pomyślnie!")

if __name__ == "__main__":
    train()
