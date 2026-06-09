"""Wymaganie 5.0: głęboka sieć neuronowa (Keras/TensorFlow) na wycinkach 5x5 px."""
import numpy as np
import cv2

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def prepare_dnn_dataset(image_paths, manual_paths, mask_paths, patch_size=5, max_samples=8000):
    """Losowe, zbalansowane wycinki 5x5 z etykietą środkowego piksela."""
    from image_processing import preprocess_image

    half = patch_size // 2
    patches = []
    labels = []

    for img_p, man_p, mask_p in zip(image_paths, manual_paths, mask_paths):
        img = cv2.imread(img_p)
        if img is None:
            continue
        preprocessed = preprocess_image(img)
        manual = cv2.imread(man_p, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

        padded = np.pad(preprocessed, half, mode='reflect')
        coords = np.argwhere(mask > half)
        vessel_coords = coords[manual[coords[:, 0], coords[:, 1]] > 0]
        non_vessel_coords = coords[manual[coords[:, 0], coords[:, 1]] == 0]

        if len(vessel_coords) == 0 or len(non_vessel_coords) == 0:
            continue

        per_class = min(
            len(vessel_coords),
            len(non_vessel_coords),
            max_samples // (2 * len(image_paths)),
        )
        if per_class == 0:
            continue

        idx_v = np.random.choice(len(vessel_coords), per_class, replace=False)
        idx_nv = np.random.choice(len(non_vessel_coords), per_class, replace=False)
        sel = np.vstack([vessel_coords[idx_v], non_vessel_coords[idx_nv]])

        for r, c in sel:
            pr, pc = r + half, c + half
            patch = padded[pr - half:pr + half + 1, pc - half:pc + half + 1]
            patch = patch.astype(np.float32) / 255.0
            patches.append(patch[..., np.newaxis])
            labels.append(1 if manual[r, c] > 0 else 0)

    X = np.array(patches, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def _build_cnn(patch_size=5):
    model = keras.Sequential([
        keras.layers.Input(shape=(patch_size, patch_size, 1)),
        keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid'),
    ], name='vessel_patch_cnn')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


class DNNVesselSegmenter:
    """CNN uczona na wycinkach obrazu (analogicznie do klasyfikatora ML z wym. 4.0)."""

    def __init__(self, patch_size=5):
        if not TF_AVAILABLE:
            raise ImportError(
                'TensorFlow nie jest zainstalowany. Uruchom: pip install tensorflow'
            )
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.model = _build_cnn(patch_size)
        self.is_trained = False

    def train(self, X, y, epochs=8, batch_size=128):
        if len(X) == 0:
            raise ValueError('Pusty zbiór treningowy dla CNN')
        
        # Rekompilacja przed treningiem rozwiązuje błąd "Unknown variable" 
        # występujący w Keras po wczytaniu modelu z pliku.
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )
        self.is_trained = True

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)
        self.is_trained = True

    def predict_vesselness(self, preprocessed_image, mask=None):
        if not self.is_trained:
            raise ValueError('Model CNN nie jest wytrenowany')

        from numpy.lib.stride_tricks import sliding_window_view

        padded = np.pad(preprocessed_image.astype(np.float32), self.half, mode='reflect')
        if mask is not None:
            rows, cols = np.where(mask > 0)
        else:
            rows, cols = np.indices(preprocessed_image.shape)
            rows, cols = rows.ravel(), cols.ravel()

        patches = sliding_window_view(padded, (self.patch_size, self.patch_size))[rows, cols]
        patches = (patches / 255.0).astype(np.float32)[..., np.newaxis]
        probs = self.model.predict(patches, batch_size=512, verbose=0).ravel()

        output = np.zeros_like(preprocessed_image, dtype=np.float32)
        output[rows, cols] = probs
        return output
