"""Wymaganie 5.0: głęboka sieć neuronowa (Keras/TensorFlow) na wycinkach 5x5 px."""
import numpy as np
import cv2

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ml_processing import get_vectorized_features

def prepare_dnn_dataset(image_paths, manual_paths, mask_paths, patch_size=5, max_samples=40000, augment=True):
    """Przygotowuje zbalansowany zbiór (50/50) dla CNN korzystając z cech analogicznych do ML."""
    from image_processing import preprocess_image
    from tqdm import tqdm
    import gc

    half = patch_size // 2
    patches = []
    labels = []

    print(f"Przygotowywanie zbalansowanych wycinków (multi-channel) dla CNN (max {max_samples})...")
    for img_p, man_p, mask_p in tqdm(zip(image_paths, manual_paths, mask_paths), total=len(image_paths)):
        img = cv2.imread(img_p)
        if img is None: continue
        manual = cv2.imread(man_p, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

        variants = [(img, manual, mask)]
        if augment:
            variants.append((cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 
                             cv2.rotate(manual, cv2.ROTATE_90_CLOCKWISE), 
                             cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)))

        for v_img, v_man, v_mask in variants:
            preprocessed = preprocess_image(v_img)
            
            # Extract features (same as ML)
            feats = get_vectorized_features(preprocessed, patch_size)
            # Stack features into a multi-channel image
            feat_img = np.stack(feats, axis=-1)
            
            # Normalize features per channel
            means = np.mean(feat_img, axis=(0,1))
            stds = np.std(feat_img, axis=(0,1)) + 1e-7
            feat_img = (feat_img - means) / stds
            
            padded = np.pad(feat_img, ((half, half), (half, half), (0, 0)), mode='reflect')
            
            coords_vessel = np.argwhere((v_man > 0) & (v_mask > 0))
            coords_background = np.argwhere((v_man == 0) & (v_mask > 0))

            if len(coords_vessel) == 0 or len(coords_background) == 0:
                del feat_img, padded, feats
                continue

            per_variant = max_samples // (len(image_paths) * len(variants))
            samples_per_class = min(len(coords_vessel), len(coords_background), per_variant // 2)

            if samples_per_class > 0:
                idx_v = np.random.choice(len(coords_vessel), samples_per_class, replace=False)
                idx_bg = np.random.choice(len(coords_background), samples_per_class, replace=False)
                sel = np.vstack([coords_vessel[idx_v], coords_background[idx_bg]])

                for r, c in sel:
                    pr, pc = r + half, c + half
                    patches.append(padded[pr - half:pr + half + 1, pc - half:pc + half + 1].copy())
                    labels.append(1 if v_man[r, c] > 0 else 0)
            
            del feat_img, padded, feats
            gc.collect()

    return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int32)


def _build_cnn(patch_size=5, num_channels=17):
    model = keras.Sequential([
        keras.layers.Input(shape=(patch_size, patch_size, num_channels)),
        
        keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        
        keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        
        keras.layers.Dense(1, activation='sigmoid'),
    ], name='vessel_patch_cnn')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


class DNNVesselSegmenter:
    """CNN uczona na wycinkach obrazu (analogicznie do klasyfikatora ML z wym. 4.0)."""

    def __init__(self, patch_size=5, num_channels=17):
        if not TF_AVAILABLE:
            raise ImportError(
                'TensorFlow nie jest zainstalowany. Uruchom: pip install tensorflow'
            )
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.half = patch_size // 2
        self.model = _build_cnn(patch_size, num_channels)
        self.is_trained = False

    def train(self, X, y, epochs=30, batch_size=256):
        if len(X) == 0:
            raise ValueError('Pusty zbiór treningowy dla CNN')
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-4),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
        ]
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1,
        )
        self.is_trained = True

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)
        self.num_channels = self.model.input_shape[-1]
        self.is_trained = True

    def predict_vesselness(self, preprocessed_image, mask=None):
        if not self.is_trained:
            raise ValueError('Model CNN nie jest wytrenowany')

        # 1. Extract features (matching training)
        h, w = preprocessed_image.shape
        feats = get_vectorized_features(preprocessed_image, self.patch_size)
        feat_img = np.stack(feats, axis=-1)
        
        # Consistent normalization
        feat_img = (feat_img - np.mean(feat_img, axis=(0,1))) / (np.std(feat_img, axis=(0,1)) + 1e-7)

        # 2. Build or use cached FCN model
        if not hasattr(self, '_fcn_model') or self._fcn_model is None:
            self._fcn_model = self._convert_to_fcn(self.model)

        # 3. Optimized model execution with tiling to avoid OOM
        output = np.zeros((h, w), dtype=np.float32)
        tile_size = 512
        overlap = self.patch_size
        
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                y_e, x_e = min(y + tile_size, h), min(x + tile_size, w)
                t_in = feat_img[y:y_e, x:x_e][np.newaxis, ...]
                
                t_p = self._fcn_model(t_in, training=False).numpy()[0, :, :, 0]
                output[y:y_e, x:x_e] = np.maximum(output[y:y_e, x:x_e], t_p)

        if mask is not None:
            output[mask == 0] = 0
            
        return np.clip(output, 0, 1)


    def _convert_to_fcn(self, model):
        """Dynamically converts a patch-based CNN to a Fully Convolutional Network."""
        from tensorflow.keras import layers, Model
        import tensorflow as tf

        inp = layers.Input(shape=(None, None, self.num_channels))
        x = inp
        
        layers_map = []
        dense_idx = 0
        
        for layer in model.layers:
            if isinstance(layer, layers.Conv2D):
                new_layer = layers.Conv2D(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    padding='same',
                    activation=layer.activation,
                )
                x = new_layer(x)
                layers_map.append((layer, new_layer))
            elif isinstance(layer, layers.BatchNormalization):
                new_layer = layers.BatchNormalization()
                x = new_layer(x)
                layers_map.append((layer, new_layer))
            elif isinstance(layer, layers.Dense):
                dense_idx += 1
                # Use same padding to maintain dimensions in FCN mode
                if dense_idx == 1:
                    new_layer = layers.Conv2D(
                        filters=layer.units,
                        kernel_size=self.patch_size,
                        padding='same', 
                        activation=layer.activation,
                    )
                else:
                    new_layer = layers.Conv2D(
                        filters=layer.units,
                        kernel_size=1,
                        padding='same',
                        activation=layer.activation,
                    )
                x = new_layer(x)
                layers_map.append((layer, new_layer))
            elif isinstance(layer, (layers.Flatten, layers.Dropout)):
                continue
        
        fcn = Model(inp, x)
        
        # Transfer weights
        for orig, new in layers_map:
            w = orig.get_weights()
            if not w: continue
            
            if isinstance(orig, layers.Dense):
                w_dense, b_dense = w
                if new.kernel_size[0] > 1:
                    depth_in = w_dense.shape[0] // (self.patch_size * self.patch_size)
                    w_new = w_dense.reshape((self.patch_size, self.patch_size, depth_in, orig.units))
                    new.set_weights([w_new, b_dense])
                else:
                    w_new = w_dense.reshape((1, 1, w_dense.shape[0], w_dense.shape[1]))
                    new.set_weights([w_new, b_dense])
            else:
                new.set_weights(w)
                
        return fcn

