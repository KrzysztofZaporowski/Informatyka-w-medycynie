"""Wymaganie 5.0: głęboka sieć neuronowa (Keras/TensorFlow) na wycinkach 5x5 px."""
import numpy as np
import cv2

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def prepare_dnn_dataset(image_paths, manual_paths, mask_paths, patch_size=5, max_samples=15000, augment=True):
    """Przygotowuje zbalansowany zbiór z naciskiem na trudne przykłady (brzegi naczyń)."""
    from image_processing import preprocess_image
    from tqdm import tqdm

    half = patch_size // 2
    patches = []
    labels = []

    print("Przygotowywanie wysokiej jakości wycinków dla CNN...")
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
            padded = np.pad(preprocessed, half, mode='reflect')
            
            # Find vessel boundaries (Hard Examples)
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(v_man, kernel, iterations=1)
            eroded = cv2.erode(v_man, kernel, iterations=1)
            boundaries = cv2.absdiff(dilated, eroded)
            
            coords_vessel = np.argwhere((v_man > 0) & (v_mask > 0))
            coords_boundary = np.argwhere((boundaries > 0) & (v_mask > 0))
            coords_background = np.argwhere((v_man == 0) & (v_mask > 0))

            if len(coords_vessel) == 0 or len(coords_background) == 0: continue

            per_variant = max_samples // (len(image_paths) * len(variants))
            # Sample 40% vessels, 40% boundaries (hard), 20% background
            n_v = int(per_variant * 0.4)
            n_b = int(per_variant * 0.4)
            n_bg = per_variant - n_v - n_b

            # Guard against small counts
            n_v = min(n_v, len(coords_vessel))
            n_b = min(n_b, len(coords_boundary))
            n_bg = min(n_bg, len(coords_background))

            idx_v = np.random.choice(len(coords_vessel), n_v, replace=False)
            idx_b = np.random.choice(len(coords_boundary), n_b, replace=False)
            idx_bg = np.random.choice(len(coords_background), n_bg, replace=False)
            
            sel = np.vstack([coords_vessel[idx_v], coords_boundary[idx_b], coords_background[idx_bg]])

            for r, c in sel:
                pr, pc = r + half, c + half
                patch = padded[pr - half:pr + half + 1, pc - half:pc + half + 1]
                patch = (patch.astype(np.float32) - 127.5) / 127.5
                patches.append(patch[..., np.newaxis])
                labels.append(1 if v_man[r, c] > 0 else 0)

    return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int32)


def _build_cnn(patch_size=5):
    model = keras.Sequential([
        keras.layers.Input(shape=(patch_size, patch_size, 1)),
        keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
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

    def train(self, X, y, epochs=15, batch_size=128):
        if len(X) == 0:
            raise ValueError('Pusty zbiór treningowy dla CNN')
        
        # Rekompilacja przed treningiem
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

        # 1. Image Resolution for prediction
        h_orig, w_orig = preprocessed_image.shape
        max_dim = 1024  
        scale = 1.0
        if max(h_orig, w_orig) > max_dim:
            scale = max_dim / max(h_orig, w_orig)
            h_new, w_new = int(h_orig * scale), int(w_orig * scale)
            img_to_proc = cv2.resize(preprocessed_image, (w_new, h_new), interpolation=cv2.INTER_AREA)
        else:
            img_to_proc = preprocessed_image

        # 2. Build or use cached FCN model
        if not hasattr(self, '_fcn_model') or self._fcn_model is None:
            self._fcn_model = self._convert_to_fcn(self.model)

        h, w = img_to_proc.shape
        
        # Consistent normalization with training
        normalized_img = (img_to_proc.astype(np.float32) - 127.5) / 127.5
        tile_input = normalized_img[np.newaxis, ..., np.newaxis]
        
        # 3. Optimized model execution
        try:
            # training=False is CRITICAL for BatchNormalization behavior
            probs_full = self._fcn_model(tile_input, training=False).numpy()[0, :, :, 0]
            output_small = probs_full
        except Exception as e:
            # Fallback for memory constraints
            output_small = np.zeros((h, w), dtype=np.float32)
            tile_size = 512 
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    y_e, x_e = min(y + tile_size, h), min(x + tile_size, w)
                    t_in = normalized_img[y:y_e, x:x_e][np.newaxis, ..., np.newaxis]
                    t_p = self._fcn_model(t_in, training=False).numpy()[0, :, :, 0]
                    output_small[y:y_e, x:x_e] = t_p

        # 4. Upscale back to original size
        if scale != 1.0:
            output = cv2.resize(output_small, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        else:
            output = output_small

        if mask is not None:
            output[mask == 0] = 0
            
        return np.clip(output, 0, 1)


    def _convert_to_fcn(self, model):
        """Dynamically converts a patch-based CNN to a Fully Convolutional Network."""
        from tensorflow.keras import layers, Model
        import tensorflow as tf

        inp = layers.Input(shape=(None, None, 1))
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

