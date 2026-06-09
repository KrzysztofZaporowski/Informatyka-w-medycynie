"""Wymaganie 5.0: głęboka sieć neuronowa (Keras/TensorFlow) na wycinkach 5x5 px."""
import numpy as np
import cv2

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def prepare_dnn_dataset(image_paths, manual_paths, mask_paths, patch_size=5, max_samples=10000, augment=False):
    """Losowe, zbalansowane wycinki 5x5 z opcjonalną augmentacją (rotacje)."""
    from image_processing import preprocess_image
    from tqdm import tqdm

    half = patch_size // 2
    patches = []
    labels = []

    print("Przygotowywanie wycinków dla CNN...")
    for img_p, man_p, mask_p in tqdm(zip(image_paths, manual_paths, mask_paths), total=len(image_paths)):
        img = cv2.imread(img_p)
        if img is None:
            continue
        manual = cv2.imread(man_p, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

        variants = [(img, manual, mask)]
        if augment:
            for angle in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                variants.append((
                    cv2.rotate(img, angle),
                    cv2.rotate(manual, angle),
                    cv2.rotate(mask, angle)
                ))

        for v_img, v_man, v_mask in variants:
            preprocessed = preprocess_image(v_img)
            padded = np.pad(preprocessed, half, mode='reflect')
            
            coords = np.argwhere(v_mask > half)
            vessel_coords = coords[v_man[coords[:, 0], coords[:, 1]] > 0]
            non_vessel_coords = coords[v_man[coords[:, 0], coords[:, 1]] == 0]

            if len(vessel_coords) == 0 or len(non_vessel_coords) == 0:
                continue

            per_variant = max_samples // (len(image_paths) * len(variants))
            per_class = min(len(vessel_coords), len(non_vessel_coords), per_variant // 2)
            
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
                labels.append(1 if v_man[r, c] > 0 else 0)

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

        # Build FCN model on the fly or use cached
        if not hasattr(self, '_fcn_model') or self._fcn_model is None:
            self._fcn_model = self._convert_to_fcn(self.model)

        h, w = preprocessed_image.shape
        tile_size = 1024 # Increased for speed
        margin = 32
        
        padded_img = np.pad(preprocessed_image.astype(np.float32), margin, mode='reflect')
        output = np.zeros((h, w), dtype=np.float32)

        # Iterate over tiles - fewer tiles means less overhead
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                # Extract tile with margin
                tile_with_margin = padded_img[y : y_end + 2*margin, x : x_end + 2*margin]
                tile_input = (tile_with_margin / 255.0)[np.newaxis, ..., np.newaxis]
                
                # Fast model call instead of .predict()
                tile_probs = self._fcn_model(tile_input, training=False).numpy()[0, :, :, 0]
                
                h_area = y_end - y
                w_area = x_end - x
                output[y:y_end, x:x_end] = tile_probs[margin : margin + h_area, margin : margin + w_area]

        if mask is not None:
            output[mask == 0] = 0
            
        return output


    def _convert_to_fcn(self, model):
        """Dynamically converts a patch-based CNN to a Fully Convolutional Network."""
        from tensorflow.keras import layers, Model
        
        # Determine input depth from the first layer
        input_depth = 1
        for layer in model.layers:
            if hasattr(layer, 'input_shape'):
                input_depth = layer.input_shape[-1]
                break
            elif hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
                input_depth = layer.get_weights()[0].shape[2]
                break

        inp = layers.Input(shape=(None, None, input_depth))
        x = inp
        
        dense_count = 0
        layers_map = []
        
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
            elif isinstance(layer, layers.Activation):
                new_layer = layers.Activation(layer.activation)
                x = new_layer(x)
                layers_map.append((layer, new_layer))
            elif isinstance(layer, layers.Dense):
                dense_count += 1
                if dense_count == 1:
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
            elif isinstance(layer, layers.Flatten) or isinstance(layer, layers.Dropout):
                continue
        
        fcn = Model(inp, x)
        
        # Transfer weights
        for orig, new in layers_map:
            w = orig.get_weights()
            if not w: continue
            
            if isinstance(orig, layers.Dense):
                w_dense, b_dense = w
                if new.kernel_size[0] > 1:
                    # First dense: reshape (patch*patch*depth_in, units) -> (patch, patch, depth_in, units)
                    depth_in = w_dense.shape[0] // (self.patch_size * self.patch_size)
                    w_new = w_dense.reshape((self.patch_size, self.patch_size, depth_in, orig.units))
                    new.set_weights([w_new, b_dense])
                else:
                    # Subsequent dense: reshape (depth_in, units) -> (1, 1, depth_in, units)
                    w_new = w_dense.reshape((1, 1, w_dense.shape[0], w_dense.shape[1]))
                    new.set_weights([w_new, b_dense])
            else:
                new.set_weights(w)
                
        return fcn

