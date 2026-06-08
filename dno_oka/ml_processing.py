import numpy as np
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
from skimage.filters import frangi

def _patch_moment_features(preprocessed, r, c, patch_size=5):
    """Momenty centralne i Hu dla wycinka 5x5 (wymaganie 4.0)."""
    half = patch_size // 2
    patch = preprocessed[r - half:r + half + 1, c - half:c + half + 1].astype(np.float32)
    return _moments_from_patch(patch)


def _moments_from_patch(patch):
    moments = cv2.moments(patch.astype(np.float32))
    if moments['m00'] == 0:
        return [0.0] * 10
    inv = 1.0 / moments['m00']
    central = [
        moments['mu20'] * inv,
        moments['mu11'] * inv,
        moments['mu02'] * inv,
    ]
    hu = cv2.HuMoments(moments).flatten().tolist()
    return central + hu


def _batch_moment_features(patches, chunk_size=50000):
    """Momenty Hu dla wielu wycinków naraz (szybsze niż pętla po pikselach)."""
    n = patches.shape[0]
    out = np.zeros((n, 10), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        for i in range(start, end):
            out[i] = _moments_from_patch(patches[i])
    return out


def get_vectorized_features(img, patch_size=5):
    """Calculates all features for the whole image at once using fast OpenCV filters."""
    img_f = img.astype(np.float32)
    ksize = (patch_size, patch_size)
    
    # 1. Local Stats (Mean, Var, Std)
    mean = cv2.blur(img_f, ksize)
    mean_sq = cv2.blur(img_f**2, ksize)
    var = np.maximum(mean_sq - mean**2, 0)
    std = np.sqrt(var)
    
    # 2. Edge features (Sobel) - larger kernel for noise robustness
    sobelx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=5)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    
    # 3. Ridge features (Laplacian)
    laplacian = cv2.Laplacian(img_f, cv2.CV_32F, ksize=5)

    frangi_feat = frangi(img, sigmas=np.arange(1, 4, 1), black_ridges=True)
    
    return [mean, std, var, sobel_mag, laplacian, frangi_feat]

def prepare_dataset(image_paths, manual_paths, mask_paths, patch_size=5, max_samples=15000):
    """Prepare training dataset using vectorized features."""
    from image_processing import preprocess_image
    X = []
    y = []
    
    for img_p, man_p, mask_p in zip(image_paths, manual_paths, mask_paths):
        img = cv2.imread(img_p)
        if img is None: continue
        preprocessed = preprocess_image(img)
        manual = cv2.imread(man_p, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        
        feats = get_vectorized_features(preprocessed, patch_size)
        
        coords = np.argwhere(mask > 0)
        vessel_coords = coords[manual[coords[:, 0], coords[:, 1]] > 0]
        non_vessel_coords = coords[manual[coords[:, 0], coords[:, 1]] == 0]
        
        if len(vessel_coords) == 0 or len(non_vessel_coords) == 0: continue
        
        samples_per_class = min(len(vessel_coords), len(non_vessel_coords), max_samples // (2 * len(image_paths)))
        if samples_per_class == 0: continue
        
        idx_v = np.random.choice(len(vessel_coords), samples_per_class, replace=False)
        idx_nv = np.random.choice(len(non_vessel_coords), samples_per_class, replace=False)
        
        sel_coords = np.vstack([vessel_coords[idx_v], non_vessel_coords[idx_nv]])
        
        for r, c in sel_coords:
            feat_vector = [f[r, c] for f in feats]
            feat_vector.extend(_patch_moment_features(preprocessed, r, c, patch_size))
            X.append(feat_vector)
            y.append(1 if manual[r, c] > 0 else 0)
            
    return np.array(X), np.array(y)

class MLVesselSegmenter:
    def __init__(self, patch_size=5):
        self.patch_size = patch_size
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        self.is_trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True

    def predict_vesselness(self, preprocessed_image, mask=None):
        if not self.is_trained:
            raise ValueError("Model not trained")

        from numpy.lib.stride_tricks import sliding_window_view

        feats = get_vectorized_features(preprocessed_image, self.patch_size)

        if mask is not None:
            rows, cols = np.where(mask > 0)
        else:
            rows, cols = np.indices(preprocessed_image.shape)
            rows, cols = rows.ravel(), cols.ravel()

        base = np.stack([f[rows, cols] for f in feats], axis=1)

        half = self.patch_size // 2
        padded = np.pad(preprocessed_image, half, mode='reflect')
        patches = sliding_window_view(padded, (self.patch_size, self.patch_size))[rows, cols]
        moment_feats = _batch_moment_features(patches)

        features_matrix = np.hstack([base, moment_feats])
        probs = self.model.predict_proba(features_matrix)[:, 1]

        output = np.zeros_like(preprocessed_image, dtype=np.float32)
        output[rows, cols] = probs
        return output
