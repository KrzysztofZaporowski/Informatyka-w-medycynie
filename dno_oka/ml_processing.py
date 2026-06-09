import numpy as np
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
from skimage.filters import frangi

def _get_fast_moments(img, patch_size=5):
    """Calculates central and Hu moments for every pixel using convolutions (vectorized)."""
    img_f = img.astype(np.float32)
    k = np.ones((patch_size, patch_size), dtype=np.float32)
    
    # Raw moments using filter2D
    m00 = cv2.filter2D(img_f, -1, k)
    
    # Coordinate grids for the patch
    y, x = np.mgrid[-patch_size//2 + 1:patch_size//2 + 1, -patch_size//2 + 1:patch_size//2 + 1]
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    m10 = cv2.filter2D(img_f, -1, x)
    m01 = cv2.filter2D(img_f, -1, y)
    m20 = cv2.filter2D(img_f, -1, x*x)
    m02 = cv2.filter2D(img_f, -1, y*y)
    m11 = cv2.filter2D(img_f, -1, x*y)
    m30 = cv2.filter2D(img_f, -1, x*x*x)
    m03 = cv2.filter2D(img_f, -1, y*y*y)
    m21 = cv2.filter2D(img_f, -1, x*x*y)
    m12 = cv2.filter2D(img_f, -1, x*y*y)

    # Central moments
    # Since we use relative coordinates (centered at 0,0), if the patch was symmetric and 
    # intensity uniform, m10/m00 would be the offset from center.
    eps = 1e-7
    x_bar = m10 / (m00 + eps)
    y_bar = m01 / (m00 + eps)

    mu20 = m20 - x_bar * m10
    mu02 = m02 - y_bar * m01
    mu11 = m11 - x_bar * m01
    
    mu30 = m30 - 3 * x_bar * m20 + 2 * x_bar**2 * m10
    mu03 = m03 - 3 * y_bar * m02 + 2 * y_bar**2 * m01
    mu21 = m21 - 2 * x_bar * m11 - y_bar * m20 + 2 * x_bar**2 * m01
    mu12 = m12 - 2 * y_bar * m11 - x_bar * m02 + 2 * y_bar**2 * m10

    # Normalized central moments
    nu20 = mu20 / (m00**2 + eps)
    nu02 = mu02 / (m00**2 + eps)
    nu11 = mu11 / (m00**2 + eps)
    nu30 = mu30 / (m00**2.5 + eps)
    nu03 = mu03 / (m00**2.5 + eps)
    nu21 = mu21 / (m00**2.5 + eps)
    nu12 = mu12 / (m00**2.5 + eps)

    # Hu moments (simplified/first few)
    h1 = nu20 + nu02
    h2 = (nu20 - nu02)**2 + 4 * nu11**2
    h3 = (nu30 - 3*nu12)**2 + (3*nu21 - nu03)**2
    h4 = (nu30 + nu12)**2 + (nu21 + nu03)**2
    h5 = (nu30 - 3*nu12)*(nu30 + nu12)*((nu30 + nu12)**2 - 3*(nu21 + nu03)**2) + \
         (3*nu21 - nu03)*(nu21 + nu03)*(3*(nu30 + nu12)**2 - (nu21 + nu03)**2)
    h6 = (nu20 - nu02)*((nu30 + nu12)**2 - (nu21 + nu03)**2) + 4*nu11*(nu30 + nu12)*(nu21 + nu03)
    h7 = (3*nu21 - nu03)*(nu30 + nu12)*((nu30 + nu12)**2 - 3*(nu21 + nu03)**2) - \
         (nu30 - 3*nu12)*(nu21 + nu03)*(3*(nu30 + nu12)**2 - (nu21 + nu03)**2)

    return [mu20/(m00+eps), mu11/(m00+eps), mu02/(m00+eps), h1, h2, h3, h4, h5, h6, h7]

from skimage.filters import frangi, sato

def get_vectorized_features(img, patch_size=5):
    """Calculates all features for the whole image at once using fast OpenCV filters."""
    img_f = img.astype(np.float32)
    ksize = (patch_size, patch_size)
    
    # 1. Local Stats (Mean, Var, Std)
    mean = cv2.blur(img_f, ksize)
    mean_sq = cv2.blur(img_f**2, ksize)
    var = np.maximum(mean_sq - mean**2, 0)
    std = np.sqrt(var)
    
    # 2. Edge features (Sobel)
    sobelx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    
    # 3. Ridge features (Laplacian)
    laplacian = cv2.Laplacian(img_f, cv2.CV_32F, ksize=3)

    # 4. Specialized Vessel Filters (Sato & Frangi)
    sigmas = np.arange(1, 4, 1)
    frangi_feat = frangi(img, sigmas=sigmas, black_ridges=True)
    sato_feat = sato(img, sigmas=sigmas, black_ridges=True)
    
    moments = _get_fast_moments(img, patch_size)
    
    return [mean, std, var, sobel_mag, laplacian, frangi_feat, sato_feat] + moments

def prepare_dataset(image_paths, manual_paths, mask_paths, patch_size=5, max_samples=20000, augment=False):
    """Prepare training dataset with optional 90-deg rotations augmentation."""
    from image_processing import preprocess_image
    from tqdm import tqdm
    X = []
    y = []
    
    print("Preprocessing and extracting features...")
    for img_p, man_p, mask_p in tqdm(zip(image_paths, manual_paths, mask_paths), total=len(image_paths)):
        img = cv2.imread(img_p)
        if img is None: continue
        manual = cv2.imread(man_p, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        
        # Data Augmentation: Original + Rotations
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
            feats = get_vectorized_features(preprocessed, patch_size)
            
            coords = np.argwhere(v_mask > 0)
            vessel_coords = coords[v_man[coords[:, 0], coords[:, 1]] > 0]
            non_vessel_coords = coords[v_man[coords[:, 0], coords[:, 1]] == 0]
            
            if len(vessel_coords) == 0 or len(non_vessel_coords) == 0: continue
            
            samples_per_variant = max_samples // (len(image_paths) * len(variants))
            samples_per_class = min(len(vessel_coords), len(non_vessel_coords), samples_per_variant // 2)
            
            if samples_per_class == 0: continue
            
            idx_v = np.random.choice(len(vessel_coords), samples_per_class, replace=False)
            idx_nv = np.random.choice(len(non_vessel_coords), samples_per_class, replace=False)
            
            sel_coords = np.vstack([vessel_coords[idx_v], non_vessel_coords[idx_nv]])
            
            for r, c in sel_coords:
                X.append([f[r, c] for f in feats])
                y.append(1 if v_man[r, c] > 0 else 0)
            
    return np.array(X), np.array(y)

class MLVesselSegmenter:
    def __init__(self, patch_size=5):
        self.patch_size = patch_size
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        self.is_trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        import joblib
        self.model = joblib.load(path)
        self.is_trained = True

    def predict_vesselness(self, preprocessed_image, mask=None):
        if not self.is_trained:
            raise ValueError("Model not trained")

        feats = get_vectorized_features(preprocessed_image, self.patch_size)

        if mask is not None:
            rows, cols = np.where(mask > 0)
        else:
            rows, cols = np.indices(preprocessed_image.shape)
            rows, cols = rows.ravel(), cols.ravel()

        features_matrix = np.stack([f[rows, cols] for f in feats], axis=1)
        probs = self.model.predict_proba(features_matrix)[:, 1]

        output = np.zeros_like(preprocessed_image, dtype=np.float32)
        output[rows, cols] = probs
        return output

