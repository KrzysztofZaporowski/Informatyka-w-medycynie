import cv2
import numpy as np
from skimage.filters import frangi, sato, threshold_sauvola, threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

def preprocess_image(image):
    # Extract green channel
    green_channel = image[:, :, 1]
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green_channel)
    
    # Slight denoising - increased blur for better noise suppression
    enhanced_green = cv2.GaussianBlur(enhanced_green, (5, 5), 0)
    
    return enhanced_green

def segment_vessels(preprocessed_image, method='Filtr Frangi', mask=None, sigmas=np.arange(1, 5, 0.5), ml_segmenter=None):
    # Vessel enhancement
    if method == 'Filtr Frangi':
        # beta=0.2 makes the filter more selective for line-like structures (vessels)
        vessel_enhanced = frangi(preprocessed_image, sigmas=sigmas, beta=0.2, black_ridges=True)
    elif method == 'Filtr Sato':
        vessel_enhanced = sato(preprocessed_image, sigmas=sigmas, black_ridges=True)
    elif method == 'Klasyfikator ML':
        if ml_segmenter is None:
            raise ValueError("ml_segmenter must be provided for 'Klasyfikator ML' method")
        # ML segmenter returns vesselness probability map (0-1)
        vessel_enhanced = ml_segmenter.predict_vesselness(preprocessed_image, mask=mask)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize to [0, 1]
    v_min, v_max = vessel_enhanced.min(), vessel_enhanced.max()
    if v_max > v_min:
        vessel_enhanced = (vessel_enhanced - v_min) / (v_max - v_min)
    
    # Thresholding
    if method == 'Filtr Sato':
        if mask is not None:
            # Oblicz Otsu tylko dla pikseli wewnątrz maski oka
            thresh = threshold_otsu(vessel_enhanced[mask > 0])
        else:
            thresh = threshold_otsu(vessel_enhanced)
        binary = vessel_enhanced > thresh
    elif method == 'Klasyfikator ML':
        binary = vessel_enhanced > 0.5
    else:
        # For Frangi Sauvola works well. 
        k_val = 0.5 
        thresh_sauvola = threshold_sauvola(vessel_enhanced, window_size=25, k=k_val)
        binary = vessel_enhanced > thresh_sauvola

    binary_uint8 = (binary.astype(np.uint8) * 255)

    # Extra denoising for ML
    if method == 'Klasyfikator ML':
        binary_uint8 = cv2.medianBlur(binary_uint8, 3)

    if method == 'Filtr Sato':    
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_uint8 = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel)
    
    # Apply mask if provided
    if mask is not None:
        binary_uint8 = cv2.bitwise_and(binary_uint8, binary_uint8, mask=mask)
    
    # Post-processing: remove small objects
    binary_bool = binary_uint8 > 0
    min_obj_size = 150 if method == 'Filtr Sato' else 100
    cleaned = remove_small_objects(binary_bool, max_size=min_obj_size - 1)
    binary_cleaned = (cleaned.astype(np.uint8) * 255)
    
    return binary_cleaned

def get_overlay(image, binary_vessels):
    overlay = image.copy()
    # Vessels in purple color for overlay
    overlay[binary_vessels > 0] = [128, 0, 255]
    return overlay
