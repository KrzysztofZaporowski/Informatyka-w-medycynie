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
    
    # Slight denoising
    enhanced_green = cv2.GaussianBlur(enhanced_green, (3, 3), 0)
    
    return enhanced_green

def segment_vessels(preprocessed_image, method='Filtr Frangi', mask=None, sigmas=np.arange(1, 5, 1)):
    # Vessel enhancement
    if method == 'Filtr Frangi':
        vessel_enhanced = frangi(preprocessed_image, sigmas=sigmas, black_ridges=True)
    elif method == 'Filtr Sato':
        vessel_enhanced = sato(preprocessed_image, sigmas=sigmas, black_ridges=True)
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
    else:
        thresh_sauvola = threshold_sauvola(vessel_enhanced, window_size=25, k=0.2)
        binary = vessel_enhanced > thresh_sauvola
    
    binary_uint8 = (binary.astype(np.uint8) * 255)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_uint8 = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel)
    
    # Apply mask if provided
    if mask is not None:
        binary_uint8 = cv2.bitwise_and(binary_uint8, binary_uint8, mask=mask)
    
    # Post-processing: remove small objects
    binary_bool = binary_uint8 > 0
    min_obj_size = 150 if method == 'Filtr Sato' else 50
    cleaned = remove_small_objects(binary_bool, min_size=min_obj_size)
    binary_cleaned = (cleaned.astype(np.uint8) * 255)
    
    return binary_cleaned

def get_overlay(image, binary_vessels):
    overlay = image.copy()
    # Vessels in green color for overlay
    overlay[binary_vessels > 0] = [0, 255, 0]
    return overlay
