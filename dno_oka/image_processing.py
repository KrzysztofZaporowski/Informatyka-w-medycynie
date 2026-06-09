import cv2
import numpy as np
from skimage.filters import frangi, sato, threshold_sauvola, threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

def preprocess_image(image):
    # Extract green channel
    green_channel = image[:, :, 1]
    
    # Refined CLAHE: lower clipLimit to reduce background noise
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green_channel)

    # Subtle sharpening
    blurred = cv2.GaussianBlur(enhanced_green, (0, 0), 2)
    enhanced_green = cv2.addWeighted(enhanced_green, 1.3, blurred, -0.3, 0)
    enhanced_green = cv2.GaussianBlur(enhanced_green, (3, 3), 0)
    
    return enhanced_green

def segment_vessels(
    preprocessed_image,
    method='Filtr Frangi',
    mask=None,
    sigmas=np.arange(1, 5, 0.5),
    ml_segmenter=None,
    dnn_segmenter=None,
):
    # Vessel enhancement
    if method == 'Filtr Frangi':
        vessel_enhanced = frangi(preprocessed_image, sigmas=sigmas, beta=0.2, black_ridges=True)
    elif method == 'Filtr Sato':
        vessel_enhanced = sato(preprocessed_image, sigmas=sigmas, black_ridges=True)
    elif method == 'Klasyfikator ML':
        if ml_segmenter is None:
            raise ValueError("ml_segmenter must be provided for 'Klasyfikator ML' method")
        vessel_enhanced = ml_segmenter.predict_vesselness(preprocessed_image, mask=mask)
    elif method == 'Sieć neuronowa (CNN)':
        if dnn_segmenter is None:
            raise ValueError("dnn_segmenter must be provided for 'Sieć neuronowa (CNN)' method")
        vessel_enhanced = dnn_segmenter.predict_vesselness(preprocessed_image, mask=mask)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize to [0, 1]
    v_min, v_max = vessel_enhanced.min(), vessel_enhanced.max()
    if v_max > v_min:
        vessel_enhanced = (vessel_enhanced - v_min) / (v_max - v_min)
    
    # Thresholding
    if method == 'Filtr Sato':
        thresh = threshold_otsu(vessel_enhanced[mask > 0]) if mask is not None else threshold_otsu(vessel_enhanced)
        binary = vessel_enhanced > thresh
    elif method in ('Klasyfikator ML', 'Sieć neuronowa (CNN)'):
        # High confidence threshold for ML/CNN to further reduce noise
        binary = vessel_enhanced > 0.55
    else:
        thresh_sauvola = threshold_sauvola(vessel_enhanced, window_size=25, k=0.5)
        binary = vessel_enhanced > thresh_sauvola

    binary_uint8 = (binary.astype(np.uint8) * 255)

    # Stronger noise removal for ML/CNN
    if method in ('Klasyfikator ML', 'Sieć neuronowa (CNN)'):
        # 5x5 Median filter is very effective for pepper noise
        binary_uint8 = cv2.medianBlur(binary_uint8, 5)
        # Morphological closing to connect broken vessels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_uint8 = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel)

    if method == 'Filtr Sato':    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_uint8 = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel)
    
    # Apply mask
    if mask is not None:
        binary_uint8 = cv2.bitwise_and(binary_uint8, binary_uint8, mask=mask)
    
    # Final cleanup
    binary_bool = binary_uint8 > 0
    min_obj_size = 200 if method in ('Klasyfikator ML', 'Sieć neuronowa (CNN)') else 100
    cleaned = remove_small_objects(binary_bool, min_size=min_obj_size)
    binary_cleaned = (cleaned.astype(np.uint8) * 255)
    
    return binary_cleaned

def get_overlay(image, binary_vessels):
    overlay = image.copy()
    # Vessels in purple color for overlay
    overlay[binary_vessels > 0] = [128, 0, 255]
    return overlay
