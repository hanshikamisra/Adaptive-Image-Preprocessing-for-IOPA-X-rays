import numpy as np
import cv2
from quality_analysis import estimate_noise

def static_preprocessing(img: np.ndarray) -> np.ndarray:
    """
    Apply static preprocessing pipeline with improved error handling.
    """
    if img is None or img.size == 0:
        return img
    
    try:
        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Histogram Equalization
        eq = cv2.equalizeHist(img)
        
        # Sharpening Filter
        sharpen_kernel = np.array([[0, -1,  0],
                                   [-1, 5, -1],
                                   [0, -1,  0]], dtype=np.float32)
        sharp = cv2.filter2D(eq, -1, sharpen_kernel)
        
        # Clip values to valid range
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)

        # Basic Denoising
        denoised = cv2.GaussianBlur(sharp, (3, 3), sigmaX=0.5)

        return denoised
    
    except Exception as e:
        print(f"Error in static preprocessing: {e}")
        return img

def calculate_metrics(img: np.ndarray) -> dict:
    """Calculate quality metrics for adaptive preprocessing."""
    if img is None or img.size == 0:
        return {'brightness': 0, 'contrast': 0, 'sharpness': 0, 'noise': 0}
    
    brightness = np.mean(img)
    contrast = np.std(img)
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    noise = estimate_noise(img)

    return {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'noise': noise
    }

def adaptive_preprocessing(img: np.ndarray) -> np.ndarray:
    """
    Adaptive preprocessing pipeline with parameter adjustment based on image metrics.
    """
    if img is None or img.size == 0:
        return img
        
    try:
        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
            
        metrics = calculate_metrics(img)

        # Adaptive CLAHE based on contrast
        contrast = metrics['contrast']
        if contrast < 30:
            clip_limit = 4.0
            tile_grid_size = (8, 8)
        elif contrast < 60:
            clip_limit = 2.0
            tile_grid_size = (8, 8)
        else:
            clip_limit = 1.0
            tile_grid_size = (8, 8)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_clahe = clahe.apply(img)

        # Adaptive Sharpening based on sharpness metric
        sharpness = metrics['sharpness']
        if sharpness < 100:
            amount = 1.5  # Strong sharpening
        elif sharpness < 300:
            amount = 1.0  # Medium sharpening
        else:
            amount = 0.5  # Mild sharpening

        # Unsharp masking
        blurred = cv2.GaussianBlur(img_clahe, (3, 3), 1.0)
        unsharp_img = cv2.addWeighted(img_clahe, 1 + amount, blurred, -amount, 0)
        unsharp_img = np.clip(unsharp_img, 0, 255).astype(np.uint8)

        # Adaptive Denoising based on noise level
        noise = metrics['noise']
        if noise > 10:
            d, sigma_color, sigma_space = 9, 75, 75  # Strong denoising
        elif noise > 5:
            d, sigma_color, sigma_space = 7, 50, 50  # Medium denoising
        else:
            d, sigma_color, sigma_space = 5, 25, 25  # Mild denoising

        denoised = cv2.bilateralFilter(unsharp_img, d=d, 
                                     sigmaColor=sigma_color, 
                                     sigmaSpace=sigma_space)

        return denoised

    except Exception as e:
        print(f"Error in adaptive preprocessing: {e}")
        return img