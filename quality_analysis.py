import numpy as np
import cv2
import pandas as pd
from dicom_handler import load_all_dicoms_from_folder

def compute_brightness(img: np.ndarray) -> float:
    """Compute mean brightness."""
    return float(np.mean(img))

def compute_contrast_std(img: np.ndarray) -> float:
    """Compute contrast using standard deviation."""
    return float(np.std(img))

def compute_rms_contrast(img: np.ndarray) -> float:
    """Compute RMS contrast."""
    return float(np.sqrt(np.mean(np.square(img - np.mean(img)))))

def compute_laplacian_variance(img: np.ndarray) -> float:
    """Compute sharpness using Laplacian variance."""
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def compute_tenengrad(img: np.ndarray) -> float:
    """Compute sharpness using Tenengrad method."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx**2 + gy**2).mean())

def estimate_noise(img: np.ndarray, patch_size: int = 8) -> float:
    """Estimate noise level using local standard deviation in flat regions."""
    if img.size == 0:
        return 0.0
    
    h, w = img.shape
    if h < patch_size or w < patch_size:
        return float(np.std(img))
    
    stds = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = img[y:y+patch_size, x:x+patch_size]
            stds.append(np.std(patch))
    
    if not stds:
        return float(np.std(img))
    
    # Use 10th percentile as noise estimate (flat regions)
    return float(np.percentile(stds, 10))

def analyze_image_quality(dicom_dir: str) -> pd.DataFrame:
    """Analyze image quality metrics for all DICOM files in directory."""
    records = []
    handlers = load_all_dicoms_from_folder(dicom_dir)
    
    for handler in handlers:
        try:
            img = handler.get_pixel_array()
            if img is None:
                continue

            metrics = {
                'filename': handler.metadata.get('series_description', 'Unknown'),
                'brightness': compute_brightness(img),
                'contrast_std': compute_contrast_std(img),
                'contrast_rms': compute_rms_contrast(img),
                'sharpness_lapvar': compute_laplacian_variance(img),
                'sharpness_tenengrad': compute_tenengrad(img),
                'noise_estimate': estimate_noise(img),
                'modality': handler.metadata.get('modality'),
                'pixel_spacing': handler.metadata.get('pixel_spacing'),
                'photometric': handler.metadata.get('photometric_interpretation'),
                'image_shape': img.shape,
            }

            records.append(metrics)

        except Exception as e:
            print(f"Error analyzing image quality: {e}")

    df = pd.DataFrame(records)
    return df