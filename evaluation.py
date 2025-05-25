import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from quality_analysis import analyze_image_quality
from comparison import compare_preprocessing_methods

def calculate_psnr_safe(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR with error handling."""
    try:
        if img1.shape != img2.shape:
            return 0.0
        return float(psnr(img1, img2, data_range=255))
    except Exception:
        return 0.0

def calculate_ssim_safe(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM with error handling."""
    try:
        if img1.shape != img2.shape:
            return 0.0
        return float(ssim(img1, img2, data_range=255))
    except Exception:
        return 0.0

def evaluate_preprocessing_pipeline(dicom_folder: str) -> None:
    """
    Comprehensive evaluation of preprocessing methods.
    """
    print("Evaluating DICOM preprocessing pipeline...")
    
    # Load and analyze images
    df_quality = analyze_image_quality(dicom_folder)
    if df_quality.empty:
        print("No images found for analysis.")
        return
    
    print(f"Analyzed {len(df_quality)} images")
    
    # Compare preprocessing methods
    df_comparison = compare_preprocessing_methods(dicom_folder, n_visualize=3)
    
    if not df_comparison.empty:
        # Visualize quality metrics distribution
        metrics = ['brightness', 'contrast_std', 'sharpness_lapvar', 'noise_estimate']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if metric in df_quality.columns:
                sns.histplot(df_quality[metric], bins=20, kde=True, ax=axes[i])
                axes[i].set_title(f"{metric.replace('_', ' ').title()} Distribution")
        
        plt.tight_layout()
        plt.show()
        
        # Save results
        df_quality.to_csv("dicom_quality_analysis.csv", index=False)
        df_comparison.to_csv("preprocessing_comparison.csv", index=False)
        print("Results saved to CSV files.")