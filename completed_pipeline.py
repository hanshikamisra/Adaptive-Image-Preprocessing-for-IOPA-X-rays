#!/usr/bin/env python3
"""
Adaptive Image Preprocessing Comparison Tool for IOPA X-rays

This tool performs side-by-side comparison of original vs adaptive preprocessing
for both DCM and RVG dental X-ray images. It includes multiple adaptive techniques
specifically designed for intraoral periapical (IOPA) radiographs.

Features:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Adaptive Gamma Correction
- Unsharp Masking
- Noise Reduction
- Edge Enhancement
- Automatic intensity windowing

Author: Adaptive Image Preprocessing Pipeline
Version: 2.0
Repository: https://github.com/hanshikamisra/Adaptive-Image-Preprocessing-for-IOPA-X-rays
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import os
from skimage import exposure, filters, restoration, morphology
from scipy import ndimage
import warnings

# Import our DICOM handler
from dicom_handler import DICOMHandler, load_all_dicoms_from_folder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=UserWarning)


class AdaptivePreprocessor:
    """
    Advanced adaptive preprocessing specifically designed for dental IOPA X-rays.
    Implements multiple enhancement techniques optimized for radiographic images.
    """
    
    def __init__(self):
        self.processing_params = {
            'clahe_clip_limit': 3.0,
            'clahe_tile_size': (8, 8),
            'gamma_correction': True,
            'unsharp_strength': 1.5,
            'noise_reduction': True,
            'edge_enhancement': True,
            'intensity_windowing': True
        }
    
    def adaptive_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive gamma correction based on image statistics.
        Optimized for dental X-ray enhancement.
        """
        # Calculate optimal gamma based on image histogram
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        
        # Find the median intensity
        cumsum = np.cumsum(hist)
        median_idx = np.where(cumsum >= cumsum[-1] // 2)[0][0]
        median_intensity = median_idx / 255.0
        
        # Calculate adaptive gamma
        if median_intensity < 0.3:
            # Dark image - increase gamma to brighten
            gamma = 0.6
        elif median_intensity > 0.7:
            # Bright image - decrease gamma to darken
            gamma = 1.4
        else:
            # Well-exposed image - mild adjustment
            gamma = 1.0 / (median_intensity + 0.1)
        
        # Apply gamma correction
        gamma_corrected = np.power(image / 255.0, gamma) * 255.0
        return np.clip(gamma_corrected, 0, 255).astype(np.uint8)
    
    def adaptive_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        with parameters optimized for dental radiographs.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.processing_params['clahe_clip_limit'],
            tileGridSize=self.processing_params['clahe_tile_size']
        )
        return clahe.apply(image)
    
    def unsharp_masking(self, image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """
        Apply unsharp masking for edge enhancement.
        Particularly effective for enhancing tooth structures and root details.
        """
        # Create Gaussian blurred version
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        
        # Create unsharp mask
        unsharp = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        
        return np.clip(unsharp, 0, 255).astype(np.uint8)
    
    def adaptive_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive noise reduction while preserving important structures.
        Uses bilateral filtering to maintain edges while reducing noise.
        """
        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Apply a mild median filter for additional noise reduction
        denoised = cv2.medianBlur(denoised, 3)
        
        return denoised
    
    def edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance edges and fine structures important in dental radiographs.
        """
        # Apply Laplacian sharpening
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpened = image - 0.3 * laplacian
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def adaptive_intensity_windowing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive intensity windowing to optimize contrast for dental structures.
        """
        # Calculate percentiles for robust windowing
        p2, p98 = np.percentile(image, (2, 98))
        
        # Apply intensity rescaling
        windowed = exposure.rescale_intensity(image, in_range=(p2, p98))
        
        return (windowed * 255).astype(np.uint8)
    
    def process_image(self, image: np.ndarray, custom_params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Apply complete adaptive preprocessing pipeline to input image.
        
        Args:
            image: Input grayscale image (0-255)
            custom_params: Optional custom processing parameters
            
        Returns:
            Dictionary containing original and all processed versions
        """
        if custom_params:
            self.processing_params.update(custom_params)
        
        results = {'original': image.copy()}
        current_image = image.copy()
        
        # Step 1: Adaptive intensity windowing
        if self.processing_params['intensity_windowing']:
            current_image = self.adaptive_intensity_windowing(current_image)
            results['windowed'] = current_image.copy()
        
        # Step 2: Noise reduction
        if self.processing_params['noise_reduction']:
            current_image = self.adaptive_noise_reduction(current_image)
            results['denoised'] = current_image.copy()
        
        # Step 3: Adaptive gamma correction
        if self.processing_params['gamma_correction']:
            current_image = self.adaptive_gamma_correction(current_image)
            results['gamma_corrected'] = current_image.copy()
        
        # Step 4: CLAHE enhancement
        current_image = self.adaptive_clahe(current_image)
        results['clahe_enhanced'] = current_image.copy()
        
        # Step 5: Edge enhancement
        if self.processing_params['edge_enhancement']:
            current_image = self.edge_enhancement(current_image)
            results['edge_enhanced'] = current_image.copy()
        
        # Step 6: Unsharp masking (final sharpening)
        current_image = self.unsharp_masking(current_image, 
                                           self.processing_params['unsharp_strength'])
        results['final_processed'] = current_image.copy()
        
        return results
    
    def calculate_image_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """
        Calculate quality metrics comparing original and processed images.
        """
        # Contrast enhancement ratio
        original_contrast = np.std(original)
        processed_contrast = np.std(processed)
        contrast_ratio = processed_contrast / (original_contrast + 1e-8)
        
        # Entropy (information content)
        def calculate_entropy(img):
            hist, _ = np.histogram(img, bins=256, range=(0, 256))
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]  # Remove zero entries
            return -np.sum(hist * np.log2(hist))
        
        original_entropy = calculate_entropy(original)
        processed_entropy = calculate_entropy(processed)
        
        # Edge strength improvement
        original_edges = cv2.Canny(original, 50, 150)
        processed_edges = cv2.Canny(processed, 50, 150)
        
        original_edge_strength = np.sum(original_edges) / 255.0
        processed_edge_strength = np.sum(processed_edges) / 255.0
        edge_improvement = processed_edge_strength / (original_edge_strength + 1e-8)
        
        return {
            'contrast_ratio': contrast_ratio,
            'original_entropy': original_entropy,
            'processed_entropy': processed_entropy,
            'entropy_ratio': processed_entropy / (original_entropy + 1e-8),
            'edge_improvement': edge_improvement
        }


class ComparisonVisualizer:
    """
    Create comprehensive visual comparisons of original vs processed images.
    """
    
    def __init__(self, output_dir: str = "comparison_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "individual_comparisons").mkdir(exist_ok=True)
        (self.output_dir / "batch_comparisons").mkdir(exist_ok=True)
        (self.output_dir / "metrics_reports").mkdir(exist_ok=True)
    
    def create_side_by_side_comparison(self, original: np.ndarray, processed: np.ndarray,
                                     title: str, metrics: Dict[str, float],
                                     file_info: Dict[str, Any]) -> plt.Figure:
        """
        Create a detailed side-by-side comparison plot.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Adaptive Preprocessing Comparison - {title}', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Processed image
        axes[0, 1].imshow(processed, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title('Adaptive Processed', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Difference image
        diff_image = cv2.absdiff(processed, original)
        im_diff = axes[0, 2].imshow(diff_image, cmap='hot', vmin=0, vmax=255)
        axes[0, 2].set_title('Enhancement Difference', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im_diff, ax=axes[0, 2], shrink=0.8)
        
        # Histograms
        axes[1, 0].hist(original.flatten(), bins=50, alpha=0.7, color='blue', label='Original')
        axes[1, 0].hist(processed.flatten(), bins=50, alpha=0.7, color='red', label='Processed')
        axes[1, 0].set_title('Intensity Histograms', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Edge comparison
        original_edges = cv2.Canny(original, 50, 150)
        processed_edges = cv2.Canny(processed, 50, 150)
        
        axes[1, 1].imshow(original_edges, cmap='gray')
        axes[1, 1].set_title('Original Edges', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(processed_edges, cmap='gray')
        axes[1, 2].set_title('Enhanced Edges', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Add metrics text
        metrics_text = f"""
        Quality Metrics:
        • Contrast Ratio: {metrics['contrast_ratio']:.2f}
        • Entropy Ratio: {metrics['entropy_ratio']:.2f}
        • Edge Improvement: {metrics['edge_improvement']:.2f}
        
        File Information:
        • Type: {file_info.get('file_type', 'Unknown')}
        • Modality: {file_info.get('modality', 'Unknown')}
        • Dimensions: {file_info.get('dimensions', 'Unknown')}
        • Manufacturer: {file_info.get('manufacturer', 'Unknown')}
        """
        
        fig.text(0.02, 0.02, metrics_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                verticalalignment='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_processing_pipeline_visualization(self, processing_results: Dict[str, np.ndarray],
                                               filename: str) -> plt.Figure:
        """
        Create a visualization showing each step of the processing pipeline.
        """
        steps = ['original', 'windowed', 'denoised', 'gamma_corrected', 
                'clahe_enhanced', 'edge_enhanced', 'final_processed']
        
        available_steps = [step for step in steps if step in processing_results]
        n_steps = len(available_steps)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Adaptive Processing Pipeline - {filename}', fontsize=16, fontweight='bold')
        
        for i, step in enumerate(available_steps[:8]):  # Limit to 8 steps
            row = i // 4
            col = i % 4
            
            if row < 2 and col < 4:
                axes[row, col].imshow(processing_results[step], cmap='gray', vmin=0, vmax=255)
                axes[row, col].set_title(step.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_steps, 8):
            row = i // 4
            col = i % 4
            if row < 2 and col < 4:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_comparison(self, fig: plt.Figure, filename: str, subdir: str = "individual_comparisons"):
        """Save comparison figure to file."""
        output_path = self.output_dir / subdir / f"{filename}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"Saved comparison: {output_path}")


class BatchComparisonProcessor:
    """
    Process multiple DICOM/RVG files and generate comprehensive comparison reports.
    """
    
    def __init__(self, output_dir: str = "comparison_results"):
        self.preprocessor = AdaptivePreprocessor()
        self.visualizer = ComparisonVisualizer(output_dir)
        self.output_dir = Path(output_dir)
        self.results_summary = []
    
    def process_single_file(self, handler: DICOMHandler) -> Dict[str, Any]:
        """
        Process a single DICOM/RVG file and generate comparisons.
        """
        try:
            # Get image data and metadata
            pixel_array = handler.get_pixel_array(normalized=True)
            metadata = handler.get_metadata()
            
            if pixel_array is None:
                logger.error(f"No pixel data available for {metadata.get('file_name', 'Unknown')}")
                return None
            
            # Apply adaptive preprocessing
            processing_results = self.preprocessor.process_image(pixel_array)
            
            # Calculate metrics
            metrics = self.preprocessor.calculate_image_metrics(
                processing_results['original'], 
                processing_results['final_processed']
            )
            
            # Prepare file information
            file_info = {
                'file_name': metadata.get('file_name', 'Unknown'),
                'file_type': 'RVG' if handler.is_rvg_file() else 'DCM',
                'modality': metadata.get('modality', 'Unknown'),
                'dimensions': f"{pixel_array.shape[1]}x{pixel_array.shape[0]}",
                'manufacturer': metadata.get('manufacturer', 'Unknown'),
                'bits_allocated': metadata.get('bits_allocated', 'Unknown')
            }
            
            # Create visualizations
            filename_base = Path(file_info['file_name']).stem
            
            # Side-by-side comparison
            comparison_fig = self.visualizer.create_side_by_side_comparison(
                processing_results['original'],
                processing_results['final_processed'],
                f"{filename_base} ({file_info['file_type']})",
                metrics,
                file_info
            )
            self.visualizer.save_comparison(comparison_fig, f"{filename_base}_comparison")
            
            # Pipeline visualization
            pipeline_fig = self.visualizer.create_processing_pipeline_visualization(
                processing_results, filename_base
            )
            self.visualizer.save_comparison(pipeline_fig, f"{filename_base}_pipeline")
            
            # Store results for summary
            result_summary = {
                **file_info,
                **metrics,
                'processing_time': datetime.now().isoformat()
            }
            
            self.results_summary.append(result_summary)
            
            return result_summary
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return None
    
    def process_folder(self, folder_path: str, max_files: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all DICOM/RVG files in a folder and generate comprehensive report.
        """
        logger.info(f"Starting batch processing of folder: {folder_path}")
        
        # Load all DICOM files
        handlers = load_all_dicoms_from_folder(folder_path)
        
        if not handlers:
            logger.error("No valid DICOM files found")
            return {"error": "No valid DICOM files found"}
        
        # Limit number of files if specified
        if max_files:
            handlers = handlers[:max_files]
            logger.info(f"Processing limited to {max_files} files")
        
        # Process each file
        successful_processes = 0
        failed_processes = 0
        
        for i, handler in enumerate(handlers):
            logger.info(f"Processing file {i+1}/{len(handlers)}: {handler.get_metadata().get('file_name', 'Unknown')}")
            
            result = self.process_single_file(handler)
            if result:
                successful_processes += 1
            else:
                failed_processes += 1
        
        # Generate summary report
        summary_report = self.generate_summary_report(folder_path, successful_processes, failed_processes)
        
        logger.info(f"Batch processing completed. Success: {successful_processes}, Failed: {failed_processes}")
        
        return summary_report
    
    def generate_summary_report(self, folder_path: str, successful: int, failed: int) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of all processed files.
        """
        if not self.results_summary:
            return {"error": "No results to summarize"}
        
        # Calculate statistics
        dcm_files = [r for r in self.results_summary if r['file_type'] == 'DCM']
        rvg_files = [r for r in self.results_summary if r['file_type'] == 'RVG']
        
        contrast_ratios = [r['contrast_ratio'] for r in self.results_summary]
        entropy_ratios = [r['entropy_ratio'] for r in self.results_summary]
        edge_improvements = [r['edge_improvement'] for r in self.results_summary]
        
        summary = {
            'processing_info': {
                'folder_path': folder_path,
                'total_files': len(self.results_summary),
                'successful_processes': successful,
                'failed_processes': failed,
                'dcm_files': len(dcm_files),
                'rvg_files': len(rvg_files),
                'processing_timestamp': datetime.now().isoformat()
            },
            'quality_metrics': {
                'avg_contrast_ratio': np.mean(contrast_ratios),
                'avg_entropy_ratio': np.mean(entropy_ratios),
                'avg_edge_improvement': np.mean(edge_improvements),
                'contrast_ratio_std': np.std(contrast_ratios),
                'entropy_ratio_std': np.std(entropy_ratios),
                'edge_improvement_std': np.std(edge_improvements)
            },
            'file_details': self.results_summary
        }
        
        # Save summary to JSON
        summary_path = self.output_dir / "metrics_reports" / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create summary visualization
        self.create_summary_visualization(summary)
        
        logger.info(f"Summary report saved to: {summary_path}")
        
        return summary
    
    def create_summary_visualization(self, summary: Dict[str, Any]):
        """
        Create visualization of processing summary statistics.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Batch Processing Summary - Adaptive Preprocessing Results', 
                    fontsize=16, fontweight='bold')
        
        # File type distribution
        dcm_count = summary['processing_info']['dcm_files']
        rvg_count = summary['processing_info']['rvg_files']
        
        axes[0, 0].pie([dcm_count, rvg_count], labels=['DCM', 'RVG'], autopct='%1.1f%%', 
                      colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('File Type Distribution')
        
        # Quality metrics
        metrics = summary['quality_metrics']
        metric_names = ['Contrast\nRatio', 'Entropy\nRatio', 'Edge\nImprovement']
        metric_values = [metrics['avg_contrast_ratio'], 
                        metrics['avg_entropy_ratio'], 
                        metrics['avg_edge_improvement']]
        metric_stds = [metrics['contrast_ratio_std'],
                      metrics['entropy_ratio_std'],
                      metrics['edge_improvement_std']]
        
        axes[0, 1].bar(metric_names, metric_values, yerr=metric_stds, 
                      capsize=5, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 1].set_title('Average Quality Metrics')
        axes[0, 1].set_ylabel('Improvement Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Processing success rate
        total_attempted = summary['processing_info']['successful_processes'] + summary['processing_info']['failed_processes']
        success_rate = summary['processing_info']['successful_processes'] / total_attempted * 100
        
        axes[0, 2].pie([success_rate, 100-success_rate], 
                      labels=[f'Success ({success_rate:.1f}%)', f'Failed ({100-success_rate:.1f}%)'],
                      colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        axes[0, 2].set_title('Processing Success Rate')
        
        # Individual file metrics (if not too many files)
        if len(summary['file_details']) <= 20:
            file_names = [f['file_name'][:15] + '...' if len(f['file_name']) > 15 else f['file_name'] 
                         for f in summary['file_details']]
            contrast_values = [f['contrast_ratio'] for f in summary['file_details']]
            
            axes[1, 0].barh(range(len(file_names)), contrast_values)
            axes[1, 0].set_yticks(range(len(file_names)))
            axes[1, 0].set_yticklabels(file_names, fontsize=8)
            axes[1, 0].set_title('Contrast Improvement by File')
            axes[1, 0].set_xlabel('Contrast Ratio')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # Show histogram of contrast ratios
            contrast_values = [f['contrast_ratio'] for f in summary['file_details']]
            axes[1, 0].hist(contrast_values, bins=20, alpha=0.7, color='skyblue')
            axes[1, 0].set_title('Distribution of Contrast Improvements')
            axes[1, 0].set_xlabel('Contrast Ratio')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Entropy improvements
        entropy_values = [f['entropy_ratio'] for f in summary['file_details']]
        axes[1, 1].hist(entropy_values, bins=20, alpha=0.7, color='lightgreen')
        axes[1, 1].set_title('Distribution of Entropy Improvements')
        axes[1, 1].set_xlabel('Entropy Ratio')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Edge improvements
        edge_values = [f['edge_improvement'] for f in summary['file_details']]
        axes[1, 2].hist(edge_values, bins=20, alpha=0.7, color='salmon')
        axes[1, 2].set_title('Distribution of Edge Improvements')
        axes[1, 2].set_xlabel('Edge Improvement Ratio')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save summary visualization
        summary_viz_path = self.output_dir / "batch_comparisons" / "processing_summary.png"
        fig.savefig(summary_viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        logger.info(f"Summary visualization saved to: {summary_viz_path}")


def main():
    """
    Main function for running the adaptive preprocessing comparison tool.
    """
    print("Adaptive Image Preprocessing for IOPA X-rays - Comparison Tool")
    print("=" * 80)
    print("Repository: https://github.com/hanshikamisra/Adaptive-Image-Preprocessing-for-IOPA-X-rays")
    print("=" * 80)
    
    # Configuration
    input_folder = input("Enter path to DICOM/RVG folder (or press Enter for 'sample_images'): ") or "sample_images"
    output_folder = input("Enter output folder (or press Enter for 'comparison_results'): ") or "comparison_results"
    max_files = input("Enter max files to process (or press Enter for all): ")
    max_files = int(max_files) if max_files.isdigit() else None
    
    # Initialize processor
    processor = BatchComparisonProcessor(output_folder)
    
    # Process folder
    try:
        results = processor.process_folder(input_folder, max_files)
        
        if "error" not in results:
            print(f"\nProcessing completed successfully!")
            print(f"Processed {results['processing_info']['total_files']} files")
            print(f"Results saved to: {output_folder}")
            print(f"DCM files: {results['processing_info']['dcm_files']}")
            print(f"RVG files: {results['processing_info']['rvg_files']}")
            print(f"\nAverage Quality Improvements:")
            print(f"   • Contrast: {results['quality_metrics']['avg_contrast_ratio']:.2f}x")
            print(f"   • Entropy: {results['quality_metrics']['avg_entropy_ratio']:.2f}x")
            print(f"   • Edge Definition: {results['quality_metrics']['avg_edge_improvement']:.2f}x")
            
            print(f"\nCheck these folders for results:")
            print(f"   • Individual comparisons: {output_folder}/individual_comparisons/")
            print(f"   • Batch summaries: {output_folder}/batch_comparisons/")
            print(f"   • Metrics reports: {output_folder}/metrics_reports/")
        else:
            print(f"Error: {results['error']}")
            
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()