import matplotlib.pyplot as plt
import pandas as pd
from dicom_handler import load_all_dicoms_from_folder
from preprocessing import static_preprocessing, adaptive_preprocessing, calculate_metrics

def compare_preprocessing_methods(dicom_folder: str, n_visualize: int = 3) -> pd.DataFrame:
    """
    Compare static vs adaptive preprocessing with comprehensive metrics.
    """
    records = []
    handlers = load_all_dicoms_from_folder(dicom_folder)
    
    if not handlers:
        print("No DICOM files loaded for comparison.")
        return pd.DataFrame()
    
    for i, handler in enumerate(handlers[:n_visualize * 2]):  # Process more than we visualize
        try:
            img = handler.get_pixel_array()
            if img is None:
                continue

            # Original metrics
            orig_metrics = calculate_metrics(img)

            # Apply preprocessing methods
            static_img = static_preprocessing(img)
            adaptive_img = adaptive_preprocessing(img)
            
            static_metrics = calculate_metrics(static_img)
            adaptive_metrics = calculate_metrics(adaptive_img)

            record = {
                'filename': handler.metadata.get('series_description', f'Image_{i}'),
                'modality': handler.metadata.get('modality', 'Unknown'),
            }
            
            # Add metrics for each method
            for prefix, metrics in [('orig', orig_metrics), 
                                  ('static', static_metrics), 
                                  ('adaptive', adaptive_metrics)]:
                for metric, value in metrics.items():
                    record[f'{metric}_{prefix}'] = value

            records.append(record)

            # Visualize first few examples
            if i < n_visualize:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                axes[1].imshow(static_img, cmap='gray', vmin=0, vmax=255)
                axes[1].set_title('Static Preprocessing')
                axes[1].axis('off')
                
                axes[2].imshow(adaptive_img, cmap='gray', vmin=0, vmax=255)
                axes[2].set_title('Adaptive Preprocessing')
                axes[2].axis('off')
                
                plt.suptitle(f"{record['filename']} - {record['modality']}")
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue

    df_comparison = pd.DataFrame(records)
    
    if not df_comparison.empty:
        # Print summary statistics
        print("\nPreprocessing Comparison Summary:")
        print("=" * 50)
        metrics = ['brightness', 'contrast', 'sharpness', 'noise']
        for metric in metrics:
            orig_col = f'{metric}_orig'
            static_col = f'{metric}_static'
            adaptive_col = f'{metric}_adaptive'
            
            if all(col in df_comparison.columns for col in [orig_col, static_col, adaptive_col]):
                orig_mean = df_comparison[orig_col].mean()
                static_mean = df_comparison[static_col].mean()
                adaptive_mean = df_comparison[adaptive_col].mean()
                
                print(f"{metric.title()}:")
                print(f"  Original: {orig_mean:.2f}")
                print(f"  Static: {static_mean:.2f} (Δ: {static_mean - orig_mean:+.2f})")
                print(f"  Adaptive: {adaptive_mean:.2f} (Δ: {adaptive_mean - orig_mean:+.2f})")
                print()
    
    return df_comparison