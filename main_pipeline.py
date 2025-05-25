from dicom_handler import load_all_dicoms_from_folder
from evaluation import evaluate_preprocessing_pipeline

def demo_dicom_pipeline(dicom_folder: str):
    """
    Demonstrate the complete DICOM processing pipeline.
    """
    print("Starting DICOM Processing Pipeline Demo")
    print("=" * 50)
    
    try:
        # Load a single DICOM for demonstration
        handlers = load_all_dicoms_from_folder(dicom_folder)
        
        if not handlers:
            print("No DICOM files found.")
            return
        
        # Show first image details
        handler = handlers[0]
        handler.print_metadata_summary()
        handler.visualize_image()
        
        # Run comprehensive evaluation
        evaluate_preprocessing_pipeline(dicom_folder)
        
        print("\nPipeline demonstration completed successfully!")
        
    except Exception as e:
        print(f"Error in pipeline demonstration: {e}")

if __name__ == "__main__":
    # Usage example:
    dicom_folder = input("Enter the path to your DICOM folder: ")
    demo_dicom_pipeline(dicom_folder)