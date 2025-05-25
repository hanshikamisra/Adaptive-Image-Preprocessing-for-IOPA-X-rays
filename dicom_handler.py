import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from typing import Dict, Any, Tuple, Optional, List

class DICOMHandler:
    """
    Robust DICOM file handler for medical image processing.
    Supports pixel extraction, metadata parsing, and visualization.
    """

    def __init__(self):
        self.dicom_data = None
        self.pixel_array = None
        self.metadata = {}

    def read_dicom(self, file_path: str) -> bool:
        """
        Read and parse a DICOM file, extracting image data and metadata.

        Args:
            file_path (str): Path to DICOM file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Handle force=True to read potentially corrupted files
            self.dicom_data = pydicom.dcmread(file_path, force=True)
            
            # Check if pixel data exists
            if not hasattr(self.dicom_data, 'pixel_array'):
                print(f"✗ No pixel data in DICOM: {Path(file_path).name}")
                return False
                
            self.pixel_array = self.dicom_data.pixel_array
            
            # Handle different pixel data types and rescaling
            if hasattr(self.dicom_data, 'RescaleSlope') and hasattr(self.dicom_data, 'RescaleIntercept'):
                slope = float(self.dicom_data.RescaleSlope)
                intercept = float(self.dicom_data.RescaleIntercept)
                self.pixel_array = self.pixel_array * slope + intercept
            
            self._extract_metadata()
            print(f"✓ Successfully loaded DICOM: {Path(file_path).name}")
            return True
        except Exception as e:
            print(f"✗ Error reading DICOM {Path(file_path).name}: {e}")
            return False

    def _extract_metadata(self) -> None:
        """Extract and store relevant DICOM metadata with better error handling."""
        tags = {
            'modality': (0x0008, 0x0060),
            'photometric_interpretation': (0x0028, 0x0004),
            'rows': (0x0028, 0x0010),
            'columns': (0x0028, 0x0011),
            'pixel_spacing': (0x0028, 0x0030),
            'bits_allocated': (0x0028, 0x0100),
            'window_center': (0x0028, 0x1050),
            'window_width': (0x0028, 0x1051),
            'rescale_intercept': (0x0028, 0x1052),
            'rescale_slope': (0x0028, 0x1053),
            'manufacturer': (0x0008, 0x0070),
            'view_position': (0x0018, 0x5101),
            'study_date': (0x0008, 0x0020),
            'series_description': (0x0008, 0x103E)
        }

        self.metadata = {}
        for key, tag in tags.items():
            try:
                if tag in self.dicom_data:
                    value = self.dicom_data[tag].value
                    # Handle multi-valued elements
                    if isinstance(value, pydicom.multival.MultiValue):
                        value = list(value)
                    self.metadata[key] = value
                else:
                    self.metadata[key] = None
            except Exception as e:
                self.metadata[key] = None
                warnings.warn(f"Error extracting tag {tag} ({key}): {e}")

    def normalize_pixel_array(self) -> np.ndarray:
        """Normalize pixel array to 0-255 range for processing."""
        if self.pixel_array is None:
            return None
        
        img = self.pixel_array.astype(np.float32)
        
        # Handle different bit depths and ranges
        if img.min() == img.max():
            return np.zeros_like(img, dtype=np.uint8)
        
        # Normalize to 0-255
        img_norm = (img - img.min()) / (img.max() - img.min()) * 255
        return img_norm.astype(np.uint8)

    def visualize_image(self, figsize: Tuple[int, int] = (10, 8), cmap: str = 'gray') -> None:
        """Display the DICOM image using matplotlib."""
        if self.pixel_array is None:
            print("✗ No image data loaded.")
            return

        normalized_img = self.normalize_pixel_array()
        
        plt.figure(figsize=figsize)
        plt.imshow(normalized_img, cmap=cmap)
        plt.title(f"DICOM Image - {self.metadata.get('modality', 'Unknown')}")
        plt.axis('off')
        plt.colorbar(label='Pixel Intensity')
        plt.show()

    def get_pixel_array(self) -> Optional[np.ndarray]:
        """Return the normalized image pixel array."""
        return self.normalize_pixel_array()

    def get_metadata(self) -> Dict[str, Any]:
        """Return the extracted metadata."""
        return self.metadata

    def print_metadata_summary(self) -> None:
        """Print a summary of the metadata."""
        if not self.metadata:
            print("No metadata available.")
            return

        print("\nDICOM Metadata Summary")
        print("=" * 50)
        for key, value in self.metadata.items():
            if value is not None:
                print(f"{key:<25}: {value}")
        if self.pixel_array is not None:
            print(f"{'Pixel Array Shape':<25}: {self.pixel_array.shape}")
            print(f"{'Pixel Array Type':<25}: {self.pixel_array.dtype}")
            print(f"{'Pixel Range':<25}: [{self.pixel_array.min():.2f}, {self.pixel_array.max():.2f}]")
        print("=" * 50)


def load_all_dicoms_from_folder(folder_path: str, recursive: bool = True) -> List[DICOMHandler]:
    """
    Load all DICOM files from a given folder with better error handling.
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"✗ Folder does not exist: {folder_path}")
        return []
    
    # Look for various DICOM file extensions
    extensions = ["*.dcm", "*.DCM", "*.dicom", "*.DICOM"]
    dicom_files = []
    
    for ext in extensions:
        pattern = f"**/{ext}" if recursive else ext
        dicom_files.extend(list(folder.glob(pattern)))

    handlers = []
    failed_files = []

    for file in dicom_files:
        handler = DICOMHandler()
        if handler.read_dicom(str(file)):
            handlers.append(handler)
        else:
            failed_files.append(file.name)

    print(f"\n✓ Loaded {len(handlers)} valid DICOM files from: {folder_path}")
    if failed_files:
        print(f"✗ Failed to load {len(failed_files)} files: {failed_files[:5]}{'...' if len(failed_files) > 5 else ''}")
    
    return handlers