# DICOM Medical Image Processing Pipeline

A comprehensive Python pipeline for processing DICOM medical images with advanced preprocessing techniques and quality analysis.

## Features

- **DICOM File Handling**: Robust loading and parsing of DICOM files with metadata extraction
- **Quality Analysis**: Comprehensive image quality metrics including brightness, contrast, sharpness, and noise estimation
- **Preprocessing Methods**: Both static and adaptive preprocessing pipelines
- **Comparison Tools**: Side-by-side comparison of preprocessing methods with quantitative metrics
- **Evaluation Framework**: Complete pipeline evaluation with visualization and CSV export

## File Structure

```
dicom-processing-pipeline/
├── dicom_handler.py        # Core DICOM file handling and metadata extraction
├── quality_analysis.py     # Image quality metrics and analysis functions
├── preprocessing.py        # Static and adaptive preprocessing methods
├── comparison.py          # Preprocessing method comparison tools
├── evaluation.py          # Pipeline evaluation and visualization
├── main_pipeline.py       # Main demonstration script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd dicom-processing-pipeline
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline demonstration:

```python
python main_pipeline.py
```

### Individual Module Usage

#### DICOM Handler
```python
from dicom_handler import DICOMHandler, load_all_dicoms_from_folder

# Load a single DICOM file
handler = DICOMHandler()
if handler.read_dicom('path/to/dicom/file.dcm'):
    handler.print_metadata_summary()
    handler.visualize_image()

# Load all DICOM files from a folder
handlers = load_all_dicoms_from_folder('path/to/dicom/folder')
```

#### Quality Analysis
```python
from quality_analysis import analyze_image_quality

# Analyze image quality for all DICOM files
df_quality = analyze_image_quality('path/to/dicom/folder')
print(df_quality.head())
```

#### Preprocessing
```python
from preprocessing import static_preprocessing, adaptive_preprocessing
from dicom_handler import DICOMHandler

handler = DICOMHandler()
handler.read_dicom('path/to/dicom/file.dcm')
img = handler.get_pixel_array()

# Apply static preprocessing
static_processed = static_preprocessing(img)

# Apply adaptive preprocessing
adaptive_processed = adaptive_preprocessing(img)
```

#### Comparison and Evaluation
```python
from comparison import compare_preprocessing_methods
from evaluation import evaluate_preprocessing_pipeline

# Compare preprocessing methods
df_comparison = compare_preprocessing_methods('path/to/dicom/folder')

# Full pipeline evaluation
evaluate_preprocessing_pipeline('path/to/dicom/folder')
```

## Key Components

### DICOMHandler Class
- Robust DICOM file reading with error handling
- Metadata extraction for 14+ key DICOM tags
- Pixel array normalization and visualization
- Support for various DICOM file formats and corrupted files

### Quality Analysis Functions
- **Brightness**: Mean pixel intensity
- **Contrast**: Standard deviation and RMS contrast
- **Sharpness**: Laplacian variance and Tenengrad methods
- **Noise Estimation**: Local standard deviation in flat regions

### Preprocessing Methods

#### Static Preprocessing
- Histogram equalization
- Sharpening filter
- Basic Gaussian denoising

#### Adaptive Preprocessing
- Adaptive CLAHE based on image contrast
- Dynamic sharpening based on image sharpness metrics
- Noise-adaptive bilateral filtering

### Evaluation Framework
- Quantitative comparison of preprocessing methods
- Quality metrics visualization
- CSV export for further analysis
- Side-by-side visual comparisons

## Output Files

The pipeline generates several output files:
- `dicom_quality_analysis.csv`: Quality metrics for all processed images
- `preprocessing_comparison.csv`: Comparison results between methods

## Dependencies

- **pydicom**: DICOM file handling
- **numpy**: Numerical operations
- **opencv-python**: Image processing
- **matplotlib**: Visualization
- **pandas**: Data manipulation
- **seaborn**: Statistical visualization
- **scipy**: Scientific computing
- **scikit-image**: Image processing metrics
- **scikit-learn**: Machine learning utilities

## Error Handling

The pipeline includes comprehensive error handling:
- Graceful handling of corrupted DICOM files
- Missing metadata handling
- Image processing error recovery
- Detailed error reporting and logging

## Performance Considerations

- Optimized for batch processing of multiple DICOM files
- Memory-efficient handling of large image arrays
- Parallel processing capabilities for folder-based operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


