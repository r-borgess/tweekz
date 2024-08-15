# tweekz

## Overview

This tool is designed as a platform for demonstrating and executing image processing operations. It features a graphical user interface for ease of use, supporting basic functionalities such as loading, displaying, modifying, and saving images. The application is developed in Python, emphasizing modularity and extensibility to facilitate future additions of more complex image processing capabilities.

## Features

- **Intensity Transformations:**
  - Blackout
  - Gamma Transform
  - Contrast Stretch
  - Bit Plane Extraction
  - Histogram Equalization
  - Intensity Slicing

- **Spatial Domain Processing:**
  - **Smoothing:**
    - Average Filter
  - **Order-Statistics:**
    - Min Filter
    - Max Filter
    - Median Filter
  - **Sharpening:**
    - Laplacian Filter

- **Frequency Domain Processing:**
  - Fourier Transform (FT)
  - **Basic Filters:**
    - Low Pass Filter
    - High Pass Filter
    - Notch Reject Filter

- **Image Restoration:**
  - **Noise Effects:**
    - Generate Noise: Gaussian, Salt and Pepper
    - Apply Noise: Gaussian, Salt and Pepper
  - **Filtering:**
    - Geometric Mean Filter
    - Alpha-Trimmed Mean Filter

- **Morphological Operations:**
  - Erosion
  - Dilation
  - Opening
  - Closing

- **Image Compression:**
  - Huffman Coding

- **Image Segmentation:**
  - Region Growing
  - Canny Edge Detection
  - Chain Code
  - Skeletonization

- **Feature Detection:**
  - Harris Corner Detection
  - MSER (Maximally Stable Extremal Regions)
  - Template Matching

## Getting Started

### Prerequisites

- Python 3.8 or later

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/r-borgess/tweekz.git
   ```
2. Navigate to the project directory:
   ```
   cd tweekz
   ```
3. Create and activate a virtual environment:
   - **Windows:**
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - **Unix/MacOS:**
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```
4. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

Execute the main script:
```
python src/main.py
```

## How to Use

- Use the GUI to interact with the application's features: load, modify, save images, and reset modifications.

## Contributing

Contributions are welcome. Please feel free to contribute by opening a pull request or filing an issue.

## License

This project is available under the MIT License. See the LICENSE file for more details.
