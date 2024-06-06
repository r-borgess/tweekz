import numpy as np
import pytest
import cv2
from src.image_processor import compute_fft_spectrum_and_phase

def test_fft_with_correct_image():
    # Provide a path to a valid test image
    magnitude, phase = compute_fft_spectrum_and_phase('C:/Users/rodri/tweekz/tests/test_images/load/Ramadan_Kyrie.jpg')
    
    # Check if the outputs are numpy arrays (as expected)
    assert isinstance(magnitude, np.ndarray), "Magnitude should be a numpy array"
    assert isinstance(phase, np.ndarray), "Phase should be a numpy array"
    
    # Optionally, you could also check the shapes or specific values if known
    img = cv2.imread('C:/Users/rodri/tweekz/tests/test_images/load/Ramadan_Kyrie.jpg', cv2.IMREAD_GRAYSCALE)
    assert magnitude.shape == img.shape, "Magnitude spectrum shape should match the image shape"
    assert phase.shape == img.shape, "Phase angle shape should match the image shape"

def test_fft_with_incorrect_image_path():
    # Test with an incorrect image path
    result = compute_fft_spectrum_and_phase('non_existent_image.jpg')
    assert result == "Error: Image not loaded. Check the file path.", "Function should return error message with non-existent image path"

def test_fft_with_non_image_file():
    # Test with a path to a non-image file (assuming a text file is available)
    result = compute_fft_spectrum_and_phase('example.txt')
    assert result == "Error: Image not loaded. Check the file path.", "Function should return error message with non-image file path"