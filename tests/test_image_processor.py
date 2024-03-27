import os
import cv2
import numpy as np
from src.image_processor import load_image, save_image

def test_load_image_success():
    # Assuming there's a test image in a known location
    image_path = 'tests/test_images/Ramadan_Kyrie.jpg'
    image = load_image(image_path)
    assert image is not None

def test_load_image_fail():
    # Test with a non-existent file
    image_path = 'non_existent.jpg'
    image = load_image(image_path)
    assert image is None

def test_save_image_success(tmp_path):
    # Use pytest's tmp_path fixture for a temporary file path
    save_path = tmp_path / "save_test.jpg"
    # Create a dummy image (100x100 black image)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    save_image(image, str(save_path))
    # Check if file exists and is not empty
    assert os.path.exists(save_path) and os.path.getsize(save_path) > 0
