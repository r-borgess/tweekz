import os
import cv2
import numpy as np
from src.image_processor import load_image, save_image, blackout_image, restore_image

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

def test_blackout_image():
    # Create a dummy image (100x100 white image)
    original_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # Apply blackout
    modified_image = blackout_image(original_image)
    # Check if all pixels are zero
    assert np.array_equal(modified_image, np.zeros((100, 100, 3), dtype=np.uint8))

def test_restore_image():
    # Path to a test image
    image_path = 'tests/test_images/Ramadan_Kyrie.jpg'
    # Load the image
    original_image = load_image(image_path)
    # Make some modification, here we just use blackout for the example
    blackout_image(original_image)
    # Attempt to restore
    restored_image = restore_image()
    # Check if restored image matches the original
    assert np.array_equal(restored_image, original_image)
