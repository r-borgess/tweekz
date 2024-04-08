import cv2
import numpy as np

original_image = None

def load_image(image_path):
    """
    Load an image from a specified file path.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    numpy.ndarray or None: The loaded image or None if loading failed.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    return image

def save_image(image, save_path):
    """
    Save an image to a specified file path.

    Parameters:
    image (numpy.ndarray): The image to be saved.
    save_path (str): The path where the image will be saved.
    """
    success = cv2.imwrite(save_path, image)
    if not success:
        print(f"Error: Could not save image to {save_path}")
        return False
    return True

'''
image_path = 'tests/test_images/Ramadan_Kyrie.jpg'
# Load the image
original_image = load_image(image_path)
'''

def blackout_image(image):
    """
    Set all pixels of the image to zero (black).

    Parameters:
    image (numpy.ndarray): The image to blackout.

    Returns:
    numpy.ndarray: The blacked out image.
    """
    global original_image
    # Store the original image if not already stored
    if original_image is None:
        original_image = image.copy()
    return np.zeros_like(image)


def restore_image():
    """
    Restore the image to its original state.

    Returns:
    numpy.ndarray or None: The original image, or None if there is no image to restore.
    """
    global original_image
    if original_image is not None:
        return original_image
    else:
        print("No image to restore.")
        return None