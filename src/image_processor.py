import cv2

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
