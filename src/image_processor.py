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

def apply_gamma_transformation(image, gamma):
    # Convert to float to avoid overflow or underflow during the power operation
    image_float = np.float32(image) / 255.0
    # Apply the gamma correction
    #corrected_img = np.power(image_float, gamma)
    corrected_img = np.power(image_float, gamma)
    # Scale back to original range
    corrected_img = np.uint8(corrected_img * 255)
    return corrected_img

def apply_contrast_stretch(image, r1, s1, r2, s2):
    """
    Apply contrast enhancement to an image using piecewise-linear transformation.

    Parameters:
    - image: Input image (numpy.ndarray).
    - r1, s1, r2, s2: Coordinates of the two points defining the piecewise-linear transformation.

    Returns:
    - numpy.ndarray: The contrast-enhanced image.
    """
    # Validate inputs (omitted for brevity)

    # Initialize the output image
    output_image = np.zeros_like(image)

    # Define the transformation function for each segment
    # Calculate slopes of each segment
    slope1 = s1 / r1 if r1 != 0 else 0
    slope2 = (s2 - s1) / (r2 - r1)
    slope3 = (255 - s2) / (255 - r2) if r2 != 255 else 0

    # Apply the transformation
    for i in range(256):
        if i < r1:
            trans_val = slope1 * i
        elif i < r2:
            trans_val = slope2 * (i - r1) + s1
        else:
            trans_val = slope3 * (i - r2) + s2
        output_image[image == i] = trans_val

    # Ensure the output values are valid
    np.clip(output_image, 0, 255, out=output_image)

    return output_image.astype(np.uint8)

def bit_plane_slicer(image, bit_plane_number):
    """
    Extract a specific bit plane from an image.

    Parameters:
    image (numpy.ndarray): The image to process.
    bit_plane (int): The specific bit plane to extract (0-7).

    Returns:
    numpy.ndarray: The extracted bit plane as a binary image.
    """
    # Convert image to grayscale if it is not
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.uint8)

    # Calculate the bit plane image
    bit_plane_image = (image >> bit_plane_number) & 1
    bit_plane_image *= 255  # Scale binary image to full intensity for visualization

    return bit_plane_image