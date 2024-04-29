import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def histogram_equalization(image):
    """
    Apply histogram equalization to the luminance channel of a color image and return the histograms before and after the process.

    Parameters:
    image (numpy.ndarray): The input color image in RGB format.

    Returns:
    tuple: A tuple containing the equalized image, histogram of the original luminance, and histogram of the equalized luminance.
    """
    # Convert RGB image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)

    # Calculate histogram of the original luminance (value channel)
    original_hist, _ = np.histogram(v.flatten(), bins=256, range=[0,256])

    # Normalize the histogram
    hist_normalized = original_hist / float(np.sum(original_hist))

    # Calculate the cumulative distribution function
    cdf = hist_normalized.cumsum()

    # Normalize the CDF to be in the range of 0-255
    cdf_normalized = np.floor(255 * cdf / cdf[-1]).astype('uint8')

    # Use the normalized CDF to set the new luminance values
    v_equalized = cdf_normalized[v.flatten()].reshape(v.shape)

    # Reconstruct the HSV image using the equalized luminance
    hsv_equalized = cv2.merge([h, s, v_equalized])

    # Convert the HSV image back to RGB
    equalized_image = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2RGB)

    # Calculate histogram of the equalized luminance
    equalized_hist, _ = np.histogram(v_equalized.flatten(), bins=256, range=[0,256])

    return equalized_image, original_hist, equalized_hist

def intensity_slicing_pseudocolor(image, num_ranges, cmap_name='viridis'):
    """
    Apply intensity slicing with automatic pseudocoloring to a grayscale image based on the selected color map.
    The intensity range is automatically divided into equal parts.

    Parameters:
    - image (numpy.ndarray): The input grayscale image.
    - num_ranges (int): Number of equal intensity ranges to create.
    - cmap_name (str): Name of the Matplotlib colormap to use.

    Returns:
    - numpy.ndarray: The pseudocolored image.
    """
    # Ensure the image is in grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a blank RGB image
    colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Calculate range boundaries
    step = 256 / num_ranges
    ranges = [(int(i * step), int((i + 1) * step) - 1) for i in range(num_ranges)]

    # Generate colors using the specified colormap
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(int(i * 255 / (num_ranges - 1)))[:3] for i in range(num_ranges)]
    colors = [(int(color[2]*255), int(color[1]*255), int(color[0]*255)) for color in colors]  # Convert to BGR format

    # Apply each color to the corresponding range
    for range_, color in zip(ranges, colors):
        mask = (image >= range_[0]) & (image <= range_[1])
        colored_image[mask] = color

    return colored_image