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

def average_filter(image, kernel_size=3):
    """
    Manually apply an average filter to the image using numpy operations.

    Parameters:
    image (numpy.ndarray): The input image, which can be grayscale or color.
    kernel_size (int): The size of the kernel (both width and height) for the averaging filter.

    Returns:
    numpy.ndarray: The filtered image after applying the average filter manually.
    """
    # Check if the kernel size is odd to ensure a valid anchor point
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd to ensure a valid anchor point.")

    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Pad the image with border replicating the edge pixels to handle borders
    pad_size = kernel_size // 2
    if len(image.shape) == 3:  # Color image
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'edge')
    else:  # Grayscale image
        padded_image = np.pad(image, pad_size, 'edge')

    # Create an empty array to store the filtered image
    filtered_image = np.zeros_like(image)

    # Perform the averaging filter
    for y in range(height):
        for x in range(width):
            # Define the current region to take the average
            region = padded_image[y:y + kernel_size, x:x + kernel_size]
            # Compute the mean for the region and assign it to the corresponding pixel
            filtered_image[y, x] = np.mean(region, axis=(0, 1))

    return filtered_image

def min_filter(image, kernel_size):
    """
    Manually apply a minimum filter to the image using numpy operations.

    Parameters:
    image (numpy.ndarray): The input image, which can be grayscale or color.
    kernel_size (int): The size of the kernel (both width and height) for the minimum filter.

    Returns:
    numpy.ndarray: The filtered image after applying the minimum filter manually.
    """
    # Check if the kernel size is odd to ensure a valid anchor point
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd to ensure a valid anchor point.")

    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Pad the image with border replicating the edge pixels to handle borders
    pad_size = kernel_size // 2
    if len(image.shape) == 3:  # Color image
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'edge')
    else:  # Grayscale image
        padded_image = np.pad(image, pad_size, 'edge')

    # Create an empty array to store the filtered image
    filtered_image = np.zeros_like(image)

    # Perform the minimum filter
    for y in range(height):
        for x in range(width):
            # Define the current region to take the minimum
            region = padded_image[y:y + kernel_size, x:x + kernel_size]
            # Compute the minimum for the region and assign it to the corresponding pixel
            filtered_image[y, x] = np.min(region, axis=(0, 1))

    return filtered_image

def max_filter(image, kernel_size):
    """
    Manually apply a maximum filter to the image using numpy operations.

    Parameters:
    image (numpy.ndarray): The input image, which can be grayscale or color.
    kernel_size (int): The size of the kernel (both width and height) for the maximum filter.

    Returns:
    numpy.ndarray: The filtered image after applying the maximum filter manually.
    """
    # Check if the kernel size is odd to ensure a valid anchor point
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd to ensure a valid anchor point.")

    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Pad the image with border replicating the edge pixels to handle borders
    pad_size = kernel_size // 2
    if len(image.shape) == 3:  # Color image
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'edge')
    else:  # Grayscale image
        padded_image = np.pad(image, pad_size, 'edge')

    # Create an empty array to store the filtered image
    filtered_image = np.zeros_like(image)

    # Perform the maximum filter
    for y in range(height):
        for x in range(width):
            # Define the current region to take the minimum
            region = padded_image[y:y + kernel_size, x:x + kernel_size]
            # Compute the maximum for the region and assign it to the corresponding pixel
            filtered_image[y, x] = np.max(region, axis=(0, 1))

    return filtered_image

def median_filter(image, kernel_size):
    """
    Manually apply a median filter to the image using numpy operations.

    Parameters:
    image (numpy.ndarray): The input image, which can be grayscale or color.
    kernel_size (int): The size of the kernel (both width and height) for the median filter.

    Returns:
    numpy.ndarray: The filtered image after applying the median filter manually.
    """
    # Check if the kernel size is odd to ensure a valid anchor point
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd to ensure a valid anchor point.")

    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Pad the image to handle borders
    pad_size = kernel_size // 2
    if len(image.shape) == 3:  # Color image
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'reflect')
    else:  # Grayscale image
        padded_image = np.pad(image, pad_size, 'reflect')

    # Create an empty array to store the filtered image
    filtered_image = np.zeros_like(image)

    # Perform the median filter
    for y in range(height):
        for x in range(width):
            # Define the current region to take the median
            region = padded_image[y:y + kernel_size, x:x + kernel_size]
            # Compute the median for the region and assign it to the corresponding pixel
            # Handle color and grayscale separately
            if len(image.shape) == 3:
                for channel in range(3):
                    filtered_image[y, x, channel] = np.median(region[:, :, channel])
            else:
                filtered_image[y, x] = np.median(region)

    return filtered_image

def apply_laplacian_kernel(image, kernel):
    img_height, img_width = image.shape
    laplacian_image = np.zeros_like(image, dtype=float)

    # Apply the Laplacian kernel
    for y in range(1, img_height - 1):
        for x in range(1, img_width - 1):
            laplacian_image[y, x] = np.sum(kernel * image[y-1:y+2, x-1:x+2])

    return laplacian_image

def laplacian_filter(image):
    # Define the Laplacian kernel
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    w = -1.2  # Weight scalar for the Laplacian

    if len(image.shape) == 3:
        # Process each channel separately if it's a color image
        channels = cv2.split(image)
    else:
        # Make it a list of one channel if it's grayscale
        channels = [image]

    sharpened_channels = []
    laplacian_channels = []
    contrast_channels = []

    for channel in channels:
        # Apply the Laplacian kernel using filter2D
        laplacian_channel = cv2.filter2D(channel, -1, kernel * w)

        # Contrast stretching
        min_val, max_val = laplacian_channel.min(), laplacian_channel.max()
        contrast_channel = (laplacian_channel - min_val) * (255 / (max_val - min_val))
        contrast_channel = np.clip((contrast_channel * 0.5) + 128, 0, 255).astype(np.uint8)

        # Sharpening the image
        sharpened_channel = cv2.add(channel, laplacian_channel)
        sharpened_channel = np.clip(sharpened_channel, 0, 255).astype(np.uint8)

        sharpened_channels.append(sharpened_channel)
        laplacian_channels.append(laplacian_channel)
        contrast_channels.append(contrast_channel)

    # Merge channels back if it was a color image
    if len(image.shape) == 3:
        sharpened_image = cv2.merge(sharpened_channels)
        laplacian_image = cv2.merge([np.clip(ch, 0, 255).astype(np.uint8) for ch in laplacian_channels])
        contrast_stretched = cv2.merge(contrast_channels)
    else:
        sharpened_image = sharpened_channels[0]
        laplacian_image = laplacian_channels[0].astype(np.uint8)
        contrast_stretched = contrast_channels[0]

    return sharpened_image, laplacian_image, contrast_stretched

def compute_fft_spectrum_and_phase(image):
    if image is None:
        return "Error: Image not loaded. Check the file path."
    
    # Check if the image is loaded properly
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the 2D Fourier Transform of the image
    f = np.fft.fftshift(np.fft.fft2(image))
    
    # Compute the magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f) + 1)  # Adding 1 to avoid log(0)
    
    # Compute the phase angle
    phase_angle = np.angle(f)
    
    return magnitude_spectrum, phase_angle

def compute_inverse_fft(magnitude_spectrum, phase_angle):
    # Recompose the complex spectrum from the magnitude and phase
    magnitude = np.exp(magnitude_spectrum / 20) - 1  # Inverting the log and scale transformation
    complex_spectrum = magnitude * (np.cos(phase_angle) + 1j * np.sin(phase_angle))
    
    # Shift the zero frequency component back to the original configuration
    f_ishift = np.fft.ifftshift(complex_spectrum)
    
    # Compute the Inverse FFT
    img_back = np.fft.ifft2(f_ishift)
    
    # Get the real part of the image
    img_back = np.real(img_back)
    
    # Normalize the image to 8-bit scale (0-255)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    
    return img_back

def high_pass(image, radius):
    """
    Applies a high pass filter to the input image.

    Parameters:
    - image: Input image in which the high pass filter is to be applied.
    - radius: The radius size for the high pass filter.

    Returns:
    - Filtered image with the high pass filter applied.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the image to the frequency domain using Fourier Transform
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Get the image dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Create a mask with the high pass filter
    mask = np.ones((rows, cols, 2), np.uint8)
    r = radius
    center = (crow, ccol)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
    mask[mask_area] = 0

    # Apply the mask to the DFT shifted image
    fshift = dft_shift * mask

    # Inverse Fourier Transform to get the image back in spatial domain
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize the result to the range [0, 255]
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    return img_back

def low_pass(image, radius):
    """
    Applies a low pass filter to the input image.

    Parameters:
    - image: Input image in which the low pass filter is to be applied.
    - radius: The radius size for the low pass filter.

    Returns:
    - Filtered image with the low pass filter applied.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the image to the frequency domain using Fourier Transform
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Get the image dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Create a mask with the low pass filter
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = radius
    center = (crow, ccol)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
    mask[mask_area] = 1

    # Apply the mask to the DFT shifted image
    fshift = dft_shift * mask

    # Inverse Fourier Transform to get the image back in spatial domain
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize the result to the range [0, 255]
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    return img_back