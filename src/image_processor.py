import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from scipy.stats import entropy as scipy_entropy
from PIL import Image
from collections import defaultdict, Counter
import heapq

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
    #print(phase_angle)
    
    return magnitude_spectrum, phase_angle, f

def compute_inverse_fft(magnitude_spectrum, phase_angle=None):
    # Recompose the complex spectrum from the magnitude and phase
    magnitude = np.exp(magnitude_spectrum / 20) - 1  # Inverting the log and scale transformation
    
    if phase_angle is not None:
        complex_spectrum = magnitude * (np.cos(phase_angle) + 1j * np.sin(phase_angle))
    else:
        complex_spectrum = magnitude
    
    # Shift the zero frequency component back to the original configuration
    f_ishift = np.fft.ifftshift(complex_spectrum)
    
    # Compute the Inverse FFT
    img_back = np.fft.ifft2(f_ishift)
    
    # Get the real part of the image
    img_back = np.real(img_back)
    
    # Normalize the image to 8-bit scale (0-255)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    #print("subtraction = " + str(magnitude - img_back))
    
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

def notch_reject(magnitude_log, fft, notch_points):
    # Convert log-magnitude back to linear scale if needed
    magnitude = np.exp(magnitude_log / 20) - 1  # Only use if fft_image is in log scale

    # Get phase from the original FFT data
    phase = np.angle(fft)

    # Construct the notch filter
    rows, cols = magnitude_log.shape
    center_row, center_col = rows // 2, cols // 2

    # Mask initialization for visualization
    mask_total = np.ones((rows, cols), dtype=np.float32)

    for point in notch_points:
        mask = np.zeros((rows, cols), dtype=np.float32)
        cv2.circle(mask, (point[0], point[1]), int(point[2]), 1, -1)
        cv2.circle(mask, (-point[0], -point[1]), int(point[2]), 1, -1)
        mask_total *= (1 - mask)  # Update total mask for visualization
        magnitude *= (1 - mask)  # Apply notch filter by multiplying with the mask

    # Optionally visualize the mask to confirm the notches
    cv2.imshow('Mask', mask_total)
    cv2.waitKey(0)

    # Recompose the complex image from magnitude and phase
    complex_image = magnitude * (np.cos(phase) + 1j * np.sin(phase))

    # Inverse FFT to convert back to spatial domain
    img_back = np.fft.ifft2(np.fft.ifftshift(complex_image))
    img_back = np.real(img_back)

    # Normalize the image to 8-bit scale (0-255)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    return img_back

def gaussian_noise(image=None, mean=0, std=25, fixed_size=(256, 256)):
    """
    Applies Gaussian noise to a provided image or to a fixed-size empty image if no image is provided,
    and returns the noisy image along with the histogram of just the noise.

    Parameters:
        image (numpy.ndarray, optional): The image to which the noise will be applied. Can be grayscale or color.
        mean (float): The mean of the Gaussian noise.
        std (float): The standard deviation of the Gaussian noise.
        fixed_size (tuple): The size of the image if no image is provided, in the format (height, width).

    Returns:
        tuple: A tuple containing:
               - numpy.ndarray: The noisy image.
               - numpy.ndarray: The histogram of the noise.
    """
    # Create an empty image if none is provided
    if image is None:
        image = np.zeros((fixed_size[0], fixed_size[1], 3), dtype=np.uint8)  # Assuming a color image

    # Check if the image is grayscale
    is_gray = len(image.shape) == 2 or image.shape[2] == 1

    # Generate Gaussian noise
    if is_gray:
        gaussian_noise = np.random.normal(mean, std, image.shape)
    else:
        gaussian_noise = np.random.normal(mean, std, image.shape)

    # Create noise image for histogram calculation
    noise_image = gaussian_noise - gaussian_noise.min()
    noise_image = (noise_image / noise_image.max()) * 255
    noise_image = noise_image.astype(np.uint8)

    # Add noise to the image
    noisy_image = cv2.add(image.astype(np.float32), gaussian_noise.astype(np.float32))
    noisy_image = noisy_image.clip(0, 255).astype(np.uint8)

    # Calculate the histogram of the noise layer
    hist = cv2.calcHist([noise_image], [0], None, [256], [0, 256])

    return noisy_image, hist

def salt_and_pepper_noise(image=None, salt_prob=0.05, pepper_prob=0.05, fixed_size=(256, 256)):
    """
    Applies salt and pepper noise to a provided image or to a fixed-size empty image if no image is provided,
    and returns the noisy image along with the histogram of the noise.

    Parameters:
        image (numpy.ndarray, optional): The image to which the noise will be applied. Can be grayscale or color.
        salt_prob (float): The probability of adding salt noise.
        pepper_prob (float): The probability of adding pepper noise.
        fixed_size (tuple): The size of the image if no image is provided, in the format (height, width).

    Returns:
        tuple: A tuple containing:
               - numpy.ndarray: The noisy image.
               - numpy.ndarray: The histogram of the noise.
    """
    # Create an empty image if none is provided
    if image is None:
        image = np.zeros((fixed_size[0], fixed_size[1], 3), dtype=np.uint8)  # Assuming a color image

    # Check if the image is grayscale
    is_gray = len(image.shape) == 2 or image.shape[2] == 1

    # Initialize the noise layer
    if is_gray:
        noise_layer = np.zeros_like(image)
    else:
        noise_layer = np.zeros_like(image)

    # Generate salt noise
    num_salt = np.ceil(salt_prob * image.size).astype(int)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noise_layer[tuple(salt_coords)] = 255

    # Generate pepper noise
    num_pepper = np.ceil(pepper_prob * image.size).astype(int)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noise_layer[tuple(pepper_coords)] = 0

    # Add noise to the original image
    noisy_image = cv2.bitwise_or(image, noise_layer)

    # Calculate the histogram of the noise layer
    hist = cv2.calcHist([noise_layer], [0], None, [256], [0, 256])

    return noisy_image, hist

def geometric_mean_filter(image, kernel_size):
    """
    Applies a geometric mean filter to an image.

    Parameters:
    - input_image: numpy array, the image to be filtered.
    - kernel_size: int, the size of the kernel (must be odd).

    Returns:
    - filtered_image: numpy array, the geometric mean filtered image.
    """

    def geometric_mean(data):
        # Avoid logarithm of zero by replacing zero with a very small number
        data = np.where(data == 0, 1e-10, data)
        # Calculate the logarithm of the pixels
        log_data = np.log(data)
        # Compute the mean of the logarithm values
        mean_log = np.mean(log_data)
        # Return the exponent of the mean log, which is the geometric mean
        return np.exp(mean_log)
    
    kernel_shape = (kernel_size, kernel_size) if image.ndim == 2 else (kernel_size, kernel_size, 1)

    # Apply the geometric mean filter
    filtered_image = generic_filter(image, geometric_mean, size=kernel_shape)

    return filtered_image

def alpha_trimmed_mean_filter(image, kernel_size, d):
    """
    Apply alpha-trimmed mean filter to an image.
    
    :param image: Input image
    :param kernel_size: Size of the kernel (must be an odd number)
    :param d: Number of pixels to trim from each end of the pixel list (must be less than half of kernel_size**2)
    :return: Alpha-trimmed mean filtered image
    """
    # Check if the kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    
    # Check if d is appropriate
    if d * 2 >= kernel_size**2:
        raise ValueError("d must be less than half of kernel_size**2.")
    
    # Define the depth of the filter
    depth = -1  # Use the same depth as the source image
    
    # Define a custom filter function
    def filter_function(window):
        window = window.flatten()
        trimmed = np.sort(window)[d:-d]  # Trim d pixels from both ends and sort
        return np.mean(trimmed)
    
    # Apply the custom filter
    filtered_image = cv2.filter2D(image, depth, np.ones((kernel_size, kernel_size)), borderType=cv2.BORDER_REFLECT)
    filtered_image = cv2.copyMakeBorder(filtered_image, d, d, d, d, cv2.BORDER_REFLECT)
    filtered_image = np.array([[filter_function(filtered_image[i:i+kernel_size, j:j+kernel_size])
                                for j in range(filtered_image.shape[1] - kernel_size + 1)]
                               for i in range(filtered_image.shape[0] - kernel_size + 1)])
    
    return filtered_image.astype(image.dtype)

def get_structuring_element(kernel_size, element_type):
    """
    Generates a structuring element (kernel) of the specified type and size.

    :param kernel_size: Size of the kernel.
    :param element_type: Type of the structuring element ('rect', 'ellipse', 'cross').
    :return: Structuring element (kernel).
    """
    if element_type == 'rect':
        return np.ones((kernel_size, kernel_size), np.uint8)
    elif element_type == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif element_type == 'cross':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    else:
        raise ValueError("Unsupported element type. Use 'rect', 'ellipse', or 'cross'.")

def erosion(image, kernel_size=3, iterations=1, element_type='rect'):
    """
    Applies erosion to the input image using a custom implementation.

    :param image: Input image (numpy array).
    :param kernel_size: Size of the kernel (default is 3).
    :param iterations: Number of times erosion is applied (default is 1).
    :param element_type: Type of the structuring element ('rect', 'ellipse', 'cross').
    :return: Eroded image (numpy array).
    """
    # Ensure the input image is binary
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Get the image dimensions
    rows, cols = binary_image.shape
    
    # Create the structuring element
    kernel = get_structuring_element(kernel_size, element_type)
    k_center = kernel_size // 2
    
    # Pad the image with zeros on all sides
    padded_image = np.pad(binary_image, pad_width=k_center, mode='constant', constant_values=0)
    
    # Perform erosion
    eroded_image = binary_image.copy()
    for _ in range(iterations):
        for i in range(rows):
            for j in range(cols):
                # Extract the region of interest
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                # Apply the erosion operation
                if np.array_equal(region & kernel, kernel):
                    eroded_image[i, j] = 255
                else:
                    eroded_image[i, j] = 0
        # Update the padded image for the next iteration
        padded_image = np.pad(eroded_image, pad_width=k_center, mode='constant', constant_values=0)
    
    return eroded_image

def dilation(image, kernel_size=3, iterations=1, element_type='rect'):
    """
    Applies dilation to the input image using a custom implementation.

    :param image: Input image (numpy array).
    :param kernel_size: Size of the kernel (default is 3).
    :param iterations: Number of times dilation is applied (default is 1).
    :param element_type: Type of the structuring element ('rect', 'ellipse', 'cross').
    :return: Dilated image (numpy array).
    """
    # Ensure the input image is binary
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Get the image dimensions
    rows, cols = binary_image.shape
    
    # Create the structuring element
    kernel = get_structuring_element(kernel_size, element_type)
    k_center = kernel_size // 2
    
    # Pad the image with zeros on all sides
    padded_image = np.pad(binary_image, pad_width=k_center, mode='constant', constant_values=0)
    
    # Perform dilation
    dilated_image = binary_image.copy()
    for _ in range(iterations):
        for i in range(rows):
            for j in range(cols):
                # Extract the region of interest
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                # Apply the dilation operation
                if np.any(region & kernel):
                    dilated_image[i, j] = 255
                else:
                    dilated_image[i, j] = 0
        # Update the padded image for the next iteration
        padded_image = np.pad(dilated_image, pad_width=k_center, mode='constant', constant_values=0)
    
    return dilated_image

def opening(image, kernel_size=3, iterations=1, element_type='rect'):
    """
    Applies opening to the input image using erosion followed by dilation.

    :param image: Input image (numpy array).
    :param kernel_size: Size of the kernel (default is 3).
    :param iterations: Number of times each operation is applied (default is 1).
    :param element_type: Type of the structuring element ('rect', 'ellipse', 'cross').
    :return: Opened image (numpy array).
    """
    eroded_image = erosion(image, kernel_size, iterations, element_type)
    opened_image = dilation(eroded_image, kernel_size, iterations, element_type)
    return opened_image

def closing(image, kernel_size=3, iterations=1, element_type='rect'):
    """
    Applies closing to the input image using dilation followed by erosion.

    :param image: Input image (numpy array).
    :param kernel_size: Size of the kernel (default is 3).
    :param iterations: Number of times each operation is applied (default is 1).
    :param element_type: Type of the structuring element ('rect', 'ellipse', 'cross').
    :return: Closed image (numpy array).
    """
    dilated_image = dilation(image, kernel_size, iterations, element_type)
    closed_image = erosion(dilated_image, kernel_size, iterations, element_type)
    return closed_image

def calculate_histogram(image):
    """Calculate the histogram of image intensities."""
    histogram = defaultdict(int)
    width, height = image.size
    pixels = list(image.getdata())
    for pixel in pixels:
        histogram[pixel] += 1
    return histogram, width * height

def build_huffman_tree(histogram):
    """Build a Huffman tree given a histogram."""
    heap = [[weight, [symbol, ""]] for symbol, weight in histogram.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def calculate_entropy(histogram, total_pixels):
    """Calculate the entropy of the image."""
    probabilities = [count / total_pixels for count in histogram.values()]
    return scipy_entropy(probabilities, base=2)

def calculate_compression_ratios(histogram, huffman_codes, total_pixels):
    """Calculate the compression ratios and relative redundancy."""
    original_size = total_pixels * 8  # 8 bits per pixel
    compressed_size = sum(len(code) * histogram[symbol] for symbol, code in huffman_codes)
    compression_ratio = original_size / compressed_size
    relative_redundancy = 1 - (1 / compression_ratio)
    return compression_ratio, relative_redundancy

def huffman_coding(image):
    """Main function to process the image and return the required metrics."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image).convert('L')
    else:
        image = Image.fromarray(image)
    histogram, total_pixels = calculate_histogram(image)
    huffman_codes = build_huffman_tree(histogram)
    entropy = calculate_entropy(histogram, total_pixels)
    compression_ratio, relative_redundancy = calculate_compression_ratios(histogram, huffman_codes, total_pixels)

    result = {
        'Huffman Codes': [(symbol, histogram[symbol] / total_pixels, code) for symbol, code in huffman_codes],
        'Entropy': entropy,
        'Compression Ratio': compression_ratio,
        'Relative Redundancy': relative_redundancy
    }
    
    return result


def canny_edge_detection(image, low_threshold, high_threshold):
    """
    Apply Canny edge detection to an image and return the resultant image, 
    the non-maxima suppressed image, the high threshold image, and the low threshold image.

    Parameters:
    image (numpy.ndarray): The input image.
    low_threshold (int): The low threshold value for the hysteresis procedure.
    high_threshold (int): The high threshold value for the hysteresis procedure.

    Returns:
    tuple: The resultant edge-detected image, the non-maxima suppressed image, 
           the high threshold image, and the low threshold image.
    """
    # Convert image to grayscale if it is not
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Apply GaussianBlur to smooth the image and reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    # Create the high threshold image by applying a binary threshold
    _, high_thresh_image = cv2.threshold(blurred_image, high_threshold, 255, cv2.THRESH_BINARY)

    # Create the low threshold image by applying a binary threshold
    _, low_thresh_image = cv2.threshold(blurred_image, low_threshold, 255, cv2.THRESH_BINARY)

    # Extract the non-maxima suppressed image (edges before hysteresis thresholding)
    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx) * (180 / np.pi) % 180

    # Non-maximum suppression
    nms_image = np.zeros_like(magnitude, dtype=np.uint8)
    angle = direction // 45 * 45

    for i in range(1, nms_image.shape[0] - 1):
        for j in range(1, nms_image.shape[1] - 1):
            try:
                q = 255
                r = 255

                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # Angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # Angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # Angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    nms_image[i, j] = magnitude[i, j]
                else:
                    nms_image[i, j] = 0

            except IndexError as e:
                pass

    return edges, nms_image, high_thresh_image, low_thresh_image
