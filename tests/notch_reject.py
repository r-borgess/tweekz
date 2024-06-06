import numpy as np
import cv2
import matplotlib.pyplot as plt

def notch_filter_fft(fft_image, notch_points):
    rows, cols = fft_image.shape
    crow, ccol = rows // 2 , cols // 2  # center

    # Create a mask with the same dimensions as the image
    mask = np.ones((rows, cols), np.uint8)

    for (x, y, radius) in notch_points:
        # Create a circular mask to filter out the specific frequencies
        cv2.circle(mask, (ccol + x, crow + y), radius, 0, -1)
        cv2.circle(mask, (ccol - x, crow - y), radius, 0, -1)

    # Apply the mask to the FFT image
    filtered_fft = fft_image * mask
    
    return filtered_fft

def ift_transform(filtered_fft_image):
    # Inverse FFT to get the filtered image back in spatial domain
    ifft_image = np.fft.ifftshift(filtered_fft_image)  # Shift back the zero frequency component to the original position
    img_back = np.fft.ifft2(ifft_image)
    img_back = np.abs(img_back)  # Take the magnitude to get the real part of the image

    return img_back

def apply_notch_filter(fft_image, notch_points):
    filtered_fft = notch_filter_fft(fft_image, notch_points)
    filtered_image = ift_transform(filtered_fft)
    return filtered_image

# Example usage
if __name__ == "__main__":
    # Load an image
    img = cv2.imread('C:/Users/rodri/tweekz/tests/test_images/load/ClownOrig.jpg', 0)  # Load in grayscale

    # Perform FFT
    fft_image = np.fft.fft2(img)
    fft_image = np.fft.fftshift(fft_image)  # Shift the zero frequency component to the center

    # Define notch points (x, y, radius)
    notch_points = [(30, 30, 100), (-30, -30, 10)]

    # Apply the notch filter
    filtered_image = apply_notch_filter(fft_image, notch_points)

    # Display the original and filtered images
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122),plt.imshow(filtered_image, cmap = 'gray')
    plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

    plt.show()
