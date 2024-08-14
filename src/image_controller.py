import image_processor
import cv2

class ImageProcessor:
    def __init__(self):
        self.current_image = None
        self.original_image = None

    def load_image(self, file_path):
        self.current_image = image_processor.load_image(file_path)
        self.original_image = self.current_image.copy()
        return self.current_image

    def save_image(self, file_path):
        return image_processor.save_image(self.current_image, file_path)

    def blackout_image(self):
        self.current_image = image_processor.blackout_image(self.current_image)
        return self.current_image

    def restore_image(self):
        self.current_image = self.original_image.copy()
        return self.current_image
    
    def gamma_transform_image(self, gamma):
        self.current_image = image_processor.apply_gamma_transformation(self.current_image, gamma)
        return self.current_image

    def contrast_stretch_image(self, r1, s1, r2, s2):
        self.current_image = image_processor.apply_contrast_stretch(self.current_image, r1, s1, r2, s2)
        return self.current_image

    def extract_bit_plane(self, bit_plane):
        self.current_image = image_processor.bit_plane_slicer(self.current_image, bit_plane)
        return self.current_image
    
    def equalize_histogram(self):
        self.current_image, original_hist, equalized_hist = image_processor.histogram_equalization(self.current_image)
        return self.current_image, original_hist, equalized_hist
    
    def intensity_slicing_pseudocolor(self, num_ranges, cmap_name):
        self.current_image = image_processor.intensity_slicing_pseudocolor(self.current_image, num_ranges, cmap_name)
        return self.current_image
    
    def apply_average_filter(self, kernel_size):
        self.current_image = image_processor.average_filter(self.current_image, kernel_size)
        return self.current_image
    
    def apply_min_filter(self, kernel_size):
        self.current_image = image_processor.min_filter(self.current_image, kernel_size)
        return self.current_image
    
    def apply_max_filter(self, kernel_size):
        self.current_image = image_processor.max_filter(self.current_image, kernel_size)
        return self.current_image
    
    def apply_median_filter(self, kernel_size):
        self.current_image = image_processor.median_filter(self.current_image, kernel_size)
        return self.current_image
    
    def apply_laplacian_filter(self):
        self.current_image, laplacian, laplacian_adjusted = image_processor.laplacian_filter(self.current_image)
        return self.current_image, laplacian, laplacian_adjusted
    
    def compute_fft_spectrum_and_phase(self):
        self.current_image, phase_angle, f = image_processor.compute_fft_spectrum_and_phase(self.current_image)
        return self.current_image, phase_angle, f
    
    def compute_inverse_fft(self, magnitude_spectrum, phase_angle):
        self.current_image = image_processor.compute_inverse_fft(magnitude_spectrum, phase_angle)
        return self.current_image
    
    def apply_high_pass(self, radius):
        self.current_image = image_processor.high_pass(self.current_image, radius)
        return self.current_image
    
    def apply_low_pass(self, radius):
        self.current_image = image_processor.low_pass(self.current_image, radius)
        return self.current_image

    def apply_notch_reject(self, fft,notch_points):
        self.current_image = image_processor.notch_reject(self.current_image, fft, notch_points)
        return self.current_image
    
    def apply_gaussian_noise(self, mean, std, fixed_size):
        self.current_image, hist = image_processor.gaussian_noise(self.current_image, mean, std, fixed_size)
        return self.current_image, hist
    
    def apply_salt_and_pepper_noise(self, salt_prob=0.05, pepper_prob=0.05, fixed_size=(640, 640)):
        self.current_image, hist = image_processor.salt_and_pepper_noise(self.current_image, salt_prob, pepper_prob, fixed_size)
        return self.current_image, hist
    
    def apply_geometric_mean_filter(self, kernel_size):
        self.current_image = image_processor.geometric_mean_filter(self.current_image, kernel_size)
        return self.current_image
    
    def apply_alpha_trimmed_mean_filter(self, kernel_size, d):
        self.current_image = image_processor.alpha_trimmed_mean_filter(self.current_image, kernel_size, d)
        return self.current_image

    def apply_erosion(self, kernel_size, iterations, element_type):
        self.current_image = image_processor.erosion(self.current_image, kernel_size, iterations, element_type)
        return self.current_image
    
    def apply_dilation(self, kernel_size=3, iterations=1, element_type='rect'):
        self.current_image = image_processor.dilation(self.current_image, kernel_size, iterations, element_type)
        return self.current_image

    def apply_opening(self, kernel_size=3, iterations=1, element_type='rect'):
        self.current_image = image_processor.opening(self.current_image, kernel_size, iterations, element_type)
        return self.current_image
    
    def apply_closinig(self, kernel_size=3, iterations=1, element_type='rect'):
        self.current_image = image_processor.closing(self.current_image, kernel_size, iterations, element_type)
        return self.current_image

    def apply_huffman_coding(self):
        self.current_image = image_processor.huffman_coding(self.current_image)
        return self.current_image
    
    def apply_canny_edge_detection(self, low_threshold, high_threshold):
        self.current_image, non_maxima_suppressed, weak_edges, strong_edges = image_processor.canny_edge_detection(self.current_image, low_threshold, high_threshold)
        return self.current_image, non_maxima_suppressed, weak_edges, strong_edges
    
    def apply_region_growing(self, seeds, threshold=10):
        self.current_image = image_processor.region_growing_segmentation(self.current_image, seeds, threshold)
        return self.current_image
    
    def apply_chain_code(self):
        if self.current_image is not None:
            gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            #_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            chain_code, min_magnitude_code, first_diff_code = image_processor.chain_code(gray_image)
            return chain_code, min_magnitude_code, first_diff_code
        else:
            raise ValueError("No image loaded.")
        
    def apply_skeletonization(self):
        if self.current_image is not None:
            gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            self.current_image = image_processor.skeletize(binary_image)
            return self.current_image
        else:
            raise ValueError("No image loaded.")
        
    def apply_harris_corner_detector(self, k, T):
        if self.current_image is not None:
            self.current_image = image_processor.harris_corner_detector(self.current_image, k, T)
            return self.current_image
        else:
            raise ValueError("No image loaded.")