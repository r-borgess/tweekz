import image_processor

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