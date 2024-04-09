import numpy as np
import pytest
from src.image_processor import apply_gamma_transformation


def create_uniform_image(shape, intensity):
    """
    Creates a uniform image of a given shape and pixel intensity.

    Parameters:
    - shape (tuple): The shape of the image (height, width, channels).
    - intensity (int): The intensity value for all pixels.

    Returns:
    - numpy.ndarray: The uniform image.
    """
    return np.full(shape, intensity, dtype=np.uint8)


@pytest.mark.parametrize("gamma,expected_intensity", [
    (0.5, 188),  # Lighten the image
    (1.0, 128),  # No change
    (2.0, 64)  # Darken the image
])
def test_apply_gamma_transformation(gamma, expected_intensity):
    # Create a uniform gray image with intensity 128
    test_image = create_uniform_image((10, 10, 3), 128)

    # Apply the gamma transformation
    transformed_image = apply_gamma_transformation(test_image, gamma)

    # Calculate the average intensity of the transformed image
    avg_intensity = np.mean(transformed_image)

    # Assert that the average intensity is close to the expected value, with a larger tolerance
    assert np.isclose(avg_intensity, expected_intensity, atol=10), f"Failed for gamma={gamma}"

