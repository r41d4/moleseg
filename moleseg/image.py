import numpy as np
import cv2

from numpy.typing import NDArray
from skimage.filters import gaussian


def lut_enhance(image: NDArray) -> NDArray:
    """
    Enhance the input image contrast using a LUT.

    Returns an image with enhanced contrast.

    Parameters
    ----------
    image : ndarray
        Two-dimensional array with image data.

    Returns
    -------
    ndarray
        Two-dimensional array with enhanced contrast image data.
    """
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype("uint8")
    image = cv2.LUT(image, table)
    return image


def preprocess_image(
    image: NDArray,
    increase_contrast: bool = True,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.0,
) -> NDArray:
    """
    Preprocess the input image.
    Enhance the contrast and apply the Gaussian filter to the input image.

    Returns a preprocessed image.

    Parameters
    ----------
    image : ndarray
        Two-dimensional array with image data.
    increase_contrast : bool, optional, default True
        Flag for increasing the contrast of the input image.
    gaussian_kernel_size : int, optional, default 5
        The size of the kernel to use for the Gaussian filter.
    gaussian_sigma : float, optional, default 1.0
        The standard deviation of the Gaussian filter.

    Returns
    -------
    ndarray
        Two-dimensional array with preprocessed image data.
    """
    preprocessed_image = image.copy()

    if increase_contrast:
        image = lut_enhance(image)

    preprocessed_image = gaussian(
        preprocessed_image,
        sigma=gaussian_sigma,
        truncate=1 / gaussian_kernel_size,
        channel_axis=2,
    )
    preprocessed_image = (preprocessed_image * 255).astype(np.uint8)
    return preprocessed_image


def create_mask(image: NDArray, object_polygon: NDArray) -> NDArray:
    """
    Create a mask.

    Returns a binary image representing a polygon mask.

    Parameters
    ----------
    image : ndarray
        Two-dimensional array with image data.
    object_polygon : ndarray
        Array containing x, y coordinates of the vertices of an object polygon.

    Returns
    -------
    ndarray
        Two-dimensional array with binary mask image data.
    """
    mask = np.zeros(image.shape[:2])
    if object_polygon is not None:
        cv2.fillPoly(mask, [object_polygon], color=255)
    mask = mask.astype(np.uint8)
    return mask
