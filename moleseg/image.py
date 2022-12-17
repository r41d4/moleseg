import numpy as np
import cv2

from skimage.filters import gaussian


def lut_enhance(image):
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype("uint8")
    image = cv2.LUT(image, table)
    return image


def preprocess_image(
    image,
    increase_contrast=True,
    gaussian_kernel_size=5,
    gaussian_sigma=1.0,
):
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


def create_mask(image, object_polygon):
    mask = np.zeros(image.shape[:2])
    if object_polygon is not None:
        cv2.fillPoly(mask, [object_polygon], color=255)
    mask = mask.astype(np.uint8)
    return mask
