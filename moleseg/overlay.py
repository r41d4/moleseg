import numpy as np
import cv2

from typing import Union, Sequence
from numpy.typing import NDArray


def overlay_object(
    image: NDArray,
    object_polygon: NDArray,
    border_color: Union[Sequence[int], NDArray],
    border_width: int,
    fill: bool,
    fill_alpha: float = 0.5,
) -> NDArray:
    """
    Draw a polygon over an input image.

    Returns an overlayed images.

    Parameters
    ----------
    image : ndarray
        Two-dimensional array with image data.

    object_polygon: ndarray
        Array containing x, y coordinates of the vertices of an object polygon.

    border_color: array-like of int or ndarray
        Array of RGB color values for the border of an overlayed polygon.
        Shifted values of this vector are used for the polygon fill.

    border_width : int
        Width of the border of an overlayed polygon.

    fill : bool
        Flag for filling the polygon within its borders.

    fill_alpha : float, optional, default 0.5
        Alpha value for the fill color of the overlayed polygon.
    """
    image_overlayed = image.copy()
    if object_polygon is not None:
        if fill:
            fill_color = [color + 50 for color in border_color]
            image_filled_object = np.zeros_like(image, np.uint8)
            cv2.fillPoly(image_filled_object, [object_polygon], color=fill_color)
            fill_mask = image_filled_object.astype(bool)
            image_overlayed[fill_mask] = cv2.addWeighted(
                image_overlayed, fill_alpha, image_filled_object, 1 - fill_alpha, 0
            )[fill_mask]
        cv2.drawContours(
            image_overlayed, [object_polygon], 0, border_color, border_width
        )
    return image_overlayed
