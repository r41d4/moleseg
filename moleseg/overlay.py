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
