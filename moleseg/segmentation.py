import numpy as np
import cv2

from numpy.typing import NDArray


def kmeans_color_quantization(
    image: NDArray, cluster_count: int, round_count: int
) -> NDArray:
    samples = image.reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(
        samples,
        cluster_count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
        round_count,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape(image.shape)


def quantize_image(
    image: NDArray, cluster_count: int, round_count: int, fill_border: bool
) -> NDArray:
    quantized_image = kmeans_color_quantization(image, cluster_count, round_count)
    if fill_border:
        original_quantized_image = quantized_image.copy()
        background_color = np.max(
            np.unique(
                original_quantized_image.reshape(-1, original_quantized_image.shape[2]),
                axis=0,
            ),
            axis=0,
        )
        background_border_size = 20
        height, width = original_quantized_image.shape[:2]
        quantized_image = np.full((height, width, 3), background_color, dtype=np.uint8)
        quantized_image[
            background_border_size : height - background_border_size,
            background_border_size : width - background_border_size,
        ] = original_quantized_image[
            background_border_size : height - background_border_size,
            background_border_size : width - background_border_size,
        ]
    return quantized_image


def extract_mole_polygon(image: NDArray, convex_hull: bool) -> NDArray:
    def choose_polygon(polygons):
        for polygon in polygons:
            if len(polygon) < 3 and not cv2.isContourConvex(polygon):
                continue

            x, y, width, height = cv2.boundingRect(polygon)
            image_height, image_width = image.shape[:2]

            if (
                x == 0
                or y == 0
                or x + width == image_width
                or y + height == image_height
            ):
                continue

            center_x = x + width / 2
            center_y = y + height / 2
            if not (
                center_x < 0.3 * image_width
                or center_x > 0.7 * image_width
                or center_y < 0.3 * image_height
                or center_y > 0.7 * image_height
            ):
                return polygon

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    thresholded = cv2.bitwise_not(thresholded)
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    polygon = choose_polygon(contours)
    if polygon is None:
        return None
    if convex_hull:
        polygon = cv2.convexHull(polygon)
    return polygon
