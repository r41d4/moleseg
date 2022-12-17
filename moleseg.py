import click

from pathlib import Path
from skimage.io import imread, imsave

from moleseg.image import preprocess_image
from moleseg.segmentation import quantize_image
from moleseg.segmentation import extract_mole_polygon
from moleseg.image import create_mask
from moleseg.overlay import overlay_object
from moleseg.enums import LabelColors


def run_pipeline(
    image,
    overlay,
    overlay_color,
    increase_contrast=True,
    gaussian_sigma=10,
    cluster_count=2,
    round_count=1,
    fill_border=True,
    convex_hull=False,
    border_width=5,
    fill=True,
):
    preprocessed_image = preprocess_image(
        image,
        increase_contrast=increase_contrast,
        gaussian_sigma=gaussian_sigma,
    )
    quantized_image = quantize_image(
        preprocessed_image, cluster_count, round_count, fill_border
    )
    mole_polygon = extract_mole_polygon(quantized_image, convex_hull)
    segmentation_mask = create_mask(image, mole_polygon)

    image_overlayed = (
        overlay_object(
            image,
            mole_polygon,
            border_color=overlay_color,
            border_width=border_width,
            fill=fill,
        )
        if overlay
        else None
    )
    return segmentation_mask, image_overlayed


@click.command()
@click.option(
    "-i",
    "--input-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(exists=True, dir_okay=True),
)
@click.option("--overlay", is_flag=True, default=False)
@click.option("--overlay-color", required=False, type=click.STRING)
def run_segmentation(input_path, output_path, overlay, overlay_color):
    input_path, output_path = Path(input_path), Path(output_path)
    image = imread(input_path)
    print(f":: Loaded image: {input_path.name}")
    if overlay_color in LabelColors.names():
        overlay_color = LabelColors[overlay_color].value
    else:
        overlay_color = LabelColors.red.value

    print(":: Running segmentation... ", end="", flush=True)
    segmentation_mask, overlayed_image = run_pipeline(
        image, overlay, overlay_color
    )
    print("Done!")

    segmentation_mask_path = output_path / f"{input_path.stem}_segmentation_mask.png"
    imsave(segmentation_mask_path, segmentation_mask)
    print(f":: Segmentation mask: {segmentation_mask_path.resolve()}")
    if overlay:
        overlayed_image_path = output_path / f"{input_path.stem}_overlayed.png"
        imsave(overlayed_image_path, overlayed_image)
        print(f":: Overlayed image: {overlayed_image_path.resolve()}")


def main():
    run_segmentation()


if __name__ == "__main__":
    main()
