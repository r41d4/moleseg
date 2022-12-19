# MoleSeg

**MoleSeg** is a Python module providing tools for skin mole image segmentation.

The current method was validated on [ISIC 2016 challenge dataset](https://challenge.isic-archive.com/landing/2016/).

## Overview

### CLI interface

MoleSeg has a CLI interface accessed via `moleseg.py`. Given an image file `mole.png` and an output directory `out`, you can run the segmentation pipeline to produce a binary image containing a segmentation mask, `out/mole_segmentation_mask.png`:

```bash
python moleseg.py -i mole.png -o out
```

You can also specify to generate the image with the segmentation mask overlay on an original image in a given color (green, red, purple, cyan):

```bash
python moleseg.py -i mole.png -o out --overlay --overlay-color cyan 
```

Omitting the `--overlay-color` option will produce an overlay in red by default.


### Modules

- `moleseg.image` - image preprocessing and segmentation mask generation;
- `moleseg.segmentation` - mole segmentation;
- `moleseg.overlay` - overlaying the input image with the segmentation mask.
