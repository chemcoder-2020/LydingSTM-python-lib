import argparse
import matplotlib.pyplot as plt

from numpy import percentile
from skimage.exposure import rescale_intensity
from pathlib import Path
from imageio import imread


def annotate_fig(ax, s, coords, size=20):
    ax.annotate(s, coords, size=size, bbox=dict(boxstyle="round4,pad=.5", fc="0.9"))


def create_fig(path, s, coords, size=20, cmap="gray", stretch_contrast=(1, 99)):
    im = imread(path)
    im = rescale_intensity(
        im,
        in_range=(
            percentile(im, stretch_contrast[0]),
            percentile(im, stretch_contrast[1]),
        ),
    )
    fig, ax = plt.subplots()
    ax.imshow(im, cmap=cmap)
    annotate_fig(ax, s, coords, size=size)
    ax.axis("off")
    return fig


def _parseargs():
    parser = argparse.ArgumentParser(
        description="annotate_images.py (version 0.1.0)\n\n",
        prog="annotate_images",
        epilog="Created by Huy Nguyen, Gruebele Group\n"
        "University of Illinois at Urbana-Champaign",
    )
    parser.add_argument(
        "--input", "-i", nargs="*", help="Input image(s)", required=True, type=str
    )
    parser.add_argument(
        "--strings",
        "-s",
        nargs="*",
        help="List of string(s) for annotation(s)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--coordinates",
        "-c",
        nargs="*",
        help="List of coordinate pairs. If one pair for multiple images, will use it for all images.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--textSize", "-z", help="Font size of the annotation text", type=int
    )
    parser.add_argument(
        "--colormap",
        "-m",
        help="Colormap to display the image(s) in. See matplotlib documentation for available colormaps.",
        choices=["gray", "gray_r", "viridis", "jet", "inferno"],
        default="gray",
    )
    parser.add_argument(
        "--contrast",
        "-e",
        help="Low and high percentiles of image values to stretch contrast.",
        nargs=2,
        type=float,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parseargs()
    input_files = [Path(inp) for inp in args.input]
    annotations = args.strings
    coordinates = [
        (args.coordinates[i], args.coordinates[i + 1])
        for i in range(len(args.coordinates))
        if i % 2 == 0
    ]
    textSize = args.textSize
    colormap = args.colormap
    contrast = args.contrast
    for i, fn in enumerate(input_files):
        if len(coordinates) == 1:
            fig = create_fig(
                fn,
                annotations[i],
                coordinates[0],
                size=textSize,
                cmap=colormap,
                stretch_contrast=contrast,
            )
        else:
            fig = create_fig(
                fn,
                annotations[i],
                coordinates[i],
                size=textSize,
                cmap=colormap,
                stretch_contrast=contrast,
            )
        fig.savefig(
            "{fn}_a.tiff".format(fn=fn),
            dpi=300,
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)
