from imageio import imread, imsave
from skimage.measure import moments
from skimage.filters import threshold_otsu, gaussian
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects
import numpy as np
import matplotlib.pyplot as plt
import argparse


def _parseargs():
    parser = argparse.ArgumentParser(
        description="Autocrop image with obvious frontground and background\n\n",
        epilog="Developed by Huy Nguyen, Gruebele-Lyding Groups\n"
        "University of Illinois at Urbana-Champaign\n",
    )
    parser.add_argument(
        "input", nargs="*", type=str, help="Put in your images"
    )
    parser.add_argument(
        "--convert-gray",
        "-g",
        action="store_true",
        help="Whether to convert to gray and save",
    )
    parser.add_argument(
        "--minsize",
        "-m",
        type=int,
        help="minsize in pixels of objects",
        default=30,
    )
    parser.add_argument(
        "--extra-space", "-x", type=float, help="extra rim space", default=0.2
    )
    args = parser.parse_args()
    return args


def autocrop(img, min_size=30, convert_gray=False, extra_space=0.2):
    """Separate front ground and background. Choose only largest objects as front ground.

    min_size: minimum size in pixels of objects to include.
    extra_space: fraction of the front ground size
    """
    img = np.array(img).copy()
    # Convert to grayscale if RGB
    if len(img.shape) > 2:
        img_gr = rgb2gray(img)
    else:
        img_gr = img.copy()

    # Filter image
    img_gr = gaussian(img_gr)

    # Convert to binary
    thr = threshold_otsu(img_gr)
    binary = img_gr > thr

    # Remove small spots
    binary = remove_small_objects(binary, min_size=min_size)

    # Find centroid
    M = moments(binary)
    centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])

    # Find edges
    fg_indices = np.argwhere(binary > 0)
    mincol = np.min(fg_indices[:, 1])
    maxcol = np.max(fg_indices[:, 1])
    minrow = np.min(fg_indices[:, 0])
    maxrow = np.max(fg_indices[:, 0])
    extension = extra_space * np.array([maxrow - minrow, maxcol - mincol])

    # Crop to 100% + extra_space length each dimension
    framerow = np.clip(
        [int(minrow - extension[0]), int(maxrow + extension[0])],
        a_min=0,
        a_max=img.shape[0],
    )
    framecol = np.clip(
        [int(mincol - extension[1]), int(maxcol + extension[1])],
        a_min=0,
        a_max=img.shape[1],
    )

    if convert_gray:
        return rgb2gray(img)[
            framerow[0] : framerow[1], framecol[0] : framecol[1]
        ]

    else:
        return img[framerow[0] : framerow[1], framecol[0] : framecol[1]]


def main():
    args = _parseargs()
    for f in args.input:
        img = imread(f)
        cropped_img = autocrop(
            img,
            min_size=args.minsize,
            convert_gray=args.convert_gray,
            extra_space=args.extra_space,
        )
        fn = f.split(".")
        imsave(
            "".join(fn[:-1]) + f"_cropped.{fn[-1]}",
            cropped_img,
            format="TIFF-PIL",
            # compression="tiff_deflate",
        )


if __name__ == "__main__":
    main()
