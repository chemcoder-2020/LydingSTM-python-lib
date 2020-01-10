import numpy as np
import argparse
import imageio
from skimage.color import rgb2gray
from myImageToolset import normalize_img, imreg

# from os import getcwd
import pystripe as stripe
from detrend2d import subtract_plane
import parseSTM as ps
from skimage.exposure import rescale_intensity


def _parseargs():
    parser = argparse.ArgumentParser(
        description="Multiply STM buffers\n\n",
        epilog="Developed by Huy Nguyen, Gruebele-Lyding Groups\n"
        "University of Illinois at Urbana-Champaign\n",
    )
    parser.add_argument(
        "input", nargs="*", type=str, help="[STM File 1 STM File 2 ...]."
    )
    parser.add_argument(
        "--buffers",
        "-b",
        help="Buffers to multiply",
        nargs="*",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--secondarySet",
        "-a",
        action="store_true",
        help="Use topo set to register images",
    )
    parser.add_argument(
        "--textout",
        "-t",
        action="store_true",
        help="Output a text file for the mult data (helpful for reusing it in other script)",
    )
    args = parser.parse_args()
    return args


def read_img(path, buf):
    stmfile = ps.STMfile(path)
    return stmfile.get_height_buffers([buf])[buf]


def get_bufset(paths, bufs):
    out = {}
    for buf in bufs:
        out[buf] = [read_img(path, buf) for path in paths]
    return out


def img_process(img):
    # Will return a 16-bit image
    img = stripe.filter_streaks(img, sigma=[10, 20], level=2, wavelet="db3")
    return img


def img_post_process(img):
    img = rescale_intensity(
        img, in_range=(np.percentile(img, 0.3), np.percentile(img, 99.7))
    )
    return img


def mult_img(imset, secondarySet=None, returnText=False):
    if secondarySet is not None:
        imset = imreg(imset, secondarySet=secondarySet, upsample_factor=100)
    else:
        imset = imreg(imset, upsample_factor=100)
    mult = np.ones_like(imset[0], dtype="float")
    for im in imset:
        mult *= im
    if not returnText:
        mult = normalize_img(mult, 0, 2 ** 16 - 1).astype("uint16")
    return mult


def main():
    # cwd = getcwd()
    args = _parseargs()
    bufset = get_bufset(args.input, args.buffers)
    topos = get_bufset(args.input, [1])[1]
    topos = [img_process(img) for img in topos]
    for buf in args.buffers:
        imset = bufset[buf]
        if args.secondarySet:
            mult = mult_img(imset, secondarySet=topos, returnText=args.textout)
        else:
            mult = mult_img(imset, returnText=args.textout)
        if not args.textout:
            mult = img_post_process(mult)
            imageio.imwrite(f"./mult_buf{buf}.tiff", mult)
        else:
            np.savetxt(f"./mult_buf{buf}.txt", mult, delimiter=",")


if __name__ == "__main__":
    main()

