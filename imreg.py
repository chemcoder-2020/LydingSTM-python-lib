from skimage.io import imread, imsave
import sys
import os
from skimage.feature import register_translation
from scipy.ndimage.interpolation import shift
import numpy as np
from skimage import img_as_float
import detrend2d as dt
from myImageToolset import imreg

# def imreg(refimg, txtdataset, full=False):
#     """perform subpixel registration translation on a data set"""

#     im0 = np.copy(refimg)
#     shifts = [
#         register_translation(im0, txtdataset[i], upsample_factor=100)[0]
#         for i in range(len(txtdataset))
#     ]
#     #  shifted images will still have the same z-value
#     shifted_images = [shift(txtdataset[i], shifts[i]) for i in range(len(txtdataset))]
#     # reconstructed_dataset = [im0] + shifted_images

#     #  cropping the zero pixels out of the translated images
#     true_points = [
#         np.argwhere(im) for im in shifted_images
#     ]  # find the non-zero points on image
#     topleft = [pts.min(axis=0) for pts in true_points]
#     bottomright = [pts.max(axis=0) for pts in true_points]
#     topleftx = [t[0] for t in topleft]
#     toplefty = [t[1] for t in topleft]
#     bottomrightx = [t[0] for t in bottomright]
#     bottomrighty = [t[1] for t in bottomright]
#     startx = max(topleftx)
#     starty = max(toplefty)
#     stopx = min(bottomrightx)
#     stopy = min(bottomrighty)
#     newset = [im[startx:stopx, starty:stopy] for im in shifted_images]
#     if full:
#         return newset, (startx, stopx, starty, stopy)
#     return newset


def natural_key(string_):
    """
    See http://www.codinghorror.com/blog/archives/001018.html
    This function is to define a key for sorting the filenames with numbers in it

    """
    import re

    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def normalize_img(img, vmin, vmax):
    img = img_as_float(img)
    scale = (img.max() - img.min()) / np.abs(vmax - vmin)
    img = (img - img.min()) / scale + vmin
    return img


def save_img(img, path="./"):
    imsave(path, normalize_img(img, 0, 2 ** 16 - 1).astype("uint16"))


def save_image_batch(image_batch, fns, path="./"):
    for e, fn in enumerate(fns):
        save_img(image_batch[e], os.path.join(path, fn))


if __name__ == "__main__":
    fpaths = []
    if len(sys.argv) < 3 or sys.argv[2] == "--help" or sys.argv[2] == "-h":
        sys.exit("\nUsage: imreg [refIMG] [im1 im2 ...]\n")
    else:
        # im0 = imread(sys.argv[2], True)
        for ind, arg in enumerate(sys.argv[1:]):
            if arg.endswith(".tif") or arg.endswith(".TIF") or arg.endswith(".tiff"):
                fpaths.append(arg)
        fpaths = sorted(fpaths, key=natural_key)
        imset = [
            imread(path, True)
            for path in fpaths
            if path.endswith(".tif") or path.endswith(".tiff") or path.endswith(".TIF")
        ]
        # registered_imset = imreg(im0, imset)
        registered_imset = imreg(imset)
        if 0 in registered_imset[0].shape:
            sys.exit("Cannot register image set.")
        # registered_imset = [dt.subtract_plane(im) for im in registered_imset]
        # Now save the images
        reg_dir = os.path.join(os.path.split(fpaths[0])[0], "registered")
        try:
            os.makedirs(reg_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
            user_input = input("%s exists. Do you want to overwrite it? (y/n) " % reg_dir)
            if user_input.lower() != "y":
                sys.exit("Registered folder was not overwritten.")
        save_image_batch(registered_imset, fpaths, path="./registered")
        sys.exit("Successfully registered and save image set")
