import parseSTM as ps
import imageio
from skimage.filters import gaussian, threshold_otsu
from skimage.exposure import rescale_intensity
import numpy as np
from myImageToolset import (
    normalize_img,
    imreg,
    crop_image_set,
    destripe_by_wavelet_svd,
)
from os.path import split, join
import pystripe as stripe
from detrend2d import subtract_plane
import argparse

from skimage.feature import register_translation, match_template
from scipy.ndimage import shift
import matplotlib as mpl
import cv2
from scipy.ndimage import zoom
from skimage.color import rgb2gray
import copy
import matplotlib
import pywt
from scipy import fftpack
from os import remove
from skimage.measure import compare_ssim
import matplotlib.ticker as mtick
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects


matplotlib.use("MacOSX")
import matplotlib.pyplot as plt


def _parseargs():
    parser = argparse.ArgumentParser(
        description="Extract image buffers from Lyding STM file\n\n",
        epilog="Developed by Huy Nguyen, Gruebele-Lyding Groups\n"
        "University of Illinois at Urbana-Champaign\n",
    )
    parser.add_argument(
        "input",
        nargs="*",
        type=str,
        help="Contact lyding@illinois.edu for information about Lyding STM file format",
    )
    parser.add_argument(
        "--buffers",
        "-b",
        help="Specific buffers to extract",
        type=int,
        required=True,
        nargs="*",
    )
    parser.add_argument("--logfile", help="Movie log file", type=str)
    parser.add_argument("--frames", help="Number of movie frames", type=int)
    parser.add_argument(
        "--register",
        "-r",
        help="Translationally register the image set to the first input file.",
        action="store_true",
    )
    parser.add_argument(
        "--colormap",
        "-c",
        help="Colormap to use for the lock-in buffers",
        type=str,
        default="gray_r",
    )
    parser.add_argument(
        "--destripe",
        "-s",
        help="Attempt to remove stripes from the buffers (helpful for topography)",
        action="store_true",
    )
    parser.add_argument(
        "--gauss",
        "-g",
        help="Specify a sigma if a Gaussian filtering operation on the images are desired.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--lockin",
        "-l",
        help="Lock-in values (in order of sensitivity, expand, offset) for the X and Y buffers. For now, this option is ignored",
        nargs=3,
        type=float,
        default=None,
    )
    parser.add_argument(
        "--movie",
        "-m",
        help="Create a movie out of the STM files in the order of inputs (will register images by default, but remember to omit register flag when having movie flag on)",
        action="store_true",
    )
    parser.add_argument(
        "--stretch-contrast",
        "-e",
        help="Specify the percentile to stretch image contrast",
        required=False,
        nargs=2,
        type=float,
        default=None,
    )
    parser.add_argument(
        "--filenames",
        "-f",
        help="Replace original filenames with a set of specified names",
        nargs="*",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--textcolor",
        "-t",
        help="Textcolor for annotation of movies. Support black and white",
        default="white",
        type=str,
    )
    parser.add_argument(
        "--normalize",
        "-n",
        help="Normalize the series of images by the first image.",
        action="store_true",
    )
    parser.add_argument(
        "--secondarySet",
        "-a",
        help="Whether to use topographic set as to align when registering images (movies and regular)",
        action="store_true",
    )
    parser.add_argument(
        "--svd",
        "-q",
        help="Perform SVD on the movie series and plot the weight of U matrix against time. Please specify an integer smaller than the minimum dimension of the frame.",
        type=int,
    )
    parser.add_argument(
        "--level",
        "-z",
        help="Maximum level up to which wavelet decomposition (image destriping) works.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--wavelet",
        "-w",
        help="Mother wavelet to use for wavelet decomposition (image destriping).",
        type=str,
        default="sym7",
    )
    parser.add_argument(
        "--level-svd",
        "-x",
        help="Maximum level up to which wavelet decomposition (wavelet svd) works.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--wavelet-svd",
        "-y",
        help="Mother wavelet to use for wavelet decomposition (wavelet svd).",
        type=str,
        default="sym7",
    )
    parser.add_argument(
        "--route2", help="Use the wavelet svd method", action="store_true"
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="If on, extract svd from foreground (using Otsu thresholding) only",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="If on, extract svd from background (using Otsu thresholding) only",
    )
    args = parser.parse_args()
    return args


def lockin_value(x, sensitivity, expand, offset):
    return (x / expand / 10000 + offset) * sensitivity


def wavelet_filter_1d(arr, wavelet_type="db3", level=3, threshold_mode="soft"):
    arr = np.array(arr).flatten()
    coeffs = pywt.wavedec(arr, wavelet_type, level=level)
    approx = coeffs[0]
    detail = coeffs[1:]
    # coeffs[1:] = [
    #     pywt.threshold(coeff, np.std(coeff) * 3, mode=threshold_mode)
    #     for coeff in coeffs[1:]
    # ]
    coeffs_filt = [approx]
    for nlevel in detail:
        fdetail = fftpack.rfft(nlevel)
        # b, a = signal.butter(6, 0.1, btype="low")
        # fdetail = signal.filtfilt(b, a, fdetail, padlen=5)
        fdetail = pywt.threshold(
            fdetail, np.std(fdetail) * 3, mode=threshold_mode
        )
        coeffs_filt.append(fftpack.irfft(fdetail))
    return pywt.waverec(coeffs_filt, wavelet_type)


def wavelet_filter_3d(array3d, wavelet="db3", level=3):
    array3d = np.array(array3d)
    out = np.apply_along_axis(
        wavelet_filter_1d, 0, array3d, wavelet, level, "soft"
    )
    if out.shape[0] == array3d.shape[0] + 1:
        out = out[:-1]
    return out


def read_img(path, buf):
    stmfile = ps.STMfile(path)
    out = normalize_img(
        stmfile.get_buffers([buf])[buf],
        0,
        2 ** 16 - 1,
        from_vmin=-32768,
        from_vmax=32767,
    )
    return out


def get_bufset(paths, bufs):
    out = {}
    for buf in bufs:
        out[buf] = np.array([read_img(path, buf) for path in paths])
    return out


def img_process(img, buf, gauss=None):
    """Will preserve the dtype of the np array and won't normalize min max"""
    # out = np.copy(img)
    out = copy.deepcopy(img)

    if buf < 3:
        out = subtract_plane(out)
    if gauss is not None:
        out = gaussian(out, sigma=gauss)
    return out.astype("float")


def img_post_process(
    img, destripe=False, wavelet="sym10", level=None, sc=None
):
    """Will return a 16-bit image. For now, buf is a placeholder to make other process easier."""
    # out = np.copy(img)
    out = copy.deepcopy(img)
    out = normalize_img(out, 0, 2 ** 16 - 1).astype("uint16")
    if destripe:
        out = stripe.filter_streaks(
            out, sigma=[20, 20], level=level, wavelet=wavelet
        )

    # img = rescale_intensity(
    #     img, in_range=(np.percentile(img, 0.3), np.percentile(img, 99.7))
    # )
    if sc is not None:
        out = np.clip(
            out, np.percentile(out, sc[0]), np.percentile(out, sc[1])
        )
    out = normalize_img(out, 0, 2 ** 16 - 1).astype("uint16")
    return out


def img_post_process2(
    img, destripe=False, wavelet="sym7", level=None, sc=None
):
    """Perform image processing to remove the stripes and stretch constrast,
    if necessary. Output will be a 16-bit image normalized to (0, 2**16-1)
    
    Arguments:
        img {numpy.2darray} -- Image to be destriped
    
    Keyword Arguments:
        destripe {bool} -- Destripe Image or not (default: {False})
        wavelet {str} -- PyWavelet's discrete wavelet to be used for destriping (default: {sym7})
        level {int} -- Maximum level of wavelet decomposition (default: {maximum level determined 
            by pywt.dwt_max_level})
        sc {(float, float)} -- tuple of floats that specify the lower and 
        upper percentiles of img intensity to stretch contrast(default: {None})
    
    Returns:
        numpy.2darray -- Post-processed array of img. dtype=uint16
    """
    # out = np.copy(img)
    out = copy.deepcopy(img)
    if destripe:
        out = destripe_by_wavelet_svd(out, wavelet=wavelet, level=level)

    # img = rescale_intensity(
    #     img, in_range=(np.percentile(img, 0.3), np.percentile(img, 99.7))
    # )
    if sc is not None:
        out = np.clip(
            out, np.percentile(out, sc[0]), np.percentile(out, sc[1])
        )
    # out = normalize_img(out, 0, 2 ** 16 - 1).astype("uint16")
    return out


def img_process_set(imset, bufs, gauss=None):
    """ Specific routine for processing the image subset of data type: imset -> {buf#1: [img1, img2, ...], buf#2: [img1, img2, ...]}
    #                                                                                 <-- subset -->
    """
    # out = imset.copy()
    out = {}
    for buf in bufs:
        out[buf] = np.array([img_process(im, buf, gauss) for im in imset[buf]])
    return out


def img_process_set2(imset, gauss=None):
    """Specific routine for processing the full image set of data type: imset -> {buf#1: [img1, img2, ...], buf#2: [img1, img2, ...]}
    """
    # out = np.copy(imset).item()
    out = {}
    for key in imset:
        out[key] = np.array([img_process(im, key, gauss) for im in imset[key]])
    return out


def img_post_process_set(
    imset, bufs, destripe=False, wavelet="sym7", level=None, sc=None
):
    """Specific routine for processing the image subset of data type: imset -> {buf#1: [img1, img2, ...], buf#2: [img1, img2, ...]}
    #                                                                                 <-- subset -->
    """
    out = {}
    for buf in bufs:
        out[buf] = np.array(
            [
                img_post_process(
                    im, destripe=destripe, wavelet=wavelet, level=level, sc=sc
                )
                for im in imset[buf]
            ]
        )
    return out


def img_post_process_set2(
    imset, destripe=False, wavelet="sym7", level=None, sc=None
):
    """Specific routine for processing the full image set of data type: imset -> {buf#1: [img1, img2, ...], buf#2: [img1, img2, ...]}"""
    # out = np.copy(imset).item()
    out = {}
    for key in imset:
        out[key] = np.array(
            # [
            #     img_post_process(im, destripe=destripe, sc=sc)
            #     for im in imset[key]
            # ]
            [
                img_post_process2(
                    im, destripe=destripe, wavelet=wavelet, level=level, sc=sc
                )
                for im in imset[key]
            ]
        )
    return out


def normalize_set_to_uint16(imset):
    out = {}
    for key in imset:
        out[key] = np.array(
            [
                normalize_img(im, 0, 2 ** 16 - 1).astype("uint16")
                for im in imset[key]
            ]
        )
    return out


def register_bufferset(imset, bufs, secondarySet=None, filter=False):
    # out = np.copy(imset).item()
    out = copy.deepcopy(imset)
    filtered_set = img_post_process_set(out, bufs, destripe=True, sc=None)
    for buf in bufs:
        if filter:
            out[buf] = np.array(
                imreg(
                    out[buf],
                    secondarySet=filtered_set[buf],
                    upsample_factor=100,
                )
            )
        else:
            out[buf] = np.array(
                imreg(out[buf], secondarySet=secondarySet, upsample_factor=100)
            )
    return out


def register_bufferset2(imset, secondarySet=None, filter=False):
    # out = np.copy(imset).item()
    out = copy.deepcopy(imset)
    filtered_set = img_post_process_set2(out, destripe=True)
    for buf in out:
        if filter:
            out[buf] = np.array(
                imreg(
                    out[buf],
                    secondarySet=filtered_set[buf],
                    upsample_factor=100,
                )
            )
        else:
            out[buf] = np.array(
                imreg(out[buf], secondarySet=secondarySet, upsample_factor=100)
            )
    return out


def zoom_buffer_set(imset):
    """Increase the DPI of the images by 6 times"""
    # out = np.copy(imset).item()
    out = copy.deepcopy(imset)
    for key in out:
        out[key] = np.array(
            [zoom(im, zoom=6, order=0, mode="constant") for im in out[key]]
        )
    return out


def annotate_image(img, text, textcolor="black", sc=(1, 99)):
    """The annotation will return an image of the same dtype"""
    # out = np.copy(img)
    out = copy.deepcopy(img)
    dtype = out.dtype
    if dtype.kind == "u":
        mn = 0
        mx = 2 ** (dtype.itemsize * 8) - 1
    else:
        mn = -2 ** (dtype.itemsize * 7)
        mx = 2 ** (dtype.itemsize * 7) - 1
    # out = np.uint16(normalize_img(out, 0, 2 ** 16 - 1))
    # out = np.uint16(out)
    # img = np.uint16(img)
    org = (int(0.08 * out.shape[0]), int(0.12 * out.shape[1]))

    # # coordinates = (int(0.4 * new_imset[0].shape[0]), int(0.12 * new_imset[0].shape[1]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = np.min(out.shape) / (80 / 0.3)
    coordinates = (org[0], org[1])
    # 3 for " ps"

    if textcolor == "black":
        fontColor = (mn, mn, mn)
    elif textcolor == "white":
        fontColor = (mx, mx, mx)
        # fontColor = (2 ** 16 - 1, 2 ** 16 - 1, 2 ** 16 - 1)
    lineType = 3
    thickness = 1
    cv2.putText(
        out, text, coordinates, font, fontScale, fontColor, lineType, thickness
    )
    out = rgb2gray(out)
    # if sc is not None:
    #     sc = list(sc)
    #     out = np.clip(
    #         out, np.percentile(out, sc[0]), np.percentile(out, sc[1])
    #     )
    # out = np.uint16(normalize_img(out, 0, 2 ** 16 - 1))
    # out = np.uint16(out)
    return out


def annotate_imset(imset, textlist, textcolor="black", sc=(1, 99)):
    """imset is a list of images, unlike the other routines. The annotation will return a normalized uint16 image"""
    assert len(textlist) == len(
        imset
    ), "Image set and textlist have different lengths."
    new_imset = []
    # imset_cp = np.copy(imset)
    imset_cp = copy.deepcopy(imset)
    for i in range(len(textlist)):
        new_imset.append(
            annotate_image(imset_cp[i], textlist[i], textcolor, sc)
        )
    return np.array(new_imset)


def annotate_bufset(bufset, textlist, textcolor="black", sc=(1, 99)):
    # out = np.copy(bufset).item()
    out = copy.deepcopy(bufset)
    for key in out:
        out[key] = annotate_imset(out[key], textlist, textcolor, sc)
    return out


def fit_colormap(img, cmap="gray_r"):
    """The colormap fit will return a uint16 image"""
    # out = np.copy(img)
    out = copy.deepcopy(img)
    cm = mpl.cm.get_cmap(cmap, 2 ** 16)
    out = normalize_img(out, 0, 1)
    # out = out / (2**16-1)
    out = np.uint16(cm(out) * (2 ** 16 - 1))
    return out


def fit_colormap_to_imset(imset, cmap="gray_r"):
    """imset is a list of images, unlike the other routines. The colormap fit will return a uint16 image"""
    new_imset = []
    # imset_cp = np.copy(imset)
    imset_cp = copy.deepcopy(imset)
    mx = imset_cp.max()
    cm = mpl.cm.get_cmap(cmap, 2 ** 16)
    for img in imset_cp:
        # img = img.astype("float") / mx
        # new_imset.append(np.uint16(cm(img) * (2**16-1)))
        new_imset.append(fit_colormap(img, cmap))
    return np.array(new_imset)


def fit_colormap_to_bufset(bufset, lockin_map="gray_r"):
    # out = np.copy(bufset).item()
    out = copy.deepcopy(bufset)
    for key in out:
        if key < 3:
            out[key] = fit_colormap_to_imset(out[key], "afmhot")
        else:
            out[key] = fit_colormap_to_imset(out[key], lockin_map)
    return out


def pad_imset(imset):
    # imset_cp = np.copy(imset)
    imset_cp = copy.deepcopy(imset)
    shapes = np.array([[im.shape[0], im.shape[1]] for im in imset_cp])
    max_i = shapes[:, 0].max()
    max_j = shapes[:, 1].max()
    new_imset = []
    for im in imset_cp:
        im = np.pad(
            im,
            pad_width=(
                (
                    int((max_i - im.shape[0]) / 2),
                    int((max_i - im.shape[0]) / 2)
                    + int((max_i - im.shape[0]) % 2),
                ),
                (
                    int((max_j - im.shape[1]) / 2),
                    int((max_j - im.shape[1]) / 2)
                    + int((max_j - im.shape[1]) % 2),
                ),
            ),
            mode="reflect",
        )
        new_imset.append(im)
    return np.array(new_imset)


def rescale_image(im, scale):
    """Scaling factor must be greater than or equal to 1."""
    assert scale >= 1, "Scaling factor must be greater than or equal to 1."
    # out = np.copy(im)
    out = copy.deepcopy(im)
    old_shape = out.shape
    out = zoom(out, scale, order=0)
    new_shape = out.shape
    i_0 = int((new_shape[0] - old_shape[0]) / 2)
    i_1 = int((new_shape[0] - old_shape[0]) / 2) + old_shape[0] + 1
    j_0 = int((new_shape[1] - old_shape[1]) / 2)
    j_1 = int((new_shape[1] - old_shape[1]) / 2) + old_shape[1] + 1
    out = out[i_0:i_1, j_0:j_1]
    return out


def standardize_data(bufset):
    bufset = copy.deepcopy(bufset)
    for key, val in bufset.items():
        val = [(img - img.mean()) / img.std() for img in val]
        val = np.array(val)
        bufset[key] = (val - val.mean()) / val.std()
    return bufset


def get_src_data(paths, buffers, gauss=None, sec=False):
    """To prepare the different set of images for the user's options.
    Purpose is to break up the main function."""

    # Get the buffer_set in the form buffer_set -> {buf#1: [img1, img2, ...], buf#2: [img1, img2, ...]}
    buffer_set_raw = get_bufset(paths, bufs=buffers)

    # Get the topographic images for registration, if sec is not None
    topos = get_bufset(paths, bufs=[1])  # dict
    topos = img_process_set2(topos, gauss=None)  # dict
    topos = img_post_process_set2(topos, destripe=True, sc=(2, 98))[1]  # array

    # Hardcoding for one example: 08_24_19 (Movie 16)
    # for key in buffer_set_raw:
    #     i = int(0.4 * buffer_set_raw[key][0].shape[0])
    #     j = int(0.6 * buffer_set_raw[key][0].shape[1])
    #     buffer_set_raw[key] = [
    #         im[i : im.shape[0], 0:j] for im in buffer_set_raw[key]
    #     ]

    # Hardcoding for one example: 09_13_19 (Movie 26)
    # All but tip change
    # for key in buffer_set_raw:
    #     buffer_set_raw[key] = np.array(
    #         [im[15:, :] for im in buffer_set_raw[key]]
    #     )

    # left dot
    # for key in buffer_set_raw:
    #     buffer_set_raw[key] = np.array(
    #         [im[40:76, 6:50] for im in buffer_set_raw[key]]
    #     )

    # right dot
    # for key in buffer_set_raw:
    #     buffer_set_raw[key] = np.array(
    #         [im[43:82, 57:] for im in buffer_set_raw[key]]
    #     )

    # topos = np.array([im[40:76, 6:50] for im in topos])
    # topos = np.array([im[43:82, 57:] for im in topos])

    # Random gold surface (for background)
    # for key in buffer_set_raw:
    #     buffer_set_raw[key] = np.array(
    #         [im[77:, :] for im in buffer_set_raw[key]]
    #     )
    # topos = np.array([im[77:, :] for im in topos])

    # Hardcoding for one example: 10_05_19 (Movie 31)
    # for key in buffer_set_raw:
    #     i = int(0.4 * buffer_set_raw[key][0].shape[0])
    #     j = int(0.6 * buffer_set_raw[key][0].shape[1])
    #     buffer_set_raw[key] = [im[i:, :j] for im in buffer_set_raw[key]]

    # Hardcoding for one example: 10_06_19 (Movie 32)
    # for key in buffer_set_raw:
    #     i = int(0.3 * buffer_set_raw[key][0].shape[0])
    #     j = int(0.5 * buffer_set_raw[key][0].shape[1])
    #     buffer_set_raw[key] = [im[i:, j:] for im in buffer_set_raw[key]]

    # Hardcoding for one example: 10_07_19 (Movie 33)
    # for key in buffer_set_raw:
    #     buffer_set_raw[key] = [im[33:88, 10:80] for im in buffer_set_raw[key]]
    # topos = np.array([im[33:88, 10:80] for im in topos])

    # Hardcoding for one example: 10_08_19 (Movie 34)
    # for key in buffer_set_raw:
    #     i = int(0.5 * buffer_set_raw[key][0].shape[0])
    #     j = int(0.4 * buffer_set_raw[key][0].shape[1])
    #     buffer_set_raw[key] = [im[:i, j:] for im in buffer_set_raw[key]]

    # Hardcoding for one example: 10_09_19 (Movie 36)
    # for key in buffer_set_raw:
    #     buffer_set_raw[key] = [im[26:75, :83] for im in buffer_set_raw[key]]
    # topos = np.array([im[26:75, :83] for im in topos])

    # Hardcoding for one example: 10_21_19 (Movie 42, Part 2)
    # for key in buffer_set_raw:
    #     buffer_set_raw[key] = [im[12:, 5:45] for im in buffer_set_raw[key]]
    # topos = np.array([im[12:, 5:45] for im in topos])

    buffer_set_processed = img_process_set2(buffer_set_raw, gauss=gauss)

    # Register the images
    if sec:
        reg_buffer_set = register_bufferset2(
            buffer_set_processed, secondarySet=topos
        )
    else:
        reg_buffer_set = register_bufferset2(buffer_set_processed, filter=True)

    return buffer_set_raw, buffer_set_processed, reg_buffer_set, topos


def get_normalized_buffer_set(reg_buffer_set):
    """Put in a sorted registered buffer set to get the normalized set"""
    # src = np.copy(reg_buffer_set).item()
    src = copy.deepcopy(reg_buffer_set)
    out = {}
    for key in src:
        out[key] = np.array(
            [src[key][i] - src[key][0] for i in range(1, len(src[key]))]
        )
        # out[key] = np.array(
        #     [
        #         compare_ssim(src[key][0], src[key][i], full=True)[1]
        #         for i in range(1, len(src[key]))
        #     ]
        # )
    return out


def get_svd(reg_buffer_set, svd_num=10, ds=False):
    """Return an svd dictionary: {buf: [ [svd image set], U, D, V, svd_num ]"""
    assert isinstance(svd_num, int), "svd num must be integer"
    # src = np.copy(reg_buffer_set).item()
    src = copy.deepcopy(reg_buffer_set)
    src = img_post_process_set2(src, destripe=ds, sc=None)
    src = zoom_buffer_set(src)
    for key in src:
        svd_arr = np.array(src[key])
        U, D, V = np.linalg.svd(svd_arr)
        d_rec = (U[:, :, :svd_num] * D[:, None, :svd_num]) @ V[
            :, :svd_num, :
        ]  # @ = matrix mult.
        weight_perc = [
            np.sum(D[i, :svd_num] ** 2) / np.sum(D[i] ** 2)
            for i in range(D.shape[0])
        ]
        accu = np.mean(weight_perc)
        stdev = np.std(weight_perc)
        print(f"Average SVD reconstruction accuracy: {accu}")
        print(f"Standard deviation: {stdev}")
        src[key] = [d_rec, U, D, V, svd_num]
    return src


def get_svd2(reg_buffer_set, svd_num=10, ds=False, wavelet="sym7", level=None):
    """Return an svd dictionary: {buf: [ [svd image set], U, D, V, svd_num ]. get_svd2 differs from get_svd in that it flatten the image stack into a matrix in which a row = an image"""
    assert isinstance(svd_num, int), "svd num must be integer"
    # src = np.copy(reg_buffer_set).item()
    src = copy.deepcopy(reg_buffer_set)
    src = img_post_process_set2(
        src, destripe=ds, sc=None, wavelet=wavelet, level=level
    )
    src = zoom_buffer_set(src)
    for key in src:
        svd_arr = np.array(src[key])
        svd_arr_rs = svd_arr.reshape(
            (svd_arr.shape[0], svd_arr.shape[1] * svd_arr.shape[2])
        )
        U, D, V = np.linalg.svd(svd_arr_rs, full_matrices=False)
        # d_rec = np.dot(
        #     U[:, :svd_num], np.dot(np.diag(D[:svd_num]), V[:svd_num, :])
        # )  # @ = matrix mult.
        # d_rec = U[:, :svd_num] @ np.diag(D[:svd_num]) @ V[:svd_num, :]
        d_rec = U[:, 1:svd_num] @ np.diag(D[1:svd_num]) @ V[1:svd_num, :]
        # d_rec = U[:, 1:] @ np.diag(D[1:]) @ V[1:, :]
        d_rec = d_rec.reshape(
            (svd_arr.shape[0], svd_arr.shape[1], svd_arr.shape[2])
        )
        # weight_perc = np.sum(D[0] ** 2) / np.sum(D ** 2)
        # accu = np.mean(weight_perc)
        # stdev = np.std(weight_perc)
        # print(f"U0 weight: {weight_perc}")
        # print(f"Standard deviation: {stdev}")

        V = V.reshape((V.shape[0], svd_arr.shape[1], svd_arr.shape[2]))
        src[key] = [d_rec, U, D, V, svd_num]
    return src


def get_wavelet_svd(
    reg_buffer_set, wavelet="sym7", level=None, svdnum=None, mode="symmetric"
):
    # assert isinstance(svdnum, int), "svd num must be integer"
    src = copy.deepcopy(reg_buffer_set)
    src = zoom_buffer_set(src)
    for key in src:
        svd_arr = np.array(src[key])
        svd_arr_rs = svd_arr.reshape(
            (svd_arr.shape[0], svd_arr.shape[1] * svd_arr.shape[2])
        )
        coeffs = pywt.wavedec2(
            svd_arr_rs, wavelet=wavelet, level=level, mode=mode
        )
        approx = coeffs[0]
        detail = coeffs[1:]
        coeffs_filt = [approx]

        for nthlevel in detail:
            ch, cv, cd = nthlevel
            fcv = fftpack.rfft(cv)
            # print(cv.shape)
            # print(fcv.shape)
            U, D, V = np.linalg.svd(fcv, full_matrices=False)
            if svdnum is None:
                svdnum = int(0.25 * np.min(fcv.shape))
            # fcv_filt = np.dot(
            #     np.matrix(U[:, svdnum:]),
            #     np.dot(np.diag(D[svdnum:]), np.matrix(V[svdnum:, :])),
            # )
            fcv_filt = U[:, svdnum:] @ np.diag(D[svdnum:]) @ V[svdnum:, :]
            cv_filt = fftpack.irfft(fcv_filt)
            coeffs_filt.append((ch, cv_filt, cd))
        svd_arr_filt = pywt.waverec2(coeffs_filt, wavelet=wavelet, mode=mode)
        print("past wavelet_svd reconstruction")
        svd_arr_filt = np.array(svd_arr_filt)
        print(svd_arr_filt.shape)
        print(svd_arr.shape)
        try:
            svd_arr_filt = svd_arr_filt[:-1].reshape(
                (svd_arr.shape[0], svd_arr.shape[1], svd_arr.shape[2])
            )
        except ValueError:
            svd_arr_filt = svd_arr_filt.reshape(
                (svd_arr.shape[0], svd_arr.shape[1], svd_arr.shape[2])
            )
        u, d, v = np.linalg.svd(
            svd_arr_filt.reshape(
                (
                    svd_arr_filt.shape[0],
                    svd_arr_filt.shape[1] * svd_arr_filt.shape[2],
                )
            ),
            full_matrices=False,
        )

        v = v.reshape(
            (v.shape[0], svd_arr_filt.shape[1], svd_arr_filt.shape[2])
        )

        print("past 2nd svd")

        print("past reshape")
        src[key] = [svd_arr_filt, u, d, v, svdnum]
    return src


def plot_svd(t, buf, U, D, V, svd_num, normalized=False, fg=None, bg=None):
    """A = UDV^T. All matrices have this property. I used this for my pump-probe analysis"""

    # With the function get_svd2, all image pixels are decomposed with time. We have:
    # A(time, location) = U(time, time) * D(time, location) * V(location,location)
    # U: relates time to "features" (broad term, some kind of concepts)
    # V: relates the location to the concepts
    # By plotting U column vectors with most weights, we can conceptualize how the "features" change against time.
    suffix = ""
    if normalized:
        suffix += "_normalized"
    if fg is not None:
        suffix += "_fg"
    elif bg is not None:
        suffix += "_bg"
    fig, ax = plt.subplots()
    for i in range(svd_num):
        ax.cla()
        # ax.minorticks_on()
        ax.plot(t, U[:, i])
        ax.set_xscale("symlog")
        ax.set_xlabel("Time, ps")
        # ax.set_xlabel("Bias voltage")
        ax.set_ylabel(f"U{i}")
        xlog_tic = mtick.LogLocator(base=10, subs=(0.2, 0.4, 0.6, 0.8))
        ax.xaxis.set_minor_locator(xlog_tic)
        ax.xaxis.set_minor_formatter(mtick.NullFormatter())
        print(f"Weight of U{i}: {D[i]**2 / np.sum(D**2)}")
        fig.savefig(
            f"U{i}_buf{buf}{suffix}_vs_time.tiff",
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    plt.close(fig)

    # Spatial components: V
    V_towrite = np.array(V)
    V_towrite = np.clip(
        V_towrite, np.percentile(V_towrite, 1), np.percentile(V_towrite, 99)
    )
    V_towrite = normalize_img(V_towrite, 0, 2 ** 8 - 1).astype("uint8")
    textlist = [""]
    textlist.extend(
        [
            f"Weight: {(1e6 * val):.1f} ppm"
            for val in (D ** 2 / np.sum(D ** 2))
        ][1:]
    )
    V_towrite = annotate_imset(V_towrite, textlist)
    imageio.mimwrite(f"V_buf{buf}{suffix}_vs_time.mov", V_towrite, fps=2)

    # Movie of each component, up to svd_num
    V_reshaped = V.reshape((V.shape[0], V.shape[1] * V.shape[2]))
    print(V_reshaped.shape)
    for i in range(svd_num):
        component = (
            D[i] * np.atleast_2d(U[:, i]).T @ np.atleast_2d(V_reshaped[i, :])
        )
        component = component.reshape((V.shape[0], V.shape[1], V.shape[2]))
        component = np.clip(
            component,
            np.percentile(component, 1),
            np.percentile(component, 99),
        )
        component = normalize_img(component, 0, 2 ** 8 - 1).astype("uint8")
        component = annotate_imset(component, [f"{val} ps" for val in t])
        imageio.mimwrite(
            f"Component_{i}_buf{buf}_{suffix}.mov", component, fps=2
        )


def save_movie_with_matplotlib(
    reg_buf, num_fn, normalize=False, svd=False, cmap="gray_r"
):
    reg_buf_cp = copy.deepcopy(reg_buf)
    suffix = ""
    if svd:
        suffix += "_svd"
    if normalize:
        suffix += "_normalized"
    for key in reg_buf_cp:
        # clim = [np.percentile(reg_buf_cp[key], 2), np.percentile(reg_buf_cp[key], 98)]
        clim = [np.min(reg_buf_cp[key]), np.max(reg_buf_cp[key])]
        fig, ax = plt.subplots()
        fig.tight_layout()
        # ax.set_axis_off()
        frame_list = []
        savename = f"./Movie_buf{key}{suffix}.mov"
        for frame in reg_buf_cp[key]:
            ax.cla()
            ax.axis("off")
            im = ax.imshow(frame, cmap=cmap)
            im.set_clim(clim)
            fig.savefig(
                "./temp.tiff",
                bbox_inches="tight",
                pad_inches=0,
                pil_kwargs={"compression": "tiff_lzw"},
            )
            frame_list.append(imageio.imread("./temp.tiff", format="TIFF-FI"))
        imageio.mimwrite(savename, frame_list, fps=2)
        remove("./temp.tiff")
        plt.close(fig)


def save_movie_and_frame(reg_buf, num_fn, normalize=False, svd=False):
    # reg_buf_cp = np.copy(reg_buf).item()
    reg_buf_cp = copy.deepcopy(reg_buf)
    reg_buf_cp = {
        key: normalize_img(val, 0, 2 ** 8 - 1).astype("uint8")
        for (key, val) in reg_buf_cp.items()
    }
    suffix = ""
    if svd:
        suffix += "_svd"
    if normalize:
        suffix += "_normalized"
    for key in reg_buf_cp:
        # write the movies
        savename = f"./Movie_buf{key}{suffix}.mov"
        imageio.mimwrite(savename, reg_buf_cp[key], fps=2)

        # write the frames
        [
            imageio.imwrite(
                f"./frame_buf{key}_{num_fn[i]}{suffix}.tiff",
                reg_buf[key][i],
                format="TIFF-FI",
            )
            for i in range(len(reg_buf_cp[key]))
        ]


def get_inputs_and_filenames_from_logfile(logfile, frames):
    info = []
    for line in reversed(list(open(logfile, "r"))):
        info.append(line.rstrip())
    inputs = [inf.split(",")[0].strip() for inf in info[:frames]]
    inputs = list(reversed(inputs))
    fn = [inf.split(",")[1].strip() for inf in info[:frames]]
    fn = list(reversed(fn))
    return inputs, fn


# def get_inputs_and_filenames_from_logfile(logfile):
#     info = []
    


def create_movies(reg_buffer_set, args, fg=None, bg=None):
    """Create movies out of the registered src data (that hasn't been altered to image form."""
    paths = args.input
    buffers = args.buffers
    logfile = args.logfile
    frames = args.frames
    register = args.register
    cmap = args.colormap
    ds = args.destripe
    gauss = args.gauss
    lockin = args.lockin
    movie = args.movie
    sc = args.stretch_contrast
    fn = args.filenames
    textcolor = args.textcolor
    normalize = args.normalize
    sec = args.secondarySet
    svd_num = args.svd
    ds_level = args.level
    ds_wavelet = args.wavelet
    svd_level = args.level_svd
    svd_wavelet = args.wavelet_svd
    route2 = args.route2

    if logfile is not None and frames is not None:
        paths, fn = get_inputs_and_filenames_from_logfile(
            logfile, frames
        )  # override the regular inputs and fn

    assert (
        fn is not None
    ), "Creating movies requires user to specify filenames for sorting frames."

    # reg_buf = np.copy(reg_buffer_set).item()
    reg_buf = copy.deepcopy(reg_buffer_set)
    num_fn = [float(f) for f in fn]
    # print(num_fn)
    # sorting the register buffers according to the filenames
    for key in reg_buf:
        # new_imset = np.copy(reg_buf[key])
        new_imset = copy.deepcopy(reg_buf[key])
        # new_imset = 
        new_imset = [
            x
            for _, x in sorted(
                zip(num_fn, new_imset), key=lambda pair: pair[0]
            )
        ]
        reg_buf[key] = new_imset

    # sorting the filenames itself for future use
    num_fn = np.sort(num_fn)
    # print(num_fn)

    # reg_buf = img_post_process_set2(reg_buf, destripe=ds, sc=None, wavelet=ds_wavelet, level=ds_level)

    # The case where the normalize flag is True, some modification to the set is done
    if normalize:
        reg_buf = get_normalized_buffer_set(reg_buf)
        num_fn = num_fn[1:]

    reg_buf = standardize_data(reg_buf)

    # get svd set for next processing before reg_buf is changed
    if svd_num is not None:
        if fg is not None:
            for_svd = {
                key: np.array([im * fg for im in val])
                for (key, val) in reg_buf.items()
            }
            reg_svd = get_svd2(for_svd, svd_num=svd_num, ds=ds)
        elif bg is not None:
            for_svd = {
                key: np.array([im * bg for im in val])
                for (key, val) in reg_buf.items()
            }
            reg_svd = get_svd2(for_svd, svd_num=svd_num, ds=ds)
        else:
            reg_svd = get_svd2(reg_buf, svd_num=svd_num, ds=ds)

    # do some post processing on the buffer set. Steps after this
    # normalizes and change dtype of images (cannot take for data operation)

    reg_buf = img_post_process_set2(reg_buf, destripe=ds, sc=None)

    ############## TESTING ################
    # reg_buf = {key: wavelet_filter_3d(np.array(val), wavelet=ds_wavelet, level=ds_level) for (key, val) in reg_buf.items()}

    #######################################
    # reg_buf = normalize_set_to_uint16(reg_buf)

    # increase the dpi on the buffer set

    reg_buf = zoom_buffer_set(reg_buf)

    ###################################### testing
    # if normalize:
    #     reg_buf = get_normalized_buffer_set(reg_buf)
    #     num_fn = num_fn[1:]

    ######################################################

    # save_movie_with_matplotlib(reg_buf, num_fn, normalize=normalize, cmap="gray_r")

    # annotate buffer set
    reg_buf = {
        key: normalize_img(
            np.clip(val, np.percentile(val, sc[0]), np.percentile(val, sc[1])),
            0,
            2 ** 16 - 1,
        ).astype("uint16")
        for (key, val) in reg_buf.items()
    }

    # reg_buf = normalize_set_to_uint16(reg_buf)

    reg_buf = annotate_bufset(
        reg_buf, [str(f) + " ps" for f in num_fn], textcolor=textcolor
    )

    # fit color to the buffer set
    reg_buf = fit_colormap_to_bufset(reg_buf, lockin_map=cmap)

    # Create the movie and write the frames to image files
    save_movie_and_frame(reg_buf, num_fn, normalize=normalize)

    # SVD
    if svd_num is not None:
        reg_svd_sim = {}
        for key in reg_svd:
            reg_svd_sim[key] = reg_svd[key][0]  # the reconstructed array

        # reg_svd_sim2 = np.copy(reg_svd_sim).item()
        reg_svd_sim2 = copy.deepcopy(reg_svd_sim)

        # post processing on the svd set
        reg_svd_sim = img_post_process_set2(reg_svd_sim, destripe=False, sc=sc)

        # annotate SVD set
        reg_svd_sim = {
            key: normalize_img(
                np.clip(
                    val, np.percentile(val, sc[0]), np.percentile(val, sc[1])
                ),
                0,
                2 ** 16 - 1,
            ).astype("uint16")
            for (key, val) in reg_svd_sim.items()
        }

        reg_svd_sim = annotate_bufset(
            reg_svd_sim,
            [str(f) + " ps" for f in num_fn],
            textcolor=textcolor,
            sc=sc,
        )

        # fit color to the svd set
        reg_svd_sim = fit_colormap_to_bufset(reg_svd_sim, lockin_map=cmap)

        # Create the movie and write the frames of svd to image files
        save_movie_and_frame(
            reg_svd_sim, num_fn, normalize, svd=svd_num is not None
        )

        # Save one with normalization for each frame

        reg_svd_sim2 = {
            key: [
                normalize_img(
                    np.clip(
                        im, np.percentile(im, sc[0]), np.percentile(im, sc[1])
                    ),
                    0,
                    2 ** 16 - 1,
                ).astype("uint16")
                for im in val
            ]
            for (key, val) in reg_svd_sim2.items()
        }

        reg_svd_sim2 = annotate_bufset(
            reg_svd_sim2,
            [str(f) + " ps" for f in num_fn],
            textcolor=textcolor,
            sc=sc,
        )

        # fit color to the svd set
        reg_svd_sim2 = fit_colormap_to_bufset(reg_svd_sim2, lockin_map=cmap)

        # Create the movie and write the frames of svd to image files
        save_movie_and_frame(
            reg_svd_sim2, num_fn, normalize, svd=svd_num is not None
        )

        # Plot image average against time
        # suffix = ""
        # if normalize:
        #     suffix += "_normalized"
        # fig, ax = plt.subplots()
        # for key in reg_svd_sim:
        #     # imset = np.copy(reg_svd_sim2[key])
        #     imset = copy.deepcopy(reg_svd_sim2[key])
        #     imset = np.reshape(
        #         imset, (imset.shape[0], imset.shape[1] * imset.shape[2])
        #     )
        #     ax.cla()
        #     ax.plot(num_fn, np.mean(imset, axis=1))
        #     ax.set_xscale("symlog")
        #     ax.set_xlabel("Time, ps")

        #     ax.set_ylabel("Normalized SVD Single Values")
        #     fig.savefig(
        #         f"SValImgAvg_buf{key}{suffix}_vs_time.tiff",
        #         bbox_inches="tight",
        #         pil_kwargs={"compression": "tiff_lzw"},
        # )

        # Plot and save SVD weights
        for key in reg_svd:
            print("Plotting SVD for buffer ", key)
            plot_svd(
                num_fn,
                key,
                *reg_svd[key][1:],
                normalized=normalize,
                fg=fg,
                bg=bg,
            )


def create_movies_route2(reg_buffer_set, args):
    """Create movies out of the registered src data (that hasn't been altered to image form."""
    paths = args.input
    buffers = args.buffers
    logfile = args.logfile
    frames = args.frames
    register = args.register
    cmap = args.colormap
    ds = args.destripe
    gauss = args.gauss
    lockin = args.lockin
    movie = args.movie
    sc = args.stretch_contrast
    fn = args.filenames
    textcolor = args.textcolor
    normalize = args.normalize
    sec = args.secondarySet
    svd_num = args.svd
    ds_level = args.level
    ds_wavelet = args.wavelet
    svd_level = args.level_svd
    svd_wavelet = args.wavelet_svd
    route2 = args.route2

    if logfile is not None and frames is not None:
        paths, fn = get_inputs_and_filenames_from_logfile(
            logfile, frames
        )  # override the regular inputs and fn

    assert (
        fn is not None
    ), "Creating movies requires user to specify filenames for sorting frames."

    reg_buf = copy.deepcopy(reg_buffer_set)
    num_fn = [float(f) for f in fn]

    # sorting the register buffers according to the filenames
    for key in reg_buf:
        new_imset = copy.deepcopy(reg_buf[key])
        new_imset = [
            x
            for _, x in sorted(
                zip(num_fn, new_imset), key=lambda pair: pair[0]
            )
        ]
        reg_buf[key] = new_imset

    # sorting the filenames itself for future use
    num_fn = np.sort(num_fn)

    reg_buf = img_post_process_set2(
        reg_buf, destripe=ds, wavelet=ds_wavelet, level=ds_level
    )

    wavelet_svd_reg_buf = get_wavelet_svd(
        reg_buf, wavelet=svd_wavelet, level=svd_level, svdnum=svd_num
    )
    print("past get_wavelet_svd")
    wavelet_svd_reg_buf_sim = copy.deepcopy(wavelet_svd_reg_buf)
    for key in wavelet_svd_reg_buf_sim:
        wavelet_svd_reg_buf_sim[key] = wavelet_svd_reg_buf_sim[key][0]

    print("past deepcopy")

    # The case where the normalize flag is True, some modification to the set is done

    if normalize:
        wavelet_svd_reg_buf_sim = get_normalized_buffer_set(
            wavelet_svd_reg_buf_sim
        )
        num_fn = num_fn[1:]

    # increase the dpi on the buffer set
    # wavelet_svd_reg_buf_sim = zoom_buffer_set(wavelet_svd_reg_buf_sim)

    wavelet_svd_reg_buf_sim = img_post_process_set2(
        wavelet_svd_reg_buf_sim,
        sc=None,
        destripe=ds,
        wavelet=ds_wavelet,
        level=ds_level,
    )

    wavelet_svd_reg_buf_sim = {
        key: normalize_img(
            np.clip(val, np.percentile(val, sc[0]), np.percentile(val, sc[1])),
            0,
            2 ** 16 - 1,
        ).astype("uint16")
        for (key, val) in wavelet_svd_reg_buf_sim.items()
    }
    print("past stretch contrast")

    # annotate buffer set
    wavelet_svd_reg_buf_sim = annotate_bufset(
        wavelet_svd_reg_buf_sim,
        [str(f) + " ps" for f in num_fn],
        textcolor=textcolor,
        sc=sc,
    )
    print("past annotation")

    # fit color to the buffer set
    wavelet_svd_reg_buf_sim = fit_colormap_to_bufset(
        wavelet_svd_reg_buf_sim, lockin_map=cmap
    )
    print("past color fit")
    # Create the movie and write the frames to image files
    save_movie_and_frame(wavelet_svd_reg_buf_sim, num_fn, normalize=normalize)
    print("past save movie")
    # Plot SVD

    for key in wavelet_svd_reg_buf:
        print("Plotting SVD for buffer ", key)
        plot_svd(
            num_fn, key, *wavelet_svd_reg_buf[key][1:], normalized=normalize
        )
    print("past plot svd")


def main3():
    args = _parseargs()
    paths = args.input
    buffers = args.buffers
    logfile = args.logfile
    frames = args.frames
    register = args.register
    cmap = args.colormap
    ds = args.destripe
    gauss = args.gauss
    lockin = args.lockin
    movie = args.movie
    sc = args.stretch_contrast
    fn = args.filenames
    textcolor = args.textcolor
    normalize = args.normalize
    sec = args.secondarySet
    svd_num = args.svd
    ds_level = args.level
    ds_wavelet = args.wavelet
    svd_level = args.level_svd
    svd_wavelet = args.wavelet_svd
    route2 = args.route2
    background = args.background
    foreground = args.foreground

    if foreground:
        background = False

    if logfile is not None and frames is not None:
        paths, fn = get_inputs_and_filenames_from_logfile(
            logfile, frames
        )  # override the regular inputs and fn

    # get src data
    buffer_set_raw, buffer_set_processed, reg_buffer_set, topos = get_src_data(
        paths, buffers, gauss=gauss, sec=sec
    )

    # standardize data
    # standardize_data(reg_buffer_set)

    # create movies, if required
    if movie:
        # Obtain topo and create a mask for foreground and background
        if foreground or background:

            # Register the images
            topos = imreg(topos, upsample_factor=100)

            fg = topos[0] > threshold_otsu(topos[0]) - 0.6 * topos[0].std()
            fg = clear_border(fg)
            fg = remove_small_objects(fg, min_size=9)
            bg = ~fg
            fg_save = normalize_img(fg, 0, 2 ** 16 - 1).astype("uint16")
            bg_save = normalize_img(bg, 0, 2 ** 16 - 1).astype("uint16")
            imageio.imwrite("foreground.tiff", fg_save, format="TIFF-FI")
            imageio.imwrite("background.tiff", bg_save, format="TIFF-FI")

        if not route2:
            if foreground:
                create_movies(reg_buffer_set, args, fg=fg, bg=None)
            elif background:
                create_movies(reg_buffer_set, args, fg=None, bg=bg)
            else:
                create_movies(reg_buffer_set, args, fg=None, bg=None)
        else:
            create_movies_route2(reg_buffer_set, args)
    else:
        buffer_set_post_processed = zoom_buffer_set(buffer_set_processed)
        buffer_set_post_processed = img_post_process_set2(
            buffer_set_post_processed,
            destripe=ds,
            sc=sc,
            wavelet=ds_wavelet,
            level=ds_level,
        )
        buffer_set_post_processed = fit_colormap_to_bufset(
            buffer_set_post_processed, lockin_map=cmap
        )
        for key in buffer_set_post_processed:
            # to_write = np.copy(buffer_set_post_processed[key])
            to_write = copy.deepcopy(buffer_set_post_processed[key])
            [
                imageio.imwrite(
                    join("./", split(paths[i])[-1] + f".buf{key}.tiff"),
                    to_write[i],
                    format="TIFF-FI",
                )
                for i in range(len(to_write))
            ]


def main2():
    args = _parseargs()
    paths = args.input
    buffers = args.buffers
    register = args.register
    cmap = args.colormap
    ds = args.destripe
    gauss = args.gauss
    lockin = args.lockin
    movie = args.movie
    sc = args.stretch_contrast
    fn = args.filenames
    textcolor = args.textcolor
    normalize = args.normalize
    sec = args.secondarySet
    svd_num = args.svd
    if movie:
        register = True

    # Get the buffer_set in the form buffer_set -> {buf#1: [img1, img2, ...], buf#2: [img1, img2, ...]}

    buffer_set = get_bufset(paths, bufs=buffers)

    # process the buffer_set but keep the original copy in case needed further down

    buffer_set2 = img_process_set(buffer_set, bufs=buffers, gauss=gauss)
    # buffer_set3 = img_post_process_set(buffer_set2, bufs=buffers, destripe=ds)

    # Prepare a topos set in case the user wants to register based on a secondarySet

    topos = get_bufset(paths, bufs=[1])
    topos = img_process_set(topos, bufs=[1], gauss=gauss)
    topos = img_post_process_set(topos, bufs=[1], destripe=ds, sc=(2, 98))[1]

    # Register the images

    if sec:
        reg_buffer_set = register_bufferset(
            buffer_set2, bufs=buffers, secondarySet=topos
        )
    else:
        reg_buffer_set = register_bufferset(
            buffer_set2, bufs=buffers, filter=True
        )

    # buffer_set, buffer_set2, reg_buffer_set = get_src_data(
    #     paths, buffers, gauss=gauss, sec=sec
    # )

    if movie:
        # Write the images and movies, if register
        for buf in buffers:
            new_imset = []
            new_imset_svd_process = []
            new_imset = np.copy(reg_buffer_set[buf])
            to_write = [
                zoom(img, 6, order=0, mode="constant") for img in new_imset
            ]
            if buf < 3:
                to_write = fit_colormap_to_imset(to_write, cmap=cmap)
            else:
                to_write = fit_colormap_to_imset(to_write, cmap="gray_r")
            if fn is not None:
                num_fn = [float(f) for f in fn]
                for i in range(len(to_write)):
                    path = join("./", fn[i] + f".buf{buf}.tiff")
                    imageio.imwrite(path, to_write[i], format="TIFF-FI")
                new_imset = [
                    x
                    for _, x in sorted(
                        zip(num_fn, new_imset), key=lambda pair: pair[0]
                    )
                ]
                num_fn = np.sort(num_fn)
                if normalize:
                    # first_ind = np.argmin(num_fn)
                    # first_img = np.copy(new_imset[first_ind])
                    # new_imset = [
                    #     im.astype("float") - first_img.astype("float")
                    #     for im in new_imset
                    # ]
                    new_imset = [im - new_imset[0] for im in new_imset[1:]]
                    num_fn = num_fn[1:]

                # Steps after this normalizes and change dtype of images (cannot take for data operation)

                new_imset_svd_process = np.copy(new_imset)

                new_imset = [
                    img_post_process(img, destripe=ds, sc=sc)
                    for img in new_imset
                ]

                # increase dpi
                new_imset = [
                    zoom(img, 6, order=0, mode="constant") for img in new_imset
                ]

                # single value decomposition
                if svd_num is not None:
                    new_imset_svd_process = [
                        img_post_process(img, destripe=ds, sc=None)
                        for img in new_imset_svd_process
                    ]
                    new_imset_svd_process = [
                        zoom(img, 6, order=0, mode="constant")
                        for img in new_imset_svd_process
                    ]
                    # new_imset_svd_process = [
                    #                 x
                    #                 for _, x in sorted(
                    #                     zip(num_fn, new_imset_svd_process), key=lambda pair: pair[0]
                    #                 )
                    #             ]

                    U, D, V = np.linalg.svd(np.array(new_imset_svd_process))
                    d_rec = (U[:, :, :svd_num] * D[:, None, :svd_num]) @ V[
                        :, :svd_num, :
                    ]  # @ = matrix mult.
                    weight_perc = [
                        np.sum(D[i, :svd_num] ** 2) / np.sum(D[i] ** 2)
                        for i in range(D.shape[0])
                    ]
                    accu = np.mean(weight_perc)
                    stdev = np.std(weight_perc)
                    print(f"Average SVD reconstruction accuracy: {accu}")
                    print(f"Standard deviation: {stdev}")
                    t = np.copy(num_fn)
                    # t = np.sort(t)

                    fig, ax = plt.subplots()
                    # ax.tick_params(axis='x', which="minor", bottom='off')
                    # ax.tick_params(axis='y', which="minor", left='off')
                    for i in range(svd_num):
                        ax.cla()
                        ax.minorticks_on()
                        ax.plot(t, D[:, i] / np.max(D))
                        ax.set_xscale("symlog")
                        ax.set_xlabel("Time, ps")

                        # ax.set_xlabel("Frame number")
                        ax.set_ylabel("Normalized SVD Single Values")
                        fig.savefig(
                            f"SVal{i}_vs_time.tiff",
                            bbox_inches="tight",
                            pil_kwargs={"compression": "tiff_lzw"},
                        )

                    svd_set = annotate_imset(
                        d_rec,
                        [str(f) + " ps" for f in num_fn],
                        textcolor=textcolor,
                        sc=sc,
                    )
                    # svd_set = [
                    #                 x
                    #                 for _, x in sorted(
                    #                     zip(num_fn, svd_set), key=lambda pair: pair[0]
                    #                 )
                    #             ]
                    if buf < 3:
                        svd_set = fit_colormap_to_imset(svd_set, cmap="afmhot")
                    else:
                        svd_set = fit_colormap_to_imset(svd_set, cmap="gray_r")
                    savename = f"./Movie_buf{buf}_{svd_num}_svd.mov"
                    if normalize:
                        savename = (
                            f"./Movie_buf{buf}_{svd_num}_svd_normalized.mov"
                        )
                    imageio.mimwrite(savename, svd_set, fps=2)

                new_imset = annotate_imset(
                    new_imset,
                    [str(f) + " ps" for f in num_fn],
                    textcolor=textcolor,
                    sc=sc,
                )
                # new_imset = [
                #     x
                #     for _, x in sorted(
                #         zip(num_fn, new_imset), key=lambda pair: pair[0]
                #     )
                # ]
                if buf < 3:
                    new_imset = fit_colormap_to_imset(new_imset, cmap="afmhot")
                else:
                    new_imset = fit_colormap_to_imset(new_imset, cmap="gray_r")
                savename = f"./Movie_buf{buf}.mov"
                if normalize:
                    savename = f"./Movie_buf{buf}_normalized.mov"
                imageio.mimwrite(savename, new_imset, fps=2)
                if normalize:
                    [
                        imageio.imwrite(
                            f"./frame_buf{buf}_{sorted(num_fn)[i]}_normalized.tiff",
                            new_imset[i],
                            format="TIFF-FI",
                        )
                        for i in range(len(new_imset))
                    ]
                else:
                    [
                        imageio.imwrite(
                            f"./frame_buf{buf}_{sorted(num_fn)[i]}.tiff",
                            new_imset[i],
                            format="TIFF-FI",
                        )
                        for i in range(len(new_imset))
                    ]
            else:
                for i in range(len(to_write)):
                    path = join("./", split(paths[i])[-1] + f".buf{buf}.tiff")
                    imageio.imwrite(path, to_write[i], format="TIFF-FI")

    else:
        buffer_set2 = img_post_process_set(
            buffer_set2, bufs=buffers, destripe=ds, sc=sc
        )
        for buf in buffers:
            to_write = np.copy(buffer_set2[buf])
            to_write = [
                zoom(img, 6, order=0, mode="constant") for img in to_write
            ]
            if buf < 3:
                to_write = fit_colormap_to_imset(to_write, cmap=cmap)
            else:
                to_write = fit_colormap_to_imset(to_write, cmap="gray_r")
            for i in range(len(to_write)):
                path = join("./", split(paths[i])[-1] + f".buf{buf}.tiff")
                imageio.imwrite(path, to_write[i], format="TIFF-FI")


def main():
    args = _parseargs()
    paths = args.input
    buffers = args.buffers
    register = args.register
    cmap = args.colormap
    ds = args.destripe
    gauss = args.gauss
    lockin = args.lockin
    movie = args.movie
    sc = args.stretch_contrast
    fn = args.filenames
    textcolor = args.textcolor
    normalize = args.normalize
    sec = args.secondarySet

    if movie:
        register = False

    stmfiles = []
    images = []
    for path in paths:
        f = ps.STMfile(path)
        stmfiles.append(f)
        images.append(
            f.get_height_buffers(buffers)
        )  # get a list of dictionaries

    for buf in buffers:
        for i, im in enumerate(images):
            _tmp = 0
            if buf > 3:
                _tmp = -im[buf].copy()
            else:
                _tmp = im[buf].copy()
            # im = (normalize_img(im, -1, 1))
            images[i][buf] = normalize_img(_tmp.copy(), -1, 1)

    # Image processing steps
    for buf in buffers:
        for i, im in enumerate(images):
            _tmp = 0
            _tmp = im[buf].copy()
            if buf < 3:
                _tmp = subtract_plane(_tmp)
            _tmp = normalize_img(_tmp, -1, 1)
            images[i][buf] = _tmp
    if ds:
        for buf in buffers:
            for i, im in enumerate(images):
                _tmp = 0
                _tmp = im[buf].copy()
                _tmp = np.uint16(normalize_img(_tmp, 0, 2 ** 16 - 1))
                _tmp = stripe.filter_streaks(
                    _tmp, sigma=[10, 20], level=2, wavelet="db3"
                )
                _tmp = normalize_img(_tmp, -1, 1)
                images[i][buf] = _tmp
    if gauss is not None:
        for buf in buffers:
            for i, im in enumerate(images):
                _tmp = 0
                _tmp = im[buf].copy()
                _tmp = normalize_img(_tmp, -1, 1)
                _tmp = gaussian(_tmp, sigma=gauss)
                _tmp = normalize_img(_tmp, -1, 1)
                images[i][buf] = _tmp
    if sc is not None:
        for buf in buffers:
            for i, im in enumerate(images):
                _tmp = 0
                _tmp = im[buf].copy()
                _tmp = normalize_img(_tmp, -1, 1)
                _tmp = rescale_intensity(
                    _tmp,
                    in_range=(
                        np.percentile(_tmp, sc[0]),
                        np.percentile(_tmp, sc[1]),
                    ),
                )
                _tmp = normalize_img(_tmp, -1, 1)
                images[i][buf] = _tmp

    # Renormalize
    # for buf in buffers:
    #     for i in range(len(images)):
    #         im = images[i][buf].copy()
    #         im = normalize_img(im, 0, 2 ** 16 - 1).astype("uint16")
    #         images[i][buf] = im

    # Post-processing operations
    images = np.array(images, dtype="object")
    images_cp1 = np.copy(images)
    images_cp2 = np.copy(images)
    if register:
        topos = [stmf.get_buffers([1])[1] for stmf in stmfiles]
        topos = [subtract_plane(im) for im in topos]
        topos = [(normalize_img(im, -1, 1)) for im in topos]
        topos = [
            stripe.filter_streaks(im, sigma=[10, 20], level=3, wavelet="db3")
            for im in topos
        ]
        topos = [(normalize_img(im, -1, 1)) for im in topos]
        # topos = [
        #     normalize_img(im, 0, 2 ** 16 - 1).astype("uint16") for im in topos
        # ]
        # for i in range(len(topos)):
        #     im = topos[i].copy()
        #     im = normalize_img(im, 0, 2 ** 16 - 1).astype("uint16")
        #     im = stripe.filter_streaks(
        #         im, sigma=[10, 20], level=3, wavelet="db3"
        #     )
        #     topos[i] = im
        # shifts = [register_translation(topos[0], topos[i], upsample_factor=1)[0] for i in range(len(topos))]
        for buf in buffers:
            imset = []
            for im in images_cp1:
                imset.append(im[buf])
            # new_imset = crop_image_set(imset, shifts)
            if sec:
                new_imset = imreg(
                    imset, upsample_factor=100, secondarySet=topos
                )
            else:
                new_imset = imreg(imset, upsample_factor=100)
            for i, im in enumerate(new_imset):
                images[i][buf] = im

    if movie:
        topo = [stmf.get_buffers([1])[1] for stmf in stmfiles]
        topo = [subtract_plane(im) for im in topo]
        topo = [np.uint16(normalize_img(im, 0, 2 ** 16 - 1)) for im in topo]
        topo = [
            stripe.filter_streaks(im, sigma=[10, 20], level=3, wavelet="db3")
            for im in topo
        ]
        topo = [(normalize_img(im, -1, 1)) for im in topo]

        for buf in buffers:
            imset = []
            new_imset = []
            for i, im in enumerate(images_cp2):
                imset.append(im[buf])
            # _, shifts = imreg(topo, upsample_factor=100, return_shifts=True)
            # new_imset = crop_image_set(imset, shifts)
            # new_imset = imreg(imset, upsample_factor=100, secondarySet=topo)
            if sec:
                new_imset = imreg(
                    imset, upsample_factor=100, secondarySet=topo
                )
            else:
                new_imset = imreg(imset, upsample_factor=100)
            # new_imset = imreg(imset, upsample_factor=100)

            zoom_factor = int(6 * 100 / np.min(new_imset[0].shape))
            new_imset = [
                zoom(im, zoom=zoom_factor, order=0, mode="constant")
                for im in new_imset
            ]
            if fn is not None:
                num_fn = [float(f) for f in fn]
                if normalize:
                    first_ind = np.argmin(num_fn)
                    first_img = new_imset[first_ind].copy()
                    new_imset = [
                        normalize_img(im, 1, 3).astype("float")
                        - normalize_img(first_img, 1, 3).astype("float")
                        for im in new_imset
                    ]
                new_imset = [normalize_img(im, -1, 1) for im in new_imset]

                # org: Bottom-left corner of the text string in the image.
                org = (
                    int(0.08 * new_imset[0].shape[0]),
                    int(0.12 * new_imset[0].shape[1]),
                )

                # # coordinates = (int(0.4 * new_imset[0].shape[0]), int(0.12 * new_imset[0].shape[1]))
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = np.min(new_imset[0].shape) / (80 / 0.3)
                coordinates = [
                    (
                        org[0]
                        # - int((len(f) + 13) * 6 * zoom_factor / fontScale)
                        ,
                        org[1],
                    )
                    for f in fn
                ]  # 3 for " ps"
                if textcolor == "black":
                    # fontColor = (
                    #     int(new_imset[0].min()),
                    #     int(new_imset[0].min()),
                    #     int(new_imset[0].min()),
                    # )
                    fontColor = (
                        new_imset[0].min(),
                        new_imset[0].min(),
                        new_imset[0].min(),
                    )
                elif textcolor == "white":
                    # fontColor = (
                    #     int(new_imset[0].max()),
                    #     int(new_imset[0].max()),
                    #     int(new_imset[0].max()),
                    # )
                    fontColor = (
                        new_imset[0].max(),
                        new_imset[0].max(),
                        new_imset[0].max(),
                    )
                    # fontColor = (2 ** 16 - 1, 2 ** 16 - 1, 2 ** 16 - 1)
                    # fontColor = (255, 255, 255)
                lineType = 3
                thickness = 1
                new_imset = [
                    cv2.putText(
                        new_imset[i],
                        str(fn[i]) + " ps",
                        coordinates[i],
                        font,
                        fontScale,
                        fontColor,
                        lineType,
                        thickness,
                    )
                    for i in range(len(fn))
                ]
                new_imset = [
                    x
                    for _, x in sorted(
                        zip(num_fn, new_imset), key=lambda pair: pair[0]
                    )
                ]
            new_imset = [rgb2gray(im) for im in new_imset]
            new_imset = [
                np.clip(im, np.percentile(im, 1), np.percentile(im, 99))
                for im in new_imset
            ]
            new_imset = [
                normalize_img(im, 0, 2 ** 8 - 1).astype("uint8")
                for im in new_imset
            ]
            imageio.mimwrite(f"./Movie_buf{buf}.mov", new_imset, fps=2)
            # for ind, im in enumerate(new_imset):
            #     imageio.imwrite(f"./Movie_buf{buf}_frame{ind}.tiff", im)

    cm = mpl.cm.get_cmap(cmap, 2 ** 16)
    for buf in buffers:
        if fn is not None and len(fn) == len(images):
            for i in range(len(images)):
                path = join("./", fn[i] + f".buf{buf}.tiff")
                if buf == 1 or buf == 2:
                    images[i][buf] = np.uint16(
                        cm(images[i][buf]) * (2 ** 16 - 1)
                    )
                images[i][buf] = np.uint16(
                    normalize_img(images[i][buf], 0, 2 ** 16 - 1)
                )
                imageio.imwrite(path, images[i][buf], format="TIFF-FI")
        else:
            for i in range(len(images)):
                path = join("./", split(paths[i])[-1] + f".buf{buf}.tiff")
                if buf == 1 or buf == 2:
                    images[i][buf] = np.uint16(
                        cm(images[i][buf]) * (2 ** 16 - 1)
                    )
                images[i][buf] = np.uint16(
                    normalize_img(images[i][buf], 0, 2 ** 16 - 1)
                )
                imageio.imwrite(path, images[i][buf])


if __name__ == "__main__":
    # main()
    # main2()
    main3()
