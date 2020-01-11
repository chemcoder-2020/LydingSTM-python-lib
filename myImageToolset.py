import numpy as np
from skimage import img_as_float
from scipy.signal import medfilt2d
import os
import imageio
import errno

from skimage.feature import register_translation
from scipy.ndimage.interpolation import shift

from skimage.exposure import equalize_adapthist, rescale_intensity
from scipy.ndimage import zoom
from scipy import fftpack
import copy
import pywt
from sklearn.decomposition import FastICA, KernelPCA, PCA


def destripe_by_wavelet_svd(img, wavelet="sym7", level=None, vecnum=None):
    """Remove stripes on images by wavelet decomposition and reconstruction.
    Effectiveness highly depends on the mother wavelet and mildly on vecnum.
    Sym7 is a good default mother wavelet; level of None is good; vecnum goes
    up to the smallest dimension of the image.

    Parameters
    ----------
    img : any real type
        Two-dimensional numpy.ndarray
    wavelet : str
        Discrete wavelet supported by PyWavelet. Recommendation: V-fib
        shaped wavelets.
    level : int
        The max level up to which to decompose the image.
    vecnum : int
        The amount of destripe. Value goes up to the smallest dimension of the image.

    Returns
    -------
    Two-dimensional numpy.ndarray
        Destriped image

    """
    src = np.array(img)
    if vecnum is None:
        # vecnum = int(0.015 * np.min(src.shape))
        vecnum = 7
    coeffs = pywt.wavedec2(src, wavelet=wavelet, level=level)
    approx = coeffs[0]
    detail = coeffs[1:]
    coeffs_filt = [approx]
    for nthlevel in detail:
        # Wavelet coefficients are in fourier space. ch = horizontal coef.
        ch, cv, cd = nthlevel
        fch = fftpack.rfft(ch)
        U, D, V = np.linalg.svd(fch, full_matrices=False)
        # fch_filt = np.matrix(U[:,vecnum:]) * np.diag(D[vecnum:]) * np.matrix(V[vecnum:, :])
        fch_filt = U[:, vecnum:] @ np.diag(D[vecnum:]) @ V[vecnum:, :]
        fch_filt = np.array(fch_filt)

        # ICA method
        # transformer = FastICA()
        # fch_filt_transformed = transformer.fit_transform(fch)
        # fch_filt_transformed[:,1] = 0
        # fch_filt = transformer.inverse_transform(fch_filt_transformed)

        # kernel PCA method
        # transformer = KernelPCA(kernel="rbf", fit_inverse_transform=True)
        # fch_filt_transformed = transformer.fit_transform(fch)
        # fch_filt_transformed[:,:10] = 0
        # fch_filt = transformer.inverse_transform(fch_filt_transformed)

        ch_filt = fftpack.irfft(fch_filt)
        coeffs_filt.append((ch_filt, cv, cd))
    img_filt = pywt.waverec2(coeffs_filt, wavelet=wavelet)
    return img_filt


def median_level(img, kernel_size=31):
    """Short summary.

    Parameters
    ----------
    img : type
        Two-dimensional numpy.ndarray
    kernel_size : type
        Odd integer, controlling the size of the median filter.

    Returns
    -------
    Two-dimensional numpy.ndarray
        Leveled image.

    """
    img = img_as_float(img)
    pad_width = 25
    padded_img = np.pad(img, pad_width, mode="reflect")
    img_bg = medfilt2d(padded_img, kernel_size)
    return (padded_img - img_bg)[pad_width:-pad_width, pad_width:-pad_width]


def clahe(img, kernel_size=21, clip_limit=0.01):
    """Short summary.

    Parameters
    ----------
    img : type
        Description of parameter `img`.
    kernel_size : type
        Description of parameter `kernel_size`.
    clip_limit : type
        Description of parameter `clip_limit`.

    Returns
    -------
    type
        Description of returned object.

    """
    img = img_as_float(img)
    img = normalize_img(img, 0, 1)
    pad_width = 25
    padded_img = np.pad(img, pad_width, mode="reflect")
    return equalize_adapthist(padded_img, kernel_size, clip_limit)[
        pad_width:-pad_width, pad_width:-pad_width
    ]


def normalize_img(img, to_vmin, to_vmax, from_vmin=None, from_vmax=None):
    """Short summary.

    Parameters
    ----------
    img : type
        Description of parameter `img`.
    to_vmin : type
        Description of parameter `vmin`.
    to_vmax : type
        Description of parameter `vmax`.
    from_vmin : type
    from_vmax : type

    Returns
    -------
    type
        Description of returned object.

    """

    img_cp = copy.deepcopy(img).astype("float")
    if from_vmin is not None and from_vmax is not None:
        scale = (from_vmax - from_vmin) / np.abs((to_vmax - to_vmin))
        out = (img_cp - from_vmin) / scale + to_vmin
    else:
        scale = (img_cp.max() - img_cp.min()) / np.abs(to_vmax - to_vmin)
        out = (img_cp - img_cp.min()) / scale + to_vmin
    return out


def natural_key(string_):
    """Short summary.

    Parameters
    ----------
    string_ : type
        Description of parameter `string_`.

    Returns
    -------
    type
        Description of returned object.

    """
    """
    See http://www.codinghorror.com/blog/archives/001018.html
    This function is to define a key for sorting the filenames with numbers in it

    """
    import re

    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def fft_stripe_remove(
    img, stripe_direction="horizontal", mask_dimension=(60, 20)
):
    """Short summary.

    Parameters
    ----------
    img : type
        Description of parameter `img`.
    stripe_direction : type
        Description of parameter `stripe_direction`.
    mask_dimension : type
        Description of parameter `mask_dimension`.

    Returns
    -------
    fft_stripe_remove(img, stripe_direction="horizontal",
        Description of returned object.

    """
    fimg = np.fft.fft2(img)
    r, c = fimg.shape
    midr, midc = (int(r / 2), int(c / 2))
    if stripe_direction == "horizontal":
        startr = midr - int(mask_dimension[0] / 2)
        if startr < 0:
            raise ("Mask row dimension to large")
        fimg[startr : startr + mask_dimension[0], : mask_dimension[1]] = 0
        fimg[startr : startr + mask_dimension[0], c - mask_dimension[1] :] = 0
    elif stripe_direction == "vertical":
        startc = midc - int(mask_dimension[1] / 2)
        if startc < 0:
            raise ("Mask column dimension to large")
        fimg[: mask_dimension[0], startc : startc + mask_dimension[1]] = 0
        fimg[r - mask_dimension[0] :, startc : startc + mask_dimension[1]] = 0
    else:
        raise (
            "{a} is not a valid parameter for stripe_direction".format(
                a=stripe_direction
            )
        )
    return np.abs(np.fft.ifft2(fimg))


def batch_stripe_remove(
    image_set_path,
    stripe_direction="horizontal",
    mask_dimension=(60, 20),
    median_level_kernel_size=31,
    clahe_clip_limit=0.01,
    clahe_kernel_size=21,
    stretch_contrast=False,
):
    """Short summary.

    Parameters
    ----------
    image_set_path : type
        Description of parameter `image_set_path`.
    stripe_direction : type
        Description of parameter `stripe_direction`.
    mask_dimension : type
        Description of parameter `mask_dimension`.
    median_level_kernel_size : type
        Description of parameter `median_level_kernel_size`.
    clahe_clip_limit : type
        Description of parameter `clahe_clip_limit`.
    clahe_kernel_size : type
        Description of parameter `clahe_kernel_size`.
    stretch_contrast : type
        Description of parameter `stretch_contrast`.

    Returns
    -------
    type
        Description of returned object.

    """
    paths = [
        os.path.join(image_set_path, f)
        for f in os.listdir(image_set_path)
        if f.endswith(".TIF") or f.endswith(".tif")
    ]
    paths = sorted(paths, key=natural_key)
    img_set = []
    for path in paths:
        img = imageio.imread(path)
        fimg = fft_stripe_remove(img, stripe_direction, mask_dimension)
        fimg = median_level(fimg, kernel_size=median_level_kernel_size)
        if clahe_clip_limit is not None and clahe_kernel_size is not None:
            fimg = clahe(
                fimg,
                kernel_size=clahe_kernel_size,
                clip_limit=clahe_clip_limit,
            )
        if stretch_contrast:
            fimg = rescale_intensity(
                fimg,
                in_range=(np.percentile(fimg, 1), np.percentile(fimg, 99)),
            )
        fimg = img_as_float(normalize_img(fimg, 0, 1))
        img_set.append(fimg)
    return img_set


def batch_stripe_remove2(
    image_set,
    stripe_direction="horizontal",
    mask_dimension=(60, 20),
    median_level_kernel_size=31,
    clahe_clip_limit=0.01,
    clahe_kernel_size=21,
    stretch_contrast=False,
):
    """Short summary.

    Parameters
    ----------
    image_set : type
        Description of parameter `image_set`.
    stripe_direction : type
        Description of parameter `stripe_direction`.
    mask_dimension : type
        Description of parameter `mask_dimension`.
    median_level_kernel_size : type
        Description of parameter `median_level_kernel_size`.
    clahe_clip_limit : type
        Description of parameter `clahe_clip_limit`.
    clahe_kernel_size : type
        Description of parameter `clahe_kernel_size`.
    stretch_contrast : type
        Description of parameter `stretch_contrast`.

    Returns
    -------
    type
        Description of returned object.

    """
    img_set = []
    for img in image_set:
        fimg = fft_stripe_remove(img, stripe_direction, mask_dimension)
        fimg = median_level(fimg, kernel_size=median_level_kernel_size)
        if clahe_clip_limit is not None and clahe_kernel_size is not None:
            fimg = clahe(
                fimg,
                kernel_size=clahe_kernel_size,
                clip_limit=clahe_clip_limit,
            )
        if stretch_contrast:
            fimg = rescale_intensity(
                fimg,
                in_range=(np.percentile(fimg, 1), np.percentile(fimg, 99)),
            )
        fimg = img_as_float(normalize_img(fimg, 0, 1))
        img_set.append(fimg)
    return img_set


def pad_imset(imset):
    shapes = np.array([[im.shape[0], im.shape[1]] for im in imset])
    max_i = shapes[:, 0].max()
    max_j = shapes[:, 1].max()
    new_imset = []
    for im in imset:
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
    return new_imset


def rescale_image(im, scale):
    """Scaling factor must be greater than or equal to 1."""
    assert scale >= 1, "Scaling factor must be greater than or equal to 1."
    old_shape = im.shape
    im = zoom(im, scale, order=0)
    new_shape = im.shape
    i_0 = int((new_shape[0] - old_shape[0]) / 2)
    i_1 = int((new_shape[0] - old_shape[0]) / 2) + old_shape[0] + 1
    j_0 = int((new_shape[1] - old_shape[1]) / 2)
    j_1 = int((new_shape[1] - old_shape[1]) / 2) + old_shape[1] + 1
    im = im[i_0:i_1, j_0:j_1]
    return im


def imreg(
    txtdataset,
    secondarySet=None,
    full=False,
    upsample_factor=1,
    return_shifts=False,
):
    """Short summary.

    Parameters
    ----------
    txtdataset : type
        Description of parameter `txtdataset`.
    full : type
        Description of parameter `full`.

    Returns
    -------
    type
        Description of returned object.

    """
    #  perform subpixel registration translation on a data set

    im0 = txtdataset[0]
    if secondarySet is None:
        ref = im0.copy()
        refSet = txtdataset.copy()
    else:
        ref = secondarySet[0]
        refSet = secondarySet
    shifts = [
        register_translation(ref, refSet[i], upsample_factor=upsample_factor)[
            0
        ]
        for i in range(len(refSet))
    ]

    # shifted images will still have the same z-value
    shifted_images = [
        shift(txtdataset[i], shifts[i]) for i in range(len(txtdataset))
    ]
    reconstructed_dataset = shifted_images.copy()
    #  cropping the zero pixels out of the translated images
    true_points = [
        np.argwhere(im) for im in reconstructed_dataset
    ]  # find the non-zero points on image
    topleft = [pts.min(axis=0) for pts in true_points]
    bottomright = [pts.max(axis=0) for pts in true_points]
    topleftx = [t[0] for t in topleft]
    toplefty = [t[1] for t in topleft]
    bottomrightx = [t[0] for t in bottomright]
    bottomrighty = [t[1] for t in bottomright]
    startx = max(topleftx)
    starty = max(toplefty)
    stopx = min(bottomrightx)
    stopy = min(bottomrighty)
    newset = [im[startx:stopx, starty:stopy] for im in reconstructed_dataset]
    if full:
        return newset, (startx, stopx, starty, stopy)
    if return_shifts:
        return newset, shifts
    return newset


def crop_image_set(imset, shifts):
    shifted_images = [shift(imset[i], shifts[i]) for i in range(len(imset))]
    true_points = [
        np.argwhere(im) for im in shifted_images
    ]  # find the non-zero points on image
    topleft = [pts.min(axis=0) for pts in true_points]
    bottomright = [pts.max(axis=0) for pts in true_points]
    topleftx = [t[0] for t in topleft]
    toplefty = [t[1] for t in topleft]
    bottomrightx = [t[0] for t in bottomright]
    bottomrighty = [t[1] for t in bottomright]
    startx = max(topleftx)
    starty = max(toplefty)
    stopx = min(bottomrightx)
    stopy = min(bottomrighty)
    newset = [im[startx:stopx, starty:stopy] for im in shifted_images]
    return newset


def create_directory(homepath, *args):
    """Short summary.

    Parameters
    ----------
    homepath : type
        Description of parameter `homepath`.
    *args : type
        Description of parameter `*args`.

    Returns
    -------
    type
        Description of returned object.

    """
    for arg in args:
        try:
            os.makedirs(os.path.join(homepath, str(arg)))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def save_img(img, path="./"):
    """Short summary.

    Parameters
    ----------
    img : type
        Description of parameter `img`.
    path : type
        Description of parameter `path`.

    Returns
    -------
    type
        Description of returned object.

    """
    imageio.imwrite(path, normalize_img(img, 0, 2 ** 16 - 1).astype("uint16"))


def save_image_batch(image_batch, fns, path="./"):
    """Short summary.

    Parameters
    ----------
    image_batch : type
        Description of parameter `image_batch`.
    fns : type
        Description of parameter `fns`.
    path : type
        Description of parameter `path`.

    Returns
    -------
    type
        Description of returned object.

    """
    for e, fn in enumerate(fns):
        save_img(image_batch[e], os.path.join(path, fn))


def save_movie(image_batch, movie_name, fps):
    """Short summary.

    Parameters
    ----------
    image_batch : type
        Description of parameter `image_batch`.
    movie_name : type
        Description of parameter `movie_name`.
    fps : type
        Description of parameter `fps`.

    Returns
    -------
    type
        Description of returned object.

    """
    imageio.mimwrite(movie_name, image_batch, fps)
