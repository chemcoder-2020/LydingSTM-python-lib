import parseSTM as ps
import sys
import imageio
from PIL import Image
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from numpy import percentile, array
from myImageToolset import normalize_img, imreg
import numpy as np


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


def isImage(path):
    if (
        path.endswith("png")
        or path.endswith("jpeg")
        or path.endswith("jpg")
        or path.endswith("tiff")
        or path.endswith("tif")
        or path.endswith("PNG")
        or path.endswith("JPEG")
        or path.endswith("JPG")
        or path.endswith("TIFF")
        or path.endswith("TIF")
    ):
        return True
    else:
        return False


if __name__ == "__main__":
    fpaths = []
    for ind, arg in enumerate(sys.argv[1:]):
        if "-h" == arg or "--help" == arg:
            sys.exit("Syntax: makeSTMmovie [lydSTMfile1/ImageFile1] [lydSTMfile2/ImageFile2] ...")
        fpaths.append(arg)
    with open("log.txt", "w") as file:
        np.savetxt(file, np.array(fpaths), delimiter=",", fmt="%s")

    # USER INPUTS
    smoothing = False
    user_input = input("Apply a 1-pixel sigma gaussian filter? y/n ")
    if user_input.lower() == "y":
        smoothing = True

    fps = 1
    user_input = input("FPS? ")
    try:
        fps = float(user_input)
    except ValueError:
        print("FPS input is not a number, default to 1 FPS")

    stretch_contrast = False
    user_input = input("Apply contrast stretch (1%, 99%)? y/n ")
    if user_input.lower() == "y":
        stretch_contrast = True

    save_image = False
    user_input = input("Save image sequence? y/n ")
    if user_input.lower() == "y":
        save_image = True

    # fpaths = sorted(fpaths, key=natural_key)
    dirpath = fpaths[0].split(".")[0].split("/")[-1]
    var = sum([isImage(path) for path in fpaths])
    isSTM = False
    isImg = True
    if var == 0:
        isSTM = True
        isImg = False
    elif var == len(fpaths):
        isSTM = False
        isImg = True
    else:
        sys.exit(status="File formats are not consistent. Use either a set of images or STM files")

    image_set = []
    if isSTM:
        for path in fpaths:
            dat = ps.STMfile(path)
            image_set.append(dat.get_all_orig_buffers())

        for buftype in image_set[0]:
            to_write = [im[buftype] for im in image_set]
            if smoothing:
                to_write = [gaussian(im) for im in to_write]

            if stretch_contrast:
                to_write = [rescale_intensity(im, in_range=(percentile(im, 1), percentile(im, 99))) for im in to_write]

            if buftype == 1:
                imageio.mimwrite(f"{dirpath}.Topo_Trace.mov", to_write, fps=fps)
                if save_image:
                    for i, im in enumerate(to_write):
                        num = fpaths[i].split(".")[-1]
                        imageio.imwrite(
                            f"{dirpath}.Topo_Trace_{num}.tif",
                            normalize_img(im * 10 ** 12, 0, 2 ** 16 - 1).astype("uint16"),
                        )
            elif buftype == 2:
                imageio.mimwrite(f"{dirpath}.Topo_Retrace.mov", to_write, fps=fps)
                if save_image:
                    for i, im in enumerate(to_write):
                        num = fpaths[i].split(".")[-1]
                        imageio.imwrite(
                            f"{dirpath}.Topo_Retrace_{num}.tif",
                            normalize_img(im * 10 ** 12, 0, 2 ** 16 - 1).astype("uint16"),
                        )
            elif buftype == 3:
                imageio.mimwrite(f"{dirpath}.Current.mov", to_write, fps=fps)
                if save_image:
                    for i, im in enumerate(to_write):
                        num = fpaths[i].split(".")[-1]
                        imageio.imwrite(
                            f"{dirpath}.Current_{num}.tif",
                            normalize_img(im * 10 ** 12, 0, 2 ** 16 - 1).astype("uint16"),
                        )
            elif buftype == 4:
                imageio.mimwrite(f"{dirpath}.LockInX.mov", array(to_write) * -1, fps=fps)  # for colormap consistency
                if save_image:
                    for i, im in enumerate(to_write):
                        num = fpaths[i].split(".")[-1]
                        imageio.imwrite(
                            f"{dirpath}.LockInX_{num}.tif",
                            normalize_img(array(im) * -1, 0, 2 ** 16 - 1).astype("uint16"),
                        )
            elif buftype == 5:
                imageio.mimwrite(f"{dirpath}.LockInY.mov", array(to_write) * -1, fps=fps)
                if save_image:
                    for i, im in enumerate(to_write):
                        num = fpaths[i].split(".")[-1]
                        imageio.imwrite(
                            f"{dirpath}.LockInY_{num}.tif",
                            normalize_img(array(im) * -1, 0, 2 ** 16 - 1).astype("uint16"),
                        )
            else:
                print(f"I'm not sure what this buffer type {buftype} is. Skipping..")
                continue
    elif isImg:
        for path in fpaths:
            # dat = imageio.imread(path)
            dat = np.array(Image.open(path))
            image_set.append(dat)
        to_write = image_set.copy()
        # to_write = imreg(image_set, upsample_factor=1)
        if smoothing:
            to_write = [gaussian(im) for im in to_write]
        if stretch_contrast:
            to_write = [rescale_intensity(im, in_range=(percentile(im, 1), percentile(im, 99))) for im in to_write]
        imageio.mimwrite(f"{dirpath}.movie.mov", to_write, fps=fps)
        if save_image:
            for i, im in enumerate(to_write):
                imageio.imwrite(f"{dirpath}.movie_{i}.tif", normalize_img(im, 0, 2 ** 16 - 1).astype("uint16"))
    else:
        sys.exit(status="File formats are not consistent. Use either a set of images or STM files")
