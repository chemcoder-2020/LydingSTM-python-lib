from myImageToolset import imreg, natural_key, normalize_img
import sys
import imageio
from numpy import percentile
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian


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
            sys.exit(
                "Syntax: registerSTM [lydSTMfile1/ImageFile1 (reference image)] [lydSTMfile2/ImageFile2] ..."
            )
        fpaths.append(arg)

    # USER INPUTS
    smoothing = False
    user_input = input("Apply a 1-pixel sigma gaussian filter? y/n ")
    if user_input.lower() == "y":
        smoothing = True

    stretch_contrast = False
    user_input = input("Apply contrast stretch (1%, 99%)? y/n ")
    if user_input.lower() == "y":
        stretch_contrast = True


    fpaths = sorted(fpaths, key=natural_key)
    dirpath = fpaths[0].split(".")[0].split("/")[-1]
    image_set = []

    for path in fpaths:
        dat = imageio.imread(path)
        image_set.append(dat)

    image_set = imreg(image_set)
    if smoothing:
        to_write = [gaussian(im) for im in image_set]
    if stretch_contrast:
        to_write = [
            rescale_intensity(im, in_range=(percentile(im, 1), percentile(im, 99)))
            for im in to_write
        ]

    for i, im in enumerate(to_write):
        imageio.imwrite(
            f"{dirpath}.reg_{i}.tif",
            normalize_img(im, 0, 2 ** 16 - 1).astype("uint16"),
        )
