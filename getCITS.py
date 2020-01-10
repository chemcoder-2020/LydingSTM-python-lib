import parseSTM as ps
import numpy as np
import matplotlib.pyplot as plt
import pywt
import argparse
from scipy.interpolate import interp1d
from skimage.filters import gaussian
from skimage.morphology import square
from scipy.signal import convolve2d
from pathlib import Path
import imageio
from os import remove
from myImageToolset import destripe_by_wavelet_svd
from detrend2d import subtract_plane
from scipy import interpolate, signal, fftpack


spectra_ylabels = {
    "original": "Tunneling Current, A",
    "didv": "Conductance, A/V",
    "dos": "Density of states",
    "logiv": "Tunneling Current, A",
}


def mean_filter_2d(img, kernel_size=2):
    kernel = square(int(kernel_size), dtype="float")
    kernel = kernel / np.sum(kernel ** 2)
    return convolve2d(img, kernel, boundary="symm", mode="same")


def _parseargs():
    parser = argparse.ArgumentParser(
        description="Function to extract CITS from Lyding STM files\n",
        epilog="Developed by Huy Nguyen in Gruebele and Lyding groups\n"
        "University of Illinois at Urbana-Champaign\n\n",
    )
    parser.add_argument(
        "input",
        nargs="*",
        type=str,
        help="Lyding STM file(s). Contact lyding@illinois.edu for more information",
    )
    parser.add_argument(
        "--raw", "-r", action="store_true", help="Use raw CITS data"
    )
    parser.add_argument(
        "--wavelet",
        "-w",
        help="Wavelet used to smooth the CITS data. Values are discrete wavelets available under pywt package.",
        type=str,
        default="db7",
    )
    parser.add_argument(
        "--level",
        "-l",
        help="Level of wavelet filtering used in smoothing the CITS data.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--movie",
        "-m",
        help="Make a movie out of the voltage deck",
        action="store_true",
    )
    parser.add_argument(
        "--voltage",
        "-v",
        type=float,
        help="Voltage at which to extract CITS data",
        nargs="*",
        # required=True,
    )
    parser.add_argument(
        "--buftype",
        "-t",
        type=str,
        help="Buffer types to extract from CITS data. Available values: topo, didv, dos, logIV.",
        default="original",
    )
    parser.add_argument(
        "--bufsmooth",
        "-f",
        help="Smooth buffer with a gaussian (or mean) filter where sigma (or kernel size) equals the specified pixel",
        type=float,
    )
    parser.add_argument(
        "--mean-filter",
        "-b",
        help="Use a mean filter instead of a gaussian blur to smooth the buffers",
        action="store_true",
    )
    parser.add_argument(
        "--buflogscale",
        "-e",
        help="Show the buffer with logscale",
        action="store_true",
    )
    parser.add_argument(
        "--colormap",
        "-a",
        help="Colormap in which to save buffers",
        type=str,
        default="inferno",
    )
    parser.add_argument(
        "--dpi",
        "-p",
        help="DPI of the saved figure(s)",
        type=float,
        default=300,
    )
    parser.add_argument(
        "--spectype",
        "-s",
        type=str,
        help="Spectral types to extract from CITS data. Available values: didv, dos, logIV.",
        default="original",
    )
    parser.add_argument(
        "--coordinates",
        "-c",
        help="XY coordinates (left-right, top-down) of the pixel to extract spectra",
        nargs="*",
        type=float,
    )
    parser.add_argument(
        "--specfromsmoothed",
        "-k",
        action="store_true",
        help="Extract spectra after smoothing buffers with a gaussian with a sigma of bufsmooth",
    )
    parser.add_argument(
        "--clim",
        "-d",
        help="Low and high percentages of color limits for the whole CITS STACK.",
        nargs=2,
        type=float,
        default=[0.3, 95],
    )
    args = parser.parse_args()
    return args


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


# def wavelet_filter_3d(array3d, wavelet="db20", level=2):
#     coeffs = pywt.wavedecn(array3d, wavelet=wavelet, level=level, axes=0)
#     coeffs[1:] = [
#         {"d": pywt.threshold(coeff["d"], np.std(coeff["d"]) * 3)}
#         for coeff in coeffs[1:]
#     ]
#     return pywt.waverecn(coeffs, wavelet=wavelet, axes=0)


def wavelet_filter_3d(array3d, wavelet="db3", level=3):
    return np.apply_along_axis(
        wavelet_filter_1d, 0, array3d, wavelet, level, "soft"
    )


def get_cits_blocks(stmfile, raw=False, wavelet="sym7", level=None):
    cits_blocks = []
    orig_blocks = stmfile.cits_blocks
    cits_bias = stmfile.cits_bias
    for block in orig_blocks.values():
        cits_blocks.append([cits_bias, block.data])
    if not raw:
        for i in range(len(cits_blocks)):
            cits_blocks[i][1] = wavelet_filter_3d(
                cits_blocks[i][1], wavelet=wavelet, level=level
            )
    return cits_blocks


def interpolate_cits_block(cits_block):
    interp_block = interp1d(cits_block[0], cits_block[1], kind="cubic", axis=0)
    return interp_block


def interpolate_cits(cits_blocks):
    interp_blocks = []
    for block in cits_blocks:
        interp_blocks.append(interpolate_cits_block(block))
    return interp_blocks


def find_didv_cits(cits_block, logscale=False):
    out = np.abs(np.gradient(cits_block[1], cits_block[0], axis=0))
    if logscale:
        out = np.log10(out)
    return [cits_block[0], out]


def suppress_peak(arr, suppress_indices):
    segment = np.array(arr)
    segment[suppress_indices] = np.interp(
        suppress_indices,
        [
            suppress_indices[0],
            suppress_indices[1],
            suppress_indices[-1],
            suppress_indices[-2],
        ],
        [
            segment[suppress_indices[0]],
            segment[suppress_indices[1]],
            segment[suppress_indices[-1]],
            segment[suppress_indices[-2]],
        ],
    )
    return segment


def find_dos_cits(cits_block, logscale=False):
    logI = np.log(np.abs(cits_block[1]))
    logV = np.log(np.abs(cits_block[0]))
    out = np.abs(np.gradient(logI, logV, axis=0))

    suppress_indices = []
    for i, volt in enumerate(cits_block[0]):
        if volt >= -0.15 and volt <= 0.15:
            suppress_indices.append(i)
    suppress_indices = np.array(suppress_indices)
    out = np.apply_along_axis(suppress_peak, 0, out, suppress_indices)

    if logscale:
        out = np.log10(out)
    return [cits_block[0], out]


def find_logiv_cits(cits_block):
    logI = np.log10(np.abs(cits_block[1]))
    return [cits_block[0], logI]


def smooth_cits(cits_block, size=1, mean_filter=False):
    new_cits_block = cits_block.copy()
    for i in range(new_cits_block[1].shape[0]):
        if mean_filter:
            new_cits_block[1][i] = mean_filter_2d(
                new_cits_block[1][i], kernel_size=size
            )
        else:
            new_cits_block[1][i] = gaussian(new_cits_block[1][i], sigma=size)
    return new_cits_block


def get_spectra(
    cits_block,
    coordinates,
    spectype="original",
    specfromsmoothed=False,
    bufsmooth=1,
    mean_filter=False,
):
    """spectra is a list of 2-item tuples: [(bias, 1Ddata)]"""
    spectra = []
    process_block = [(0, 0)]
    if spectype.lower() == "didv":
        process_block = find_didv_cits(cits_block)
    elif spectype.lower() == "dos":
        process_block = find_dos_cits(cits_block)
    elif spectype.lower() == "logiv":
        process_block = find_logiv_cits(cits_block)
    else:
        process_block = cits_block.copy()

    if specfromsmoothed:
        process_block = smooth_cits(
            process_block, bufsmooth, mean_filter=mean_filter
        )

    for coord in coordinates:
        spectra.append(
            (process_block[0], process_block[1][:, coord[1], coord[0]], coord)
        )
    return spectra


def get_cits_buf_from_voltage(cits_block, voltages):
    buf_list = []
    block = interpolate_cits_block(cits_block)
    for voltage in voltages:
        if (
            block.x[0] <= voltage <= block.x[-1]
            or block.x[0] >= voltage >= block.x[-1]
        ):
            buf_list.append([voltage, block(voltage)])
    return buf_list


def make_figures(
    buf_list,
    spectra=None,
    buftype="original",
    spectype="original",
    cmap="inferno",
    dpi=300,
    bufsmooth=None,
    prefix="",
    mean_filter=False,
):
    if buf_list is not None:
        for buf in buf_list:
            fig, ax = plt.subplots()
            if bufsmooth is not None:
                if mean_filter:
                    buf[1] = mean_filter_2d(buf[1], bufsmooth)
                else:
                    buf[1] = gaussian(buf[1], bufsmooth)
            ax.imshow(buf[1], cmap=cmap)
            ax.axis("off")
            fig.savefig(
                f"{prefix}.{buftype}_{buf[0]}V.tiff",
                bbox_inches="tight",
                pad_inches=0,
                pil_kwargs={"compression": "tiff_lzw"},
                dpi=dpi,
            )
            plt.close(fig)

    if spectra is not None:
        for spec in spectra:
            fig, ax = plt.subplots()
            ax.plot(spec[0], spec[1], ms=0, lw=1, label=str(spec[2]))
            ax.set_xlabel("Bias Voltage, V")
            if spectype in spectra_ylabels:
                ax.set_ylabel(spectra_ylabels[spectype.lower()])
            else:
                ax.set_ylabel(spectra_ylabels["original"])
            ax.legend()
            fig.savefig(
                f"{prefix}.{str(spec[2])}.tiff",
                # bbox_inches="tight",
                # pad_inches=0,
                pil_kwargs={"compression": "tiff_lzw"},
                dpi=dpi,
            )
            plt.close(fig)


def make_movie(
    cits_block,
    buftype="current",
    bufsmooth=None,
    cmap="inferno",
    prefix="",
    mean_filter=False,
    clim=(0.3, 95),
):
    x, y = cits_block[1][0].shape
    cits_block_cp = cits_block.copy()
    fig, ax = plt.subplots()

    if bufsmooth is not None:
        cits_block_cp = smooth_cits(
            cits_block_cp, bufsmooth, mean_filter=mean_filter
        )

    clim = [
        np.percentile(cits_block_cp[1], clim[0]),
        np.percentile(cits_block_cp[1], clim[1]),
    ]
    fig.tight_layout()

    frame_list = []
    for n in range(len(cits_block_cp[1])):
        ax.cla()
        ax.axis("off")
        im = ax.imshow(cits_block_cp[1][n], cmap=cmap)
        im.set_clim(clim)
        ax.annotate(
            f"{cits_block_cp[0][n]:.2f} V",
            xy=(1,1),
            xycoords="axes fraction",
            size=int(0.3 * x),
            horizontalalignment="right",
            verticalalignment="top",
            color="white",
            # bbox=dict(boxstyle="round4,pad=.5", fc="0.9"),
        )
        fig.savefig("./temp.tiff", dpi=100, pad_inches=0)
        frame_list.append(imageio.imread("./temp.tiff"))

    imageio.mimwrite(f"./{prefix}.cits_{buftype}.mov", frame_list, fps=5)
    remove("./temp.tiff")
    plt.close(fig)


def main():
    args = _parseargs()
    input_files = [Path(inp) for inp in args.input]
    for inp in input_files:
        stmf = ps.STMfile(inp)
        ydim, xdim = stmf.dimensions
        if args.buftype == "topo":
            topo = stmf.get_buffers([1])[1]
            topo = subtract_plane(topo)
            topo = destripe_by_wavelet_svd(topo)
            topo = np.clip(
                topo, np.percentile(topo, 0.5), np.percentile(topo, 99.5)
            )
            np.save(f"{inp}.topo.npy", topo)
            fig, ax = plt.subplots()
            ax.imshow(topo, cmap="afmhot")
            ax.axis("off")
            fig.savefig(
                f"{inp}.topo.tiff",
                pil_kwargs={"compression": "tiff_lzw"},
                dpi=args.dpi,
                bbox_inches="tight",
                pad_inches=0,
            )
            continue
        citsblocks = get_cits_blocks(
            stmf, raw=args.raw, wavelet=args.wavelet, level=args.level
        )
        for block in citsblocks:
            to_get = block.copy()
            buf_list = None
            if args.buftype is not None:
                if args.buftype.lower() == "didv":
                    to_get = find_didv_cits(block, logscale=args.buflogscale)
                elif args.buftype.lower() == "dos":
                    to_get = find_dos_cits(block, logscale=args.buflogscale)
                elif args.buftype.lower() == "logiv":
                    to_get = find_logiv_cits(block)
                else:
                    pass
            else:
                pass

            if args.voltage is not None:
                buf_list = get_cits_buf_from_voltage(to_get, args.voltage)

            if args.coordinates is not None:
                coordinates = []
                for coord in args.coordinates:
                    if coord >= 1:
                        coordinates.append(int(coord))
                    else:
                        coordinates.append(int(coord * xdim))
                coordinates = np.array(coordinates).reshape((-1, 2))
                # coordinates = (np.reshape(args.coordinates, (-1, 2)) * xdim).astype("int")
                spectra = get_spectra(
                    block,
                    coordinates,
                    args.spectype,
                    specfromsmoothed=args.specfromsmoothed,
                    bufsmooth=args.bufsmooth,
                    mean_filter=args.mean_filter,
                )

            else:
                spectra = []

            make_figures(
                buf_list,
                spectra,
                args.buftype,
                args.spectype,
                cmap=args.colormap,
                dpi=args.dpi,
                bufsmooth=args.bufsmooth,
                prefix=inp,
                mean_filter=args.mean_filter,
            )

            np.savez(f"./{inp}.cits_{args.buftype}.npz", *to_get)

            if args.movie:

                make_movie(
                    to_get,
                    args.buftype,
                    args.bufsmooth,
                    args.colormap,
                    prefix=inp,
                    mean_filter=args.mean_filter,
                    clim=args.clim,
                )


if __name__ == "__main__":
    main()
