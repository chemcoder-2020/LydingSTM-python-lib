import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy import interpolate, signal, fftpack

import parseSTM as ps
import os
import pywt
from myImageToolset import destripe_by_wavelet_svd
from detrend2d import subtract_plane

plt.ioff()

sts_mode = {
    0: ("Bias voltage", "Tunneling current, A"),
    1: ("Bias voltage", "Tunneling current, A"),
    3: ("Bias voltage", "Tunneling current, A"),
    4: ("Bias voltage", "Tunneling current, A"),
    2: ("Tip z position (nm)", "Tunneling current"),
}


def get_stmfiles(filepaths):
    stmfiles = [ps.STMfile(path) for path in filepaths]
    return stmfiles


def get_stsblocks(stmfile):
    all_blocks = stmfile.stsblocks
    return all_blocks


def wavelet_filter_1d(
    arr, wavelet_type="db20", level=0, threshold_mode="soft"
):
    arr = np.array(arr).flatten()
    coeffs = pywt.wavedec(arr, wavelet_type, level=level)
    approx = coeffs[0]
    detail = coeffs[1:]
    coeffs[1:] = [
        pywt.threshold(coeff, np.std(coeff) * 3, mode=threshold_mode)
        for coeff in coeffs[1:]
    ]
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


def get_curves_from_stsblock(
    stsblock, raw=False, shiftIV=False, level=None, wavelet_type="sym4"
):
    """Extract STS curves as numpy array of shape (n, m) where n is the length of the spectroscopy data and m is the number of curves

    Arguments:
        stsblock {parseSTM.ASpecBlock} -- Lyding spectroscopy data block

    Returns:
        numpy.ndarray -- shape (n, m)
    """
    sts_curves = np.array(stsblock.stsdata)
    spec_mode = stsblock.spec_mode
    if spec_mode in [0, 1, 3, 4]:
        X = stsblock.vrange
    elif spec_mode == 2:
        X = stsblock.srange * 0.1  # Angstrom to nm
    X = np.atleast_2d(X).reshape((-1, 1))
    curves = X.copy()
    for sts_curve in sts_curves:
        curve = np.array(sts_curve)
        if not raw:
            curve = np.atleast_2d(
                wavelet_filter_1d(
                    sts_curve, level=level, wavelet_type=wavelet_type
                )
            )
        if shiftIV:
            interp = interpolate.interp1d(
                X.flatten(),
                curve.flatten(),
                kind="cubic",
                fill_value="extrapolate",
            )
            curve = curve - interp(0)
        curves = np.hstack((curves, curve.reshape((-1, 1))))
    return curves


def wavelet_filter_curves(curves, wavelet="sym3", level=None):
    X = curves[:, 0]
    filtered_curves = np.atleast_2d(np.array(X)).T
    for curve in np.array(curves).T[1:]:
        curve = np.atleast_2d(
            wavelet_filter_1d(curve, level=level, wavelet_type=wavelet)
        )
        filtered_curves = np.hstack((filtered_curves, curve.reshape((-1, 1))))
    return filtered_curves


def median_filter_curves(curves, kernel_size=3):
    X = curves[:, 0]
    filtered_curves = np.atleast_2d(np.array(X)).T
    for curve in np.array(curves).T[1:]:
        curve = np.atleast_2d(signal.medfilt(curve, kernel_size=kernel_size))
        filtered_curves = np.hstack((filtered_curves, curve.reshape((-1, 1))))
    return filtered_curves


def svd_filter_2d(curves, reconst_vecnum=10):
    X = curves[:, 0]
    filtered_curves = np.atleast_2d(np.array(X)).T
    for curve in np.array(curves).T[1:]:
        U, D, V = np.linalg.svd(np.atleast_2d(curve).T, full_matrices=False)
        reconst = (
            U[:, :reconst_vecnum]
            @ np.diag(D[:reconst_vecnum])
            @ np.matrix(V[:reconst_vecnum, :])
        )
        filtered_curves = np.hstack((filtered_curves, reconst))
    return filtered_curves


def get_didv(curves):
    X = np.atleast_2d(
        curves[:, 0]
    ).T  # by default atleast_2d convert 1d array to (1,n) 2d array
    rest = curves[:, 1:]
    didv = np.abs(np.gradient(rest, curves[:, 0], axis=0)).reshape(
        (X.shape[0], -1)
    )
    return np.hstack((X, didv))


def suppress_peak(arr):
    assert (
        len(arr) >= 5
    ), "Please provide longer voltage range to suppress peak"
    segment = np.array(arr)
    deriv = np.gradient(segment)
    peak_ind = np.argmax(deriv)

    f = interpolate.interp1d(
        np.arange(0, peak_ind - 3, 1),
        segment[: peak_ind - 3],
        fill_value="extrapolate",
    )

    segment[peak_ind - 3 :] = np.array(
        [f(i) for i in range(peak_ind - 3, len(segment))]
    )
    return segment


def get_dos(curves):
    X = np.atleast_2d(curves[:, 0]).T
    rest = curves[:, 1:]
    dos = np.abs(np.gradient(rest, curves[:, 0], axis=0)).reshape(
        (X.shape[0], -1)
    ) / np.abs((1e-20 + rest) / (X + 1e-20))

    # logI = np.log(np.abs(1+rest))
    # logV = np.log(np.abs(curves[:,0]))
    # dos = np.abs(np.gradient(logI, logV, axis=0)).reshape((X.shape[0],-1))
    # dos = np.abs(np.gradient(np.log(np.abs(rest)), np.log(np.abs(X.flatten())), axis=0)).reshape((X.shape[0], -1))

    # suppress_indices = []
    # for i, volt in enumerate(curves[:, 0]):
    #     if volt >= -0.1 and volt <= 0.1:
    #         suppress_indices.append(i)
    # suppress_indices = np.array(suppress_indices)

    # for i, curve in enumerate(dos.T):
    #     # suppressed = suppress_peak(curve[suppress_indices])
    #     # dos[suppress_indices, i] = suppressed
    #     dos[suppress_indices, i] = np.interp(
    #         suppress_indices,
    #         [
    #             suppress_indices[0],
    #             suppress_indices[1],
    #             suppress_indices[-1],
    #             suppress_indices[-2],
    #         ],
    #         [
    #             dos[suppress_indices[0], i],
    #             dos[suppress_indices[1], i],
    #             dos[suppress_indices[-1], i],
    #             dos[suppress_indices[-2], i],
    #         ],
    #     )
    return np.hstack((X, dos))


def get_logIV(curves):
    X = np.atleast_2d(curves[:, 0]).T
    rest = curves[:, 1:]
    logI = np.log10(np.abs(rest)).reshape((X.shape[0], -1))
    return np.hstack((X, logI))


def save_separate_stsblocks(
    all_blocks,
    prefix="",
    didv=False,
    dos=False,
    logIV=False,
    raw=False,
    shiftIV=False,
    level=None,
    wavelet_type="sym4",
):
    for i, stsblock in enumerate(all_blocks):
        curves = get_curves_from_stsblock(
            all_blocks[stsblock],
            raw=False,
            shiftIV=shiftIV,
            level=level,
            wavelet_type=wavelet_type,
        )
        curves = median_filter_curves(curves, 13)

        curves = wavelet_filter_curves(
            curves, wavelet=wavelet_type, level=level
        )
        with open(f"{prefix}.stsblock{i}.txt", "w") as file:
            file.write(f"[Block {i}]\n")
            if logIV:
                np.savetxt(file, get_logIV(curves), delimiter=",", fmt="%1.4e")
            elif didv:
                np.savetxt(file, get_didv(curves), delimiter=",", fmt="%1.4e")
            elif dos:
                np.savetxt(file, get_dos(curves), delimiter=",", fmt="%1.4e")
            else:
                np.savetxt(file, curves, delimiter=",", fmt="%1.4e")
            file.write("\n")


def save_all_stsblocks(
    all_blocks,
    prefix="",
    didv=False,
    dos=False,
    logIV=False,
    raw=False,
    shiftIV=False,
    level=None,
    wavelet_type="sym4",
):
    with open(f"{prefix}.allSTSblocks.txt", "w") as file:
        for i, stsblock in enumerate(all_blocks):
            curves = get_curves_from_stsblock(
                all_blocks[stsblock],
                raw=False,
                shiftIV=shiftIV,
                level=level,
                wavelet_type=wavelet_type,
            )
            curves = median_filter_curves(curves, 13)

            curves = wavelet_filter_curves(
                curves, wavelet=wavelet_type, level=level
            )
            file.write(f"[Block {i}]\n")
            if logIV:
                np.savetxt(file, get_logIV(curves), delimiter=",", fmt="%1.4e")
            elif didv:
                np.savetxt(file, get_didv(curves), delimiter=",", fmt="%1.4e")
            elif dos:
                np.savetxt(file, get_dos(curves), delimiter=",", fmt="%1.4e")
            else:
                np.savetxt(file, curves, delimiter=",", fmt="%1.4e")
            file.write("\n")


def plot_save_figures(
    all_blocks,
    stmfile=None,
    prefix="",
    didv=False,
    dos=False,
    logIV=False,
    raw=False,
    shiftIV=False,
    level=None,
    wavelet_type="db3",
    curves_nums=None,
    block=None,
    ylim=None,
):
    if curves_nums is not None and block is None:
        block = 0

    if curves_nums is None:
        for i, stsblock in enumerate(all_blocks):
            curves = get_curves_from_stsblock(
                all_blocks[stsblock],
                raw=True,
                shiftIV=shiftIV,
                level=level,
                wavelet_type=wavelet_type,
            )
            curves = median_filter_curves(curves, 13)

            curves = wavelet_filter_curves(
                curves, wavelet=wavelet_type, level=level
            )

            xlabel, ylabel = sts_mode[all_blocks[stsblock].spec_mode]
            # if logIV:
            #     curves = get_logIV(curves)
            if didv:
                curves = get_didv(curves)
                ylabel = "Conductance, A/V"
            elif dos:
                curves = get_dos(curves)
                ylabel = "Density of states"
            else:
                pass
            # curves = wavelet_filter_curves(
            #     curves, wavelet=wavelet_type, level=level
            # )
            if stmfile is not None:
                get_and_annotate_topo(
                    stmfile, stsblock, list(range(1, curves.shape[1]))
                )
            # fig, ax = plt.subplots(figsize=(16, 3.8))
            fig, ax = plt.subplots(figsize=(6, 4))
            if ylim is not None:
                ax.set_ylim(ylim)
            for j in range(1, curves.shape[1]):

                if logIV:
                    ax.plot(curves[:, 0], np.abs(curves[:, j]), ms=0, lw=0.7)
                else:
                    ax.plot(curves[:, 0], curves[:, j], ms=0, lw=0.7)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                if logIV:
                    ax.set_yscale("log")
            fig.savefig(
                f"{prefix}.stsblock{i}.tiff",
                dpi=600,
                pil_kwargs={"compression": "tiff_lzw"},
            )
            plt.close(fig)
    else:
        curves = get_curves_from_stsblock(
            all_blocks[block], True, shiftIV, level, wavelet_type
        )

        curves = median_filter_curves(curves, 13)

        curves = wavelet_filter_curves(
            curves, wavelet=wavelet_type, level=level
        )

        xlabel, ylabel = sts_mode[all_blocks[block].spec_mode]
        if didv:
            curves = get_didv(curves)
            ylabel = "Conductance, A/V"
        elif dos:
            curves = get_dos(curves)
            ylabel = "Density of states"
        else:
            pass
        # curves = wavelet_filter_curves(
        #     curves, wavelet=wavelet_type, level=level
        # )
        # curves = svd_filter_2d(curves,1)
        if stmfile is not None:
            get_and_annotate_topo(stmfile, block, list(curves_nums))
        # fig, ax = plt.subplots(figsize=(16, 3.8))
        fig, ax = plt.subplots(figsize=(6, 4))
        if ylim is not None:
            ax.set_ylim(ylim)
        for j in curves_nums:

            if logIV:
                ax.plot(
                    curves[:, 0],
                    np.abs(curves[:, j]),
                    ms=0,
                    lw=0.5,
                    label=str(j),
                )
            else:
                ax.plot(curves[:, 0], curves[:, j], ms=0, lw=0.5, label=str(j))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
            if logIV:
                ax.set_yscale("log")
        fig.savefig(
            f"{prefix}.stsblock{block}.curves{str(curves_nums)}.tiff",
            # dpi=600,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        plt.close(fig)


def get_and_annotate_topo(stmfile, block, curves):
    topo = stmfile.get_buffers([1])[1]
    topo = destripe_by_wavelet_svd(topo)
    topo = subtract_plane(topo)
    sts_blocks = get_stsblocks(stmfile)
    coords = sts_blocks[block].coords
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.axis("off")
    ax.imshow(topo, "afmhot")
    for curve in list(curves):
        ax.annotate(
            str(curve),
            coords[curve - 1],
            fontsize=7,
            horizontalalignment="right",
            verticalalignment="top",
            color="purple",
        )
    fig.savefig(
        f"./topo_block_{block}_curves_{str(curves)}.tiff",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
        pad_inches=0,
    )


def plot_save_separate_figures(
    all_blocks,
    stmfile=None,
    prefix="",
    didv=False,
    dos=False,
    logIV=False,
    raw=False,
    shiftIV=False,
    level=None,
    wavelet_type="sym4",
    curves_nums=None,
    block=None,
    ylim=None,
):
    if curves_nums is not None and block is None:
        block = 0

    if curves_nums is not None and block is not None:
        # curves = get_curves_from_stsblock(
        #     all_blocks[block], raw, shiftIV, level, wavelet_type
        # )

        curves = get_curves_from_stsblock(
            all_blocks[block], True, shiftIV, level, wavelet_type
        )
        if not raw:
            curves = median_filter_curves(curves, 13)

            curves = wavelet_filter_curves(
                curves, wavelet=wavelet_type, level=level
            )

        xlabel, ylabel = sts_mode[all_blocks[block].spec_mode]
        if didv:
            curves = get_didv(curves)
            ylabel = "Conductance, A/V"
        elif dos:
            curves = get_dos(curves)
            ylabel = "Density of states"
        else:
            pass
        # curves = wavelet_filter_curves(
        #     curves, wavelet=wavelet_type, level=level
        # )
        # curves = svd_filter_2d(curves,1)

        np.save(f"{prefix}.stsblock{block}.npy", curves)
        if stmfile is not None:
            get_and_annotate_topo(stmfile, block, list(curves_nums))
        for j in curves_nums:
            fig, ax = plt.subplots(figsize=(6, 4))
            # fig, ax = plt.subplots(figsize=(16, 3.8))
            if logIV:
                ax.plot(curves[:, 0], np.abs(curves[:, j]), ms=0, lw=0.5)
            else:
                ax.plot(curves[:, 0], curves[:, j], ms=0, lw=0.5)
            ax.set_xlabel(xlabel, fontsize=17)
            ax.set_ylabel(ylabel, fontsize=17)
            ax.tick_params(labelsize=15)
            ax.tick_params(which="minor", axis="y", left=False)
            if ylim is not None:
                ax.set_ylim(ylim)
            if logIV:
                ax.set_yscale("log")
            fig.savefig(
                f"{prefix}.stsblock{block}.curve{j}.tiff",
                # dpi=600,
                pil_kwargs={"compression": "tiff_lzw"},
            )
            plt.close(fig)
    else:
        for i, stsblock in enumerate(all_blocks):

            curves = get_curves_from_stsblock(
                all_blocks[stsblock],
                raw=True,
                shiftIV=shiftIV,
                level=level,
                wavelet_type=wavelet_type,
            )
            if not raw:
                curves = median_filter_curves(curves, 13)

                curves = wavelet_filter_curves(
                    curves, wavelet=wavelet_type, level=level
                )
            # if logIV:
            #     curves = get_logIV(curves)
            xlabel, ylabel = sts_mode[all_blocks[stsblock].spec_mode]
            if didv:
                curves = get_didv(curves)
                ylabel = "Conductance, A/V"
            elif dos:
                curves = get_dos(curves)
                ylabel = "Density of states"
            else:
                pass
            if stmfile is not None:
                get_and_annotate_topo(
                    stmfile, stsblock, list(range(1, curves.shape[1]))
                )
            for j in range(1, curves.shape[1]):

                fig, ax = plt.subplots(figsize=(16, 3.8))
                if ylim is not None:
                    ax.set_ylim(ylim)
                if logIV:
                    ax.plot(curves[:, 0], np.abs(curves[:, j]), ms=0, lw=0.7)
                else:
                    ax.plot(curves[:, 0], curves[:, j], ms=0, lw=0.7)

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                if logIV:
                    ax.set_yscale("log")
                fig.savefig(
                    f"{prefix}.stsblock{i}.curve{j}.tiff",
                    dpi=600,
                    pil_kwargs={"compression": "tiff_lzw"},
                )
                plt.close(fig)


def _parseargs():
    parser = argparse.ArgumentParser(
        description="Extract and plot STS data from Lyding STM file\n\n",
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
        "--many-txt",
        "-t",
        help="Save many txt files, each for one STSblock",
        action="store_true",  # default to false
    )
    parser.add_argument(
        "--plot-separate-curves",
        "-s",
        help="Save a figure for each STS curve",
        action="store_true",  # default to true
    )
    parser.add_argument(
        # By default, data is filtered with wavelet filter
        "--raw",
        "-r",
        help="Save raw txt and/or raw plot",
        action="store_true",
    )
    parser.add_argument(
        "--wavelet",
        "-w",
        help="Discrete wavelet to use for filtering data",
        type=str,
        default="sym4",
    )
    parser.add_argument(
        "--level",
        "-e",
        type=int,
        help="Level of wavelet filtering",
        default=None,
    )
    parser.add_argument("--didv", "-x", help="Get didv", action="store_true")
    parser.add_argument(
        "--dos", "-y", help="Get d(lnI)/d(lnV)", action="store_true"
    )
    parser.add_argument(
        "--logIV",
        "-l",
        help="Use log scale for tunneling current",
        action="store_true",
    )
    parser.add_argument(
        "--shiftIV",
        "-z",
        help="shift IV curve so I = 0 at V = 0",
        action="store_true",
    )
    parser.add_argument(
        "--curves",
        "-c",
        help="Specify the curve numbers, if known, to extract (will not extract data, only figure)",
        type=int,
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--block",
        "-b",
        help="Specify the block from which to extract curves specified.",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--ylim",
        "-m",
        help="Specify the y limits while extracting a specific curve.",
        type=float,
        default=None,
        nargs=2,
    )
    args = parser.parse_args()
    return args


def main():
    args = _parseargs()
    stmfiles = get_stmfiles(args.input)
    if args.logIV:
        args.didv = False
        args.dos = False
    for i, stmfile in enumerate(stmfiles):
        all_blocks = get_stsblocks(stmfile)
        if args.many_txt:
            save_separate_stsblocks(
                all_blocks=all_blocks,
                prefix=os.path.split(args.input[i])[1],
                didv=args.didv,
                dos=args.dos,
                logIV=args.logIV,
                raw=args.raw,
                shiftIV=args.shiftIV,
                level=args.level,
                wavelet_type=args.wavelet,
            )
        else:
            save_all_stsblocks(
                all_blocks=all_blocks,
                prefix=os.path.split(args.input[i])[1],
                didv=args.didv,
                dos=args.dos,
                logIV=args.logIV,
                raw=args.raw,
                shiftIV=args.shiftIV,
                level=args.level,
                wavelet_type=args.wavelet,
            )
        if args.plot_separate_curves:
            plot_save_separate_figures(
                all_blocks=all_blocks,
                stmfile=stmfile,
                prefix=os.path.split(args.input[i])[1],
                didv=args.didv,
                dos=args.dos,
                logIV=args.logIV,
                raw=args.raw,
                shiftIV=args.shiftIV,
                level=args.level,
                wavelet_type=args.wavelet,
                curves_nums=args.curves,
                block=args.block,
                ylim=args.ylim,
            )
        else:
            plot_save_figures(
                all_blocks=all_blocks,
                stmfile=stmfile,
                prefix=os.path.split(args.input[i])[1],
                didv=args.didv,
                dos=args.dos,
                logIV=args.logIV,
                raw=args.raw,
                shiftIV=args.shiftIV,
                level=args.level,
                wavelet_type=args.wavelet,
                ylim=args.ylim,
                block=args.block,
                curves_nums=args.curves,
            )


if __name__ == "__main__":
    main()
