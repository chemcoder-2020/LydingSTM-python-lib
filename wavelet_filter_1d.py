import pywt
import argparse
import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile
from pathlib import Path


def wavelet_filter_1d(
    input_data,
    wavelet_type="bior6.8",
    level=0,
    threshold_mode="soft",
    data_delimiter=",",
    data_skiprows=0,
    read_columns=[],
    save_plot=False,
    abscissa_column=None,
):
    try:
        if isfile(input_data):
            data_full = np.loadtxt(
                input_data, delimiter=data_delimiter, skiprows=data_skiprows
            )
            if read_columns != []:
                data = data_full[:, read_columns].copy()
            else:
                data = data_full.copy()
    except (UnicodeDecodeError, ValueError) as e:
        data_full = input_data.copy()
        if read_columns != []:
            data = data_full[:, read_columns]
        else:
            data = data_full.copy()

    if wavelet_type not in pywt.wavelist(kind="discrete"):
        raise (
            "Wavelet type is not supported. Run print(pywt.wavelist(kind='discrete')) to see the options available"
        )

    # Wavelet decomposition
    fdata = []
    data = np.atleast_2d(data)
    for i in range(data.shape[1]):
        coeffs = pywt.wavedec(data[:, i], wavelet_type, level=level)
        coeffs[1:] = [
            pywt.threshold(coeff, np.std(coeff) * 3, mode=threshold_mode)
            for coeff in coeffs[1:]
        ]
        fdata.append(
            pywt.waverec(coeffs, wavelet_type)
        )  # reconstructed data have 1 more points

    fdata = np.array(fdata)
    if level > 0:
        fdata = fdata[:, 1:]
    fdata = fdata.transpose()

    if read_columns != []:
        data_full[:, read_columns] = fdata
    else:
        data_full = fdata

    try:
        if isfile(input_data):
            with open(
                "{input_data}_wavelet_filtered.txt".format(input_data=input_data), "w"
            ) as f1:
                with open(input_data, "r") as f2:
                    for i in range(data_skiprows):
                        line = f2.readline()
                        f1.write(line)
                np.savetxt(f1, data_full, fmt="%10.5f", delimiter=",", newline="\n")
    except (UnicodeDecodeError, ValueError) as e:
        pass

    if save_plot:
        fig, ax = plt.subplots()
        for i in range(fdata.shape[1]):
            if abscissa_column is not None:
                ax.plot(
                    data_full[:, abscissa_column],
                    fdata[:, i],
                    ms=2,
                    lw=1,
                    alpha=0.4,
                    label="Column {a}".format(a=read_columns[i]),
                )
            else:
                ax.plot(
                    fdata[:, i],
                    ms=2,
                    lw=1,
                    alpha=0.4,
                    label="Column {a}".format(a=read_columns[i]),
                )
        ax.legend()
        fig.tight_layout()
        try:
            if isfile(input_data):
                fig.savefig(
                    "{b}_plot.tiff".format(b=input_data),
                    dpi=300,
                    pil_kwargs={"compression": "tiff_lzw"},
                )
        except (UnicodeDecodeError, ValueError) as e:
            fig.savefig(
                "./wavelet_filtered.tiff",
                dpi=300,
                pil_kwargs={"compression": "tiff_lzw"},
            )
    return data_full


def _parseargs():
    parser = argparse.ArgumentParser(
        description="wavelet_filter_1d (version 0.1.0)\n\n",
        epilog="Developed 2019 by Huy Nguyen, Gruebele-Lyding Groups\n"
        "University of Illinois at Urbana-Champaign\n",
    )
    parser.add_argument(
        "--input", "-i", help="Input path to data file", type=str, required=True
    )
    parser.add_argument(
        "--level",
        "-l",
        help="Number of decomposition levels (Default: 0)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--wavelet",
        "-w",
        help="Name of the mother wavelet (Default: biorthogonal 6.8)",
        type=str,
        default="bior6.8",
    )
    parser.add_argument(
        "--thresholdMode",
        "-t",
        help="Threshold mode: soft, hard, greater, less, garrote",
        type=str,
        choices=["soft", "hard", "greater", "less", "garrote"],
        default="soft",
    )
    parser.add_argument(
        "--delimiter",
        "-d",
        help="Data file delimiter. Default: ',' ",
        type=str,
        default=",",
    )
    parser.add_argument(
        "--skiprows", "-s", help="Rows to skip. Default: 0 ", type=int, default=0
    )
    parser.add_argument(
        "--readColumns",
        "-r",
        help="Columns of data to read. Default: []",
        type=int,
        nargs='*',
        default=[],
    )
    parser.add_argument(
        "--savePlot",
        "-p",
        help="Save data plot? Default: False",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--abscissaColumn",
        "-a",
        help="Column to plot data against. Default: None",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parseargs()
    input_data = Path(args.input)
    wavelet_type = args.wavelet
    level = args.level
    thresholdMode = args.thresholdMode
    delimiter = args.delimiter
    skiprows = args.skiprows
    readColumns = args.readColumns
    # for i in args.readColumns:
    #     try:
    #         readColumns.append(int(i))
    #     except ValueError:
    #         pass
    savePlot = args.savePlot
    abscissaColumn = args.abscissaColumn
    wavelet_filter_1d(
        input_data,
        wavelet_type=wavelet_type,
        level=level,
        threshold_mode=thresholdMode,
        data_delimiter=delimiter,
        data_skiprows=skiprows,
        read_columns=readColumns,
        save_plot=savePlot,
        abscissa_column=abscissaColumn,
    )
