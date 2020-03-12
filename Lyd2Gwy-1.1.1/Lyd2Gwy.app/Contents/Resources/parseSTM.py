#!/Users/ordinary/anaconda3/bin/python

#  Script is based on Adrian Radocea's parseme.py
import struct, os
import numpy as np


class ACitsBlock:
    def __init__(
        self,
        block_data_type,
        block_mode,
        data,
        block_number,
        cits_spec_raw_deglitch,
        cits_spec_raw_glitch_threshold,
        cits_spec_raw_smooth,
        cits_spec_raw_smooth_order,
        cits_spec_raw_smooth_n_fit,
    ):
        self.cits_block_data_type = block_data_type
        self.cits_block_mode = block_mode
        self.data = data
        self.cits_spec_raw_deglitch = cits_spec_raw_deglitch
        self.cits_spec_raw_glitch_threshold = cits_spec_raw_glitch_threshold
        self.cits_spec_raw_smooth = cits_spec_raw_smooth
        self.cits_spec_raw_smooth_order = cits_spec_raw_smooth_order
        self.cits_spec_raw_smooth_n_fit = cits_spec_raw_smooth_n_fit
        self.realdata = 0
        self.intdata = 0
        if self.cits_block_data_type == 0:
            self.intdata = data
        elif self.cits_block_data_type == 1:
            self.realdata = data
        self.block_number = block_number


class ASpecBlock:
    def __init__(
        self,
        max_spec_per_coord,
        spec_block_data_type,
        spec_block_label,
        spec_mode,
        spec_num_spec,
        spec_pt_per_spec,
        spec_avg_num_spec,
        spec_hex_num_spec,
        spec_x_num_spec,
        spec_y_num_spec,
        spec_u_num_spec,
        spec_spread_type,
        spec_lead_pts,
        spec_settle,
        spec_vstrt,
        spec_vfnsh,
        spec_istrt,
        spec_ifnsh,
        spec_pt_del,
        spec_zstrt,
        spec_zfnsh,
        spec_zscan,
        spec_avg_del,
        spec_x_spec_inc,
        spec_y_spec_inc,
        spec_r_x_cen,
        spec_r_y_cen,
        spec_rect_angl,
        spec_u_x_cen,
        spec_u_y_cen,
        spec_user_angl,
        spec_hex_pt_sep,
        spec_h_x_cen,
        spec_h_y_cen,
        spec_hex_angl,
        spec_lock_in_der,
        spec_lock_in_range,
        spec_lock_in_tau,
        spec_dith_ampl,
        spec_dith_freq,
        spec_cusp_index,
        spec_lock_in_point_delay,
        spec_lock_in_full_scale_v,
        spec_skip_endpoint_ramps,
        spec_pt_num_average,
        spec_pt_avg_delay,
        spec_potentiometry,
        spec_potentio_use_samp_intv,
        spec_potentio_samp_interval,
        spec_potentio_lower_rail_v,
        spec_potentio_upper_rail_v,
        spec_potentio_lower_rail_fixed,
        spec_potentio_upper_rail_fixed,
        spec_initial_v_use_scan_value,
        spec_initial_i_use_scan_value,
        spec_initial_trans_together,
        spec_initial_trans_i_first,
        spec_initial_v,
        spec_initial_v_trans_time,
        spec_initial_i,
        spec_initial_i_trans_time,
        spec_cusp_voltage,
        spec_delay_before_atod,
        spec_set_initial_ds,
        spec_initial_ds,
        spec_initial_ds_trans_time,
        spec_current_channel_0_on,
        spec_current_channel_1_on,
        spec_current_channel_2_on,
        spec_current_channel_3_on,
        spec_current_average_mode,
        spec_dither_only,
        spec_leave_dither_on,
    ):
        self.max_spec_per_coord = max_spec_per_coord
        self.spec_block_data_type = spec_block_data_type
        self.spec_block_label = spec_block_label
        self.spec_mode = spec_mode
        self.spec_num_spec = spec_num_spec
        self.spec_pt_per_spec = spec_pt_per_spec
        self.spec_avg_num_spec = spec_avg_num_spec
        self.spec_hex_num_spec = spec_hex_num_spec
        self.spec_x_num_spec = spec_x_num_spec
        self.spec_y_num_spec = spec_y_num_spec
        self.spec_u_num_spec = spec_u_num_spec
        self.spec_spread_type = spec_spread_type
        self.spec_lead_pts = spec_lead_pts
        self.spec_settle = spec_settle
        self.spec_vstrt = spec_vstrt
        self.spec_vfnsh = spec_vfnsh
        self.spec_istrt = spec_istrt
        self.spec_ifnsh = spec_ifnsh
        self.spec_pt_del = spec_pt_del
        self.spec_zstrt = spec_zstrt
        self.spec_zfnsh = spec_zfnsh
        self.spec_zscan = spec_zscan
        self.spec_avg_del = spec_avg_del
        self.spec_x_spec_inc = spec_x_spec_inc
        self.spec_y_spec_inc = spec_y_spec_inc
        self.spec_r_x_cen = spec_r_x_cen
        self.spec_r_y_cen = spec_r_y_cen
        self.spec_rect_angl = spec_rect_angl
        self.spec_u_x_cen = spec_u_x_cen
        self.spec_u_y_cen = spec_u_y_cen
        self.spec_user_angl = spec_user_angl
        self.spec_hex_pt_sep = spec_hex_pt_sep
        self.spec_h_x_cen = spec_h_x_cen
        self.spec_h_y_cen = spec_h_y_cen
        self.spec_hex_angl = spec_hex_angl
        self.spec_lock_in_der = spec_lock_in_der
        self.spec_lock_in_range = spec_lock_in_range
        self.spec_lock_in_tau = spec_lock_in_tau
        self.spec_dith_ampl = spec_dith_ampl
        self.spec_dith_freq = spec_dith_freq
        self.spec_cusp_index = spec_cusp_index
        self.spec_lock_in_point_delay = spec_lock_in_point_delay
        self.spec_lock_in_full_scale_v = spec_lock_in_full_scale_v
        self.spec_skip_endpoint_ramps = spec_skip_endpoint_ramps
        self.spec_pt_num_average = spec_pt_num_average
        self.spec_pt_avg_delay = spec_pt_avg_delay
        self.spec_potentiometry = spec_potentiometry
        self.spec_potentio_use_samp_intv = spec_potentio_use_samp_intv
        self.spec_potentio_samp_interval = spec_potentio_samp_interval
        self.spec_potentio_lower_rail_v = spec_potentio_lower_rail_v
        self.spec_potentio_upper_rail_v = spec_potentio_upper_rail_v
        self.spec_potentio_lower_rail_fixed = spec_potentio_lower_rail_fixed
        self.spec_potentio_upper_rail_fixed = spec_potentio_upper_rail_fixed
        self.spec_initial_v_use_scan_value = spec_initial_v_use_scan_value
        self.spec_initial_i_use_scan_value = spec_initial_i_use_scan_value
        self.spec_initial_trans_together = spec_initial_trans_together
        self.spec_initial_trans_i_first = spec_initial_trans_i_first
        self.spec_initial_v = spec_initial_v
        self.spec_initial_v_trans_time = spec_initial_v_trans_time
        self.spec_initial_i = spec_initial_i
        self.spec_initial_i_trans_time = spec_initial_i_trans_time
        self.spec_cusp_voltage = spec_cusp_voltage
        self.spec_delay_before_atod = (spec_delay_before_atod,)
        self.spec_set_initial_ds = spec_set_initial_ds
        self.spec_initial_ds = spec_initial_ds
        self.spec_initial_ds_trans_time = spec_initial_ds_trans_time
        self.spec_current_channel_0_on = spec_current_channel_0_on
        self.spec_current_channel_1_on = spec_current_channel_1_on
        self.spec_current_channel_2_on = spec_current_channel_2_on
        self.spec_current_channel_3_on = spec_current_channel_3_on
        self.spec_current_average_mode = spec_current_average_mode
        self.spec_dither_only = spec_dither_only
        self.spec_leave_dither_on = spec_leave_dither_on
        self.coords = []
        self.stsdata = []

        # NOTE: for now I don't know what spec_mode corresponds to which, except 4: I vs V var. S; 2: I vs S con. V
        if self.spec_mode in [0, 1, 3, 4]:
            self.vrange = np.linspace(
                self.spec_vstrt, self.spec_vfnsh, num=self.spec_pt_per_spec
            )
        elif spec_mode == 2:
            self.srange = np.linspace(
                self.spec_zstrt, self.spec_zfnsh, num=self.spec_pt_per_spec
            )
        # else:
        #     self.irange = np.linspace(
        #         self.spec_istrt, self.spec_ifnsh, num=self.spec_pt_per_spec
        #     )


class STMfile:
    def __init__(self, file_path):
        self.fp = file_path
        self.fn = os.path.split(file_path)[1]
        with open(file_path, "rb") as f:
            self.bin_data = open(file_path, "rb").read()
        self.numbufs = self.parseChunk(b"BUFF_0", "<H")[0]
        self.dimensions = self.findDimension()
        self.BufDataTypes = self.parseChunk(b"SCAN_5A ", "<" + "H" * self.numbufs)
        self.modTime = os.path.getmtime(file_path)
        self.iscan, self.vscan, self.tsamp = self.parseChunk(b"I_V_T   ", "<" + "f" * 3)
        self.scan_mode = self.parseChunk(b"SCAN    ", "<h")[0]
        self.buf_scan_modes = self.parseChunk(b"S_MODE_1", "<" + "h" * self.numbufs)
        self.xst, self.xfin, self.xinc, self.x_ofset, self.yst, self.yfin, self.yinc, self.y_ofset, self.scnxin, self.scnyin, self.theta, self.scan_del = self.parseChunk(
            b"SCAN    ", "<" + "f" * 12, skip_steps=2
        )
        self.scan_ad_check, self.scan_up_down, self.scanning_up, self.nsampl, self.nscans, self.xnum, self.ynum = self.parseChunk(
            b"SCAN    ", "<" + "h" * 7, skip_steps=50
        )
        self.xcal, self.ycal, self.zcal, self.xver, self.yver, self.zver = self.parseChunk(
            b"CAL     ", "<" + "f" * 6
        )
        self.cur_gain, self.hv_gain, self.amplif = self.parseChunk(
            b"GAIN_1  ", "<" + "f" * 3
        )
        self.top_ad_ver, self.top_ad_max_gain, self.cur_ad_ver, self.cur_ad_gain, self.err_ad_ver, self.err_ad_gain, self.lock_ad_ver, self.lock_ad_gain = self.parseChunk(
            b"ATOD    ", "<" + "f" * 8
        )
        self.top_ad_gain = 1 + (self.top_ad_ver / 10) * self.top_ad_max_gain
        try:
            self.prop_gain, self.intg_gain, self.der_gain, self.atod1_gain, self.atod2_gain = self.parseChunk(
                b"ELEC1   ", "<" + "f" * 5
                )
        except Exception:
            self.prop_gain, self.intg_gain, self.der_gain, self.atod1_gain, self.atod2_gain = (
                None,
                None,
                None,
                None,
                None,
            )
        try:
            self.atod1_chanl, self.atod2_chanl = self.parseChunk(
                b"ELEC1   ", "<" + "h" * 2, skip_steps=20
            )
        except Exception:
            self.atod1_chanl, self.atod2_chanl = (None, None)
        self.valid_spec, self.bias_to_probe, self.cits_on = self.parseChunk(
            b"FLAG    ", "<" + "h" * 3
        )
        self.valid_cits = self.parseChunk(b"FLAG_1  ", "<h")[0]
        self.dsp_atod_max_v = 2.75
        self.dsp_dtoa_max_v = 3.00
        self.dtoa_max_value = 32767
        self.atodsign = 0
        if self.bin_data.find(b"ATODSIGN") != -1:
            self.atodsign = 1
        self.jj_z_gain = 3.35  # Jim Janick Gain

        # specify some titles
        _titles = {
            0: "Current Buffer: ",
            1: "Topographic Buffer: ",
            2: "Dig. Topographic Buffer: ",
            3: "Error Image: ",
            4: "Lock-In Image: ",
            5: "d2I/dV2 Buffer: ",
        }
        self.BufTitles = {}
        for bufnum, scan_mode in enumerate(self.buf_scan_modes):
            if scan_mode in _titles:
                self.BufTitles[bufnum] = _titles[scan_mode] + "%s\0" % bufnum

        # specify some units
        _units = {0: "A\0", 1: "m\0", 2: "m\0", 3: " \0", 4: "V\0", 5: "V\0"}
        self.BufUnits = {}
        for bufnum, scan_mode in enumerate(self.buf_scan_modes):
            if scan_mode in _units:
                self.BufUnits[bufnum] = _units[scan_mode]

        # STS BLOCKS
        self.stsblocks = {}
        if self.valid_spec == 1 and self.valid_cits == 0:
            self.spec_num_blocks, self.spec_array_offset, self.spec_max_points_per_spec, self.spec_max_spec_per_coord, self.spec_max_num_spec, self.spec_active_block = self.parseChunk(
                b"SPEC_M1", "<hfhhhh"
            )
            if self.spec_num_blocks > 0:
                for i in range(self.spec_num_blocks):
                    _tmp = self.parseChunk(
                        b"SPEC_M1", "<h80s8hfh22fh4fh2f2hf2h3f6h6fh2f7h", skip_steps=14
                    )
                    spec_block = ASpecBlock(2, *_tmp)
                    for j in range(spec_block.spec_num_spec):
                        spc = 1
                        if spec_block.spec_potentiometry and (
                            spec_block.spec_mode == 1 or spec_block.spec_mode == 4
                        ):
                            spc = 2
                        if spec_block.spec_lock_in_der:
                            spc = 2
                        spec_block.max_spec_per_coord = spc
                    _parsed = spec_block.spec_num_spec * 2
                    _skipped = _parsed * i * struct.calcsize("<h")
                    spec_block.coords = np.array(
                        self.parseChunk(
                            b"SPC_MDAT", "<{a}h".format(a=_parsed), skip_steps=_skipped
                        )
                    )
                    spec_block.coords = spec_block.coords.reshape(
                        (2, spec_block.spec_num_spec)
                    )
                    spec_block.coords = np.stack(
                        (spec_block.coords[0, :], spec_block.coords[1, :]), axis=-1
                    )
                    spec_block.coords[:, 1] = self.ynum - spec_block.coords[:, 1]
                    self.stsblocks[i] = spec_block

                for i in range(self.spec_num_blocks):
                    _skipped = (
                        self.stsblocks[i].spec_num_spec
                        * 2
                        * self.spec_num_blocks
                        * struct.calcsize("<h")
                    )
                    if self.stsblocks[i].max_spec_per_coord == 1:
                        _skipped1 = (
                            self.stsblocks[i].spec_pt_per_spec
                            * self.stsblocks[i].spec_num_spec
                            * i
                            * struct.calcsize("<f")
                        )
                        self.stsblocks[i].stsdata = self.parseChunk(
                            b"SPC_MDAT",
                            "<{a}f".format(
                                a=self.stsblocks[i].spec_pt_per_spec
                                * self.stsblocks[i].spec_num_spec
                            ),
                            skip_steps=_skipped + _skipped1,
                        )
                        self.stsblocks[i].stsdata = np.reshape(
                            self.stsblocks[i].stsdata,
                            (
                                self.stsblocks[i].spec_num_spec,
                                1,
                                self.stsblocks[i].spec_pt_per_spec,
                            ),
                        )
                        self.stsblocks[i].stsdata = (
                            self.stsblocks[i].stsdata
                            * 2
                            * float(self.dsp_atod_max_v)
                            / 2 ** 16
                            / self.cur_gain
                            / self.cur_ad_gain
                        )
                    elif self.stsblocks[i].max_spec_per_coord == 2:
                        _skipped1 = (
                            self.stsblocks[i].spec_pt_per_spec
                            * self.stsblocks[i].spec_num_spec
                            * 2
                            * i
                            * struct.calcsize("<f")
                        )
                        self.stsblocks[i].stsdata = self.parseChunk(
                            b"SPC_MDAT",
                            "<{a}f".format(a=2 * self.stsblocks[i].spec_pt_per_spec),
                            skip_steps=_skipped + _skipped1,
                        )
                        self.stsblocks[i].stsdata = np.reshape(
                            self.stsblocks[i].stsdata,
                            (
                                self.stsblocks[i].spec_num_spec,
                                2,
                                self.stsblocks[i].spec_pt_per_spec,
                            ),
                        )
                        self.stsblocks[i].stsdata = (
                            self.stsblocks[i].stsdata
                            * 2
                            * float(self.dsp_atod_max_v)
                            / 2 ** 16
                            / self.cur_gain
                            / self.cur_ad_gain
                        )

            # CITS BLOCKS
        self.cits_blocks = {}
        if self.valid_cits == 1:
            self.cits_spec_mode, self.cits_num_buff, self.cits_oversample_mult, self.cits_spec_vstrt, self.cits_spec_vfnsh, self.cits_spec_pt_del, self.cits_spec_avg_num_spec, self.cits_spec_avg_del, self.cits_spec_lock_in_point_delay = self.parseChunk(
                b"CITS", "<hhhfffhff"
            )
            self.cits_bias = []
            _cits_bias_start_ind = self.find(b"CITS") + 8 + 2 * 3 + 4 * 3 + 2 + 4 * 2
            for i in range(self.cits_num_buff):
                self.cits_bias.append(
                    struct.unpack(
                        "<f",
                        self.bin_data[
                            _cits_bias_start_ind
                            + 4 * i : _cits_bias_start_ind
                            + 4 * i
                            + 4
                        ],
                    )[0]
                )

            if self.find(b"CITS_1") != -1:
                self.cits_spec_dither_only, self.cits_topo_average, self.cits_topo_num_average, self.cits_spec_fit, self.cits_spec_fit_pts = self.parseChunk(
                    b"CITS_1", "<hhhhh"
                )
            if self.find(b"CITS_2") != -1:
                self.cits_num_blocks, self.cits_analysis_block, self.cits_display_block = self.parseChunk(
                    b"CITS_2", "<hhh"
                )
            if self.find(b"CITS_3") != -1:
                self.cits_spec_istrt, self.cits_spec_ifnsh, self.cits_spec_zstrt, self.cits_spec_zfnsh, self.cits_spec_zscan = self.parseChunk(
                    b"CITS_2", "<fffff"
                )
            if self.find(b"CITS_5") != -1:
                self.cits_log_temperature = self.parseChunk(b"CITS_5", "<h")[0]
                if self.cits_log_temperature:
                    self.cits_temperature_pts, self.cits_temperature_atod_channel, self.cits_temperature_log_interval, self.cits_temperature_log_sampl_dt, self.cits_temperature_conv_factor = self.parseChunk(
                        b"CITS_5", "<hhfff", skip_steps=2
                    )
            if self.find(b"CITS_6") != -1:
                self.cits_spec_dither_only, self.cits_topo_average, self.cits_topo_num_average = self.parseChunk(
                    b"CITS_6", "<hhh"
                )
            _cits_block_mode = []
            _cits2_i = self.find(b"CITS_2")
            if _cits2_i == -1:
                raise Exception("CITS_2 not found")
            for i in range(self.cits_num_blocks):
                _skipped = 2 * 3 + 2 * i
                _cits_block_mode.append(
                    self.parseChunk(b"CITS_2", "<h", skip_steps=_skipped)[0]
                )
            _cits_block_data_type = []
            if self.find(b"CITS_4") != -1:
                for i in range(self.cits_num_blocks):
                    _cits_block_data_type.append(self.parseChunk(b"CITS_4", "<h")[0])
            # if self.find(b'CITS_6') != -1:
            #     for i in range(self.cits_num_blocks):
            #         _tmp = self.parseChunk(b'CITS_6', "<5h", skip_steps=6)  # skip 3 integers after cits_6 flag
            _intdata = []
            _realdata = []

            for j in range(self.cits_num_blocks):
                _tmp = self.parseChunk(
                    b"CITS_6", "<5h", skip_steps=6 + j * 5 * struct.calcsize("<h")
                )  # skip 3 integers after cits_6 flag
                if _cits_block_data_type[j] == 0:
                    for i in range(self.cits_num_buff):
                        _skipped = (
                            self.dimensions[0]
                            * self.dimensions[1]
                            * (
                                struct.calcsize("<h") * self.numbufs
                                + struct.calcsize("<h") * i
                            )
                        )
                        _arr = self.parseChunk(
                            b"IMG_BUF",
                            "<{j}h".format(j=self.dimensions[0] * self.dimensions[1]),
                            skip_steps=_skipped,
                        )
                        _arr = np.reshape(_arr, (self.ynum, self.xnum))
                        _arr = np.flip(_arr, 0)
                        _intdata.append(_arr)
                    _intdata = np.array(_intdata)
                    _intdata = (
                        _intdata
                        * self.dsp_atod_max_v
                        * 2
                        / 2 ** 16
                        / self.cur_gain
                        / self.cur_ad_gain
                    )
                    _ABlock = ACitsBlock(
                        _cits_block_data_type[j],
                        _cits_block_mode[j],
                        _intdata,
                        j,
                        *_tmp
                    )
                    self.cits_blocks[j] = _ABlock

                elif _cits_block_data_type[j] == 1:
                    for i in range(self.cits_num_buff):
                        _skipped = (
                            self.dimensions[0]
                            * self.dimensions[1]
                            * (
                                struct.calcsize("<h") * self.numbufs
                                + struct.calcsize("<f") * i
                            )
                        )
                        _arr = self.parseChunk(
                            b"IMG_BUF",
                            "<{j}f".format(j=self.dimensions[0] * self.dimensions[1]),
                            skip_steps=_skipped,
                        )
                        _arr = np.reshape(_arr, (self.ynum, self.xnum))
                        _arr = np.flip(_arr, 0)
                        _realdata.append(_arr)
                    _realdata = np.array(_realdata)
                    _realdata = (
                        _realdata
                        * self.dsp_atod_max_v
                        * 2
                        / 2 ** 16
                        / self.cur_gain
                        / self.cur_ad_gain
                    )
                    _ABlock = ACitsBlock(
                        _cits_block_data_type[j],
                        _cits_block_mode[j],
                        _realdata,
                        j,
                        *_tmp
                    )
                    self.cits_blocks[j] = _ABlock

    def find(self, header):
        return self.bin_data.find(header)

    def findDimension(self):
        i = self.bin_data.find(b"SCAN_1")
        if i == -1:
            raise Exception("Missing data")
        return struct.unpack(
            "<HH", self.bin_data[i - 4 : i]
        )  # this will be a tuple of 2 numbers

    def parseChunk(self, header, format, skip_steps=0):
        # if header == "CITS_2" or header == "CITS_4":
        #     pass
        i = self.find(header)
        if i == -1:
            raise Exception("missing data")
        return struct.unpack(
            format,
            self.bin_data[
                i + 8 + skip_steps : i + 8 + skip_steps + struct.calcsize(format)
            ],
        )  # There are 8 spaces from the header to the actual data

    def convCharToByte(self, c):
        num = ord(c)
        return struct.pack("<B", num)

    def convStringToByte(self, s, format="utf-8"):
        s = s.encode(format)
        return struct.pack("<" + "%is" % len(s), s)

    def convIntToByte(self, i, format="<i"):
        return struct.pack(format, i)

    def convDoubleToByte(self, d):
        return struct.pack("<d", d)

    def convBoolToByte(self, b):
        return struct.pack("<?", b)

    def si_unit_Object(self, bufnum, unit=None):
        """Return unit object and length in number of bytes"""
        this_unit = self.BufUnits[bufnum]
        if unit != None:
            this_unit = unit + "\0"
        chanByte = self.convStringToByte("unitstr\0")
        chanByte += self.convCharToByte("s")
        chanByte += self.convStringToByte(this_unit)
        return chanByte, len(chanByte)

    def cits_unit_Object(self, unit=None):
        """Return unit object and length in number of bytes"""
        this_unit = " \0"
        if unit != None:
            this_unit = unit + "\0"
        brickByte = self.convStringToByte("unitstr\0")
        brickByte += self.convCharToByte("s")
        brickByte += self.convStringToByte(this_unit)
        return brickByte, len(brickByte)

    def buildGwyDataLine(self, stsblocknum, sts_spec_num):
        # return channel and channel length
        if stsblocknum >= self.spec_num_blocks:
            raise Exception(
                "addGwyDataline went out of range of number of linenum (stsblocks)"
            )
        elif sts_spec_num >= self.stsblocks[stsblocknum].spec_num_spec:
            raise Exception(
                "addGwyDataline went out of range of number of linenum (spec_num_spec)"
            )
        # res
        chanByte = self.convStringToByte("res\0")
        chanByte += self.convCharToByte("i")
        if self.stsblocks[stsblocknum].max_spec_per_coord == 2:
            chanByte += self.convIntToByte(
                self.stsblocks[stsblocknum].spec_pt_per_spec * 2
            )
        else:
            chanByte += self.convIntToByte(self.stsblocks[stsblocknum].spec_pt_per_spec)

        # real
        chanByte += self.convStringToByte("real\0")
        chanByte += self.convCharToByte("d")
        if self.stsblocks[stsblocknum].spec_mode in [0, 1, 3, 4]:
            chanByte += self.convDoubleToByte(
                np.abs(
                    self.stsblocks[stsblocknum].spec_vstrt
                    - self.stsblocks[stsblocknum].spec_vfnsh
                )
            )
        elif self.stsblocks[stsblocknum].spec_mode == 2:
            chanByte += self.convDoubleToByte(
                0.1
                * np.abs(
                    self.stsblocks[stsblocknum].spec_zstrt
                    - self.stsblocks[stsblocknum].spec_zfnsh
                )
            )
        # elif self.stsblocks[stsblocknum].spec_mode == 4:
        #     chanByte += self.convDoubleToByte(
        #         np.abs(
        #             self.stsblocks[stsblocknum].spec_istrt
        #             - self.stsblocks[stsblocknum].spec_ifnsh
        #         )
        #     )

        # off
        chanByte += self.convStringToByte("off\0")
        chanByte += self.convCharToByte("d")
        if self.stsblocks[stsblocknum].spec_mode in [0, 1, 3, 4]:
            chanByte += self.convDoubleToByte(
                np.min(
                    (
                        self.stsblocks[stsblocknum].spec_vstrt,
                        self.stsblocks[stsblocknum].spec_vfnsh,
                    )
                )
            )
        elif self.stsblocks[stsblocknum].spec_mode == 2:
            chanByte += self.convDoubleToByte(
                0.1
                * np.min(
                    (
                        self.stsblocks[stsblocknum].spec_zstrt,
                        self.stsblocks[stsblocknum].spec_zfnsh,
                    )
                )  # Angstrom to nm
            )
        # else:
        #     chanByte += self.convDoubleToByte(
        #         np.min(
        #             (
        #                 self.stsblocks[stsblocknum].spec_istrt,
        #                 self.stsblocks[stsblocknum].spec_ifnsh,
        #             )
        #         )
        #     )

        # si_unit_x
        chanByte += self.convStringToByte("si_unit_x\0")
        chanByte += self.convCharToByte("o")
        chanByte += self.convStringToByte("GwySIUnit\0", format="ascii")
        if self.stsblocks[stsblocknum].spec_mode in [0, 1, 3, 4]:
            x_unit_chunk, length_x_unit = self.cits_unit_Object(unit="V")
        elif self.stsblocks[stsblocknum].spec_mode == 2:
            x_unit_chunk, length_x_unit = self.cits_unit_Object(unit="nm")
        # else:
        #     x_unit_chunk, length_x_unit = self.cits_unit_Object(unit="A")
        chanByte += self.convIntToByte(length_x_unit)
        chanByte += x_unit_chunk

        # si_unit_y
        chanByte += self.convStringToByte("si_unit_y\0")
        chanByte += self.convCharToByte("o")
        chanByte += self.convStringToByte("GwySIUnit\0", format="ascii")
        # if self.stsblocks[stsblocknum].spec_mode in [0, 1, 2]:
        y_unit_chunk, length_y_unit = self.cits_unit_Object(unit="A")
        # else:
        #     y_unit_chunk, length_y_unit = self.cits_unit_Object(unit="nm")
        chanByte += self.convIntToByte(length_y_unit)
        chanByte += y_unit_chunk

        # data
        chanByte += self.convStringToByte("data\0")
        chanByte += self.convCharToByte("D")
        if self.stsblocks[stsblocknum].max_spec_per_coord == 2:
            chanByte += self.convIntToByte(
                self.stsblocks[stsblocknum].spec_pt_per_spec * 2, format="<I"
            )
        elif self.stsblocks[stsblocknum].max_spec_per_coord == 1:
            chanByte += self.convIntToByte(
                self.stsblocks[stsblocknum].spec_pt_per_spec, format="<I"
            )
        to_data = self.stsblocks[stsblocknum].stsdata[sts_spec_num, :, :].copy()
        # if self.stsblocks[stsblocknum].spec_mode in [3, 4]:
        # to_data *= 0.1
        if self.stsblocks[stsblocknum].spec_mode in [0, 1, 3, 4]:
            if (
                self.stsblocks[stsblocknum].spec_vstrt
                > self.stsblocks[stsblocknum].spec_vfnsh
            ):
                chanByte += (np.flip(np.array(to_data).flatten())).tobytes()
            else:
                chanByte += (np.array(to_data).flatten()).tobytes()
        elif self.stsblocks[stsblocknum].spec_mode == 2:
            if (
                self.stsblocks[stsblocknum].spec_zstrt
                > self.stsblocks[stsblocknum].spec_zfnsh
            ):
                chanByte += (np.flip(np.array(to_data).flatten())).tobytes()
            else:
                chanByte += (np.array(to_data).flatten()).tobytes()
        # else:
        #     if (
        #         self.stsblocks[stsblocknum].spec_istrt
        #         > self.stsblocks[stsblocknum].spec_ifnsh
        #     ):
        #         chanByte += (np.flip(np.array(to_data).flatten())).tobytes()
        #     else:
        #         chanByte += (np.array(to_data).flatten()).tobytes()
        return chanByte, len(chanByte)

    def addGwyDataLine(self, stsblocknum, sts_spec_num):
        dataLine = self.convStringToByte("GwyDataLine\0", format="ascii")
        chanByteObject, length_of_it = self.buildGwyDataLine(stsblocknum, sts_spec_num)
        dataLine += self.convIntToByte(length_of_it, format="<I")
        dataLine += chanByteObject
        return dataLine, len(dataLine)

    def buildGwySpectra(self, stsblocknum):
        if stsblocknum >= self.spec_num_blocks:
            raise Exception("stsblocknum out of bound in buildGwySpectra")
        # title
        coordByte = self.convStringToByte("title\0")
        coordByte += self.convCharToByte("s")
        coordByte += self.convStringToByte("STS data\0")

        # si_unit_xy
        coordByte += self.convStringToByte("si_unit_xy\0")
        coordByte += self.convCharToByte("o")
        coordByte += self.convStringToByte("GwySIUnit\0", format="ascii")
        xy_unit_chunk, length_of_it_coord = self.cits_unit_Object(unit="m")
        coordByte += self.convIntToByte(length_of_it_coord)
        coordByte += xy_unit_chunk

        # coords
        coordByte += self.convStringToByte("coords\0")
        coordByte += self.convCharToByte("D")
        _no_coords = 2 * self.stsblocks[stsblocknum].spec_num_spec
        coordByte += self.convIntToByte(_no_coords, format="<I")
        _coords = np.array(self.stsblocks[stsblocknum].coords).flatten()
        coordByte += ((self.xfin - self.xst) / self.xnum * 1e-10 * _coords).tobytes()

        # data
        coordByte += self.convStringToByte("data\0", format="ascii")
        coordByte += self.convCharToByte("O")
        coordByte += self.convIntToByte(
            self.stsblocks[stsblocknum].spec_num_spec, format="<I"
        )
        _tmpbyte = []
        _lenbyte = []
        for num_spec in range(self.stsblocks[stsblocknum].spec_num_spec):
            _tmpvar = self.addGwyDataLine(stsblocknum, num_spec)
            _tmpbyte.append(_tmpvar[0])
            _lenbyte.append(_tmpvar[1])

        for i in range(self.stsblocks[stsblocknum].spec_num_spec):
            coordByte += _tmpbyte[i]
        return coordByte, len(coordByte)

    def addGwySpectra(self, stsblocknum):
        spectraField = self.convStringToByte(
            "/sps/{i}\0".format(i=stsblocknum), format="ascii"
        )
        spectraField += self.convCharToByte("o")
        spectraField += self.convStringToByte("GwySpectra\0", format="ascii")
        spectra, length_of_it_spectra = self.buildGwySpectra(stsblocknum)
        spectraField += self.convIntToByte(length_of_it_spectra, "<I")
        spectraField += spectra
        return spectraField, len(spectraField)

    def brickDataByteObject(self, bricknum):
        if bricknum >= self.cits_num_blocks:
            raise Exception(
                "brickDataByteObject went out of range of number of CITS blocks"
            )
        # xres
        brickByte = self.convStringToByte("xres\0")
        brickByte += self.convCharToByte("i")
        brickByte += self.convIntToByte(self.xnum)

        # yres
        brickByte += self.convStringToByte("yres\0")
        brickByte += self.convCharToByte("i")
        brickByte += self.convIntToByte(self.ynum)

        # zres
        brickByte += self.convStringToByte("zres\0")
        brickByte += self.convCharToByte("i")
        brickByte += self.convIntToByte(self.cits_num_buff)

        # xreal
        brickByte += self.convStringToByte("xreal\0")
        brickByte += self.convCharToByte("d")
        brickByte += self.convDoubleToByte((self.xfin - self.xst) * 1e-10)

        # yreal
        brickByte += self.convStringToByte("yreal\0")
        brickByte += self.convCharToByte("d")
        brickByte += self.convDoubleToByte((self.yfin - self.yst) * 1e-10)

        # zreal
        brickByte += self.convStringToByte("zreal\0")
        brickByte += self.convCharToByte("d")
        brickByte += self.convDoubleToByte(self.cits_spec_vfnsh - self.cits_spec_vstrt)

        # zoff
        brickByte += self.convStringToByte("zoff\0")
        brickByte += self.convCharToByte("d")
        brickByte += self.convDoubleToByte(self.cits_spec_vstrt)

        # si_unit_x
        brickByte += self.convStringToByte("si_unit_x\0")
        brickByte += self.convCharToByte("o")
        brickByte += self.convStringToByte("GwySIUnit\0", format="ascii")
        x_unit_chunk, length_x_unit = self.cits_unit_Object(unit="m")
        brickByte += self.convIntToByte(length_x_unit)
        brickByte += x_unit_chunk

        # si_unit_y
        brickByte += self.convStringToByte("si_unit_y\0")
        brickByte += self.convCharToByte("o")
        brickByte += self.convStringToByte("GwySIUnit\0", format="ascii")
        y_unit_chunk, length_y_unit = self.cits_unit_Object(unit="m")
        brickByte += self.convIntToByte(length_y_unit)
        brickByte += y_unit_chunk

        # si_unit_z
        brickByte += self.convStringToByte("si_unit_z\0")
        brickByte += self.convCharToByte("o")
        brickByte += self.convStringToByte("GwySIUnit\0", format="ascii")
        z_unit_chunk, length_z_unit = self.cits_unit_Object(unit="V")
        brickByte += self.convIntToByte(length_z_unit)
        brickByte += z_unit_chunk

        # si_unit_w
        brickByte += self.convStringToByte("si_unit_w\0")
        brickByte += self.convCharToByte("o")
        brickByte += self.convStringToByte("GwySIUnit\0", format="ascii")
        w_unit_chunk, length_w_unit = self.cits_unit_Object(unit="A")
        brickByte += self.convIntToByte(length_w_unit)
        brickByte += w_unit_chunk

        # data
        brickByte += self.convStringToByte("data\0")
        brickByte += self.convCharToByte("D")
        brickByte += self.convIntToByte(
            self.xnum * self.ynum * self.cits_num_buff, format="<I"
        )

        if self.cits_blocks[bricknum].cits_block_data_type == 0:
            brickByte += self.cits_blocks[bricknum].intdata.tobytes()
        elif self.cits_blocks[bricknum].cits_block_data_type == 1:
            brickByte += self.cits_blocks[bricknum].realdata.tobytes()
        return brickByte, len(brickByte)

    def addGwyBrick(self, bricknum):
        if bricknum >= self.cits_num_buff:
            raise Exception("GwyBrick went out of range of number of volume buffers")
        brickField = self.convStringToByte("/brick/%i\0" % bricknum, format="ascii")
        brickField += self.convCharToByte("o")
        brickField += self.convStringToByte("GwyBrick\0", format="ascii")
        brickByte, length_of_it_brick = self.brickDataByteObject(bricknum)
        brickField += self.convIntToByte(length_of_it_brick, format="<I")
        brickField += brickByte
        brickField += self.convStringToByte(
            "/brick/%i/preview\0" % bricknum, format="ascii"
        )
        brickField += self.convCharToByte("o")
        brickField += self.convStringToByte("GwyDataField\0", format="ascii")
        chanByteObject, length_of_it_field = self.channelDataByteObject(
            0
        )  # show topographic buffer in preview
        brickField += self.convIntToByte(length_of_it_field, format="<I")
        brickField += chanByteObject
        return brickField, len(brickField)

    def addBrickTitle(self, bricknum):
        if bricknum >= self.cits_num_blocks:
            raise Exception("BrickTitle went out of range of number of brick blocks")
        BrickTitle = self.convStringToByte(
            "/brick/%i/title\0" % bricknum, format="ascii"
        )
        BrickTitle += self.convCharToByte("s")
        BrickTitle += self.convStringToByte("CITS %i\0" % bricknum, format="ascii")
        return BrickTitle, len(BrickTitle)

    def channelDataByteObject(self, bufnum):
        # return channel and channel length
        if bufnum >= self.numbufs:
            raise Exception(
                "channelDataByteObject went out of range of number of buffers"
            )
        # xres
        chanByte = self.convStringToByte("xres\0")
        chanByte += self.convCharToByte("i")
        chanByte += self.convIntToByte(self.xnum)

        # yres
        chanByte += self.convStringToByte("yres\0")
        chanByte += self.convCharToByte("i")
        chanByte += self.convIntToByte(self.ynum)

        # xreal
        chanByte += self.convStringToByte("xreal\0")
        chanByte += self.convCharToByte("d")
        chanByte += self.convDoubleToByte((self.xfin - self.xst) * 1e-10)

        # yreal
        chanByte += self.convStringToByte("yreal\0")
        chanByte += self.convCharToByte("d")
        chanByte += self.convDoubleToByte((self.yfin - self.yst) * 1e-10)

        # si_unit_xy
        chanByte += self.convStringToByte("si_unit_xy\0")
        chanByte += self.convCharToByte("o")
        chanByte += self.convStringToByte("GwySIUnit\0", format="ascii")
        xy_unit_chunk, length_xy_unit = self.si_unit_Object(bufnum, unit="m")
        chanByte += self.convIntToByte(length_xy_unit)
        chanByte += xy_unit_chunk

        # si_unit_z
        chanByte += self.convStringToByte("si_unit_z\0")
        chanByte += self.convCharToByte("o")
        chanByte += self.convStringToByte("GwySIUnit\0", format="ascii")
        z_unit_chunk, length_z_unit = self.si_unit_Object(bufnum)
        chanByte += self.convIntToByte(length_z_unit)
        chanByte += z_unit_chunk

        # data
        chanByte += self.convStringToByte("data\0")
        chanByte += self.convCharToByte("D")
        chanByte += self.convIntToByte(
            self.xnum * self.ynum, format="<I"
        )  # times 8 or not? Maybe just number of items in array

        # adjust data's multiplication factor
        mulfac = 1
        addfac = 0
        if self.BufDataTypes[bufnum] == 0:
            if self.buf_scan_modes[bufnum] == 0:
                mulfac = (
                    self.dsp_atod_max_v / self.dtoa_max_value
                ) / self.cur_ad_gain ** 2
                if not self.bias_to_probe and self.atodsign:
                    mulfac *= -1

            elif self.buf_scan_modes[bufnum] == 1 or self.buf_scan_modes[bufnum] == 2:
                mulfac = (self.dsp_dtoa_max_v / self.dtoa_max_value) * (
                    self.jj_z_gain * self.hv_gain * self.zcal * self.zver / 10
                )

            elif self.buf_scan_modes[bufnum] == 3:
                mulfac = (
                    self.dsp_atod_max_v / self.dtoa_max_value
                ) / self.cur_ad_gain ** 2

            elif self.buf_scan_modes[bufnum] == 4 or self.buf_scan_modes[bufnum] == 5:
                mulfac = self.dsp_atod_max_v / self.dtoa_max_value
                if not self.bias_to_probe and self.atodsign:
                    mulfac *= -1

        # add to chanByte
        # for i in reversed(range(self.ynum)):
        #     for j in range(self.xnum):
        #         chanByte += self.convDoubleToByte(self.allBuffers[bufnum][j,i]*mulfac*1E-10 + addfac)
        chanByte += (
            self.get_buffers([bufnum + 1])[bufnum + 1] * mulfac * 1e-10 + addfac
        ).tobytes()
        return chanByte, len(chanByte)

    def addGwyDataField(self, bufnum):
        if bufnum >= self.numbufs:
            raise Exception("GwyDataField went out of range of number of buffers")
        dataField = self.convStringToByte("/%i/data\0" % bufnum, format="ascii")
        dataField += self.convCharToByte("o")
        dataField += self.convStringToByte("GwyDataField\0", format="ascii")
        chanByteObject, length_of_it = self.channelDataByteObject(bufnum)
        dataField += self.convIntToByte(length_of_it, format="<I")
        dataField += chanByteObject
        return dataField, len(dataField)

    def addGwySelection(self, bufnum):
        """Return the selection/foo object and its length in number of bytes"""
        if bufnum >= self.numbufs:
            raise Exception("GwySelection went out of range of number of buffers")
        GwySelection = self.convStringToByte(
            "/%i/select/pointer\0" % bufnum, format="ascii"
        )
        GwySelection += self.convCharToByte("o")
        GwySelection += self.convStringToByte("GwySelectionPoint\0", format="ascii")
        GwySelection += self.convIntToByte(9, "<I")
        GwySelection += self.convStringToByte("max\0")
        GwySelection += self.convCharToByte("i")
        GwySelection += self.convIntToByte(1)
        return GwySelection, len(GwySelection)

    def addGwyTitle(self, bufnum):
        """Return the data/title object and its length in number of bytes"""
        if bufnum >= self.numbufs:
            raise Exception("GwyTitle went out of range of number of buffers")
        GwyTitle = self.convStringToByte("/%i/data/title\0" % bufnum, format="ascii")
        GwyTitle += self.convCharToByte("s")
        GwyTitle += self.convStringToByte(self.BufTitles[bufnum])
        return GwyTitle, len(GwyTitle)

    def addGwyVisible(self, bufnum):
        """Return the data/visible object and its length in number of bytes"""
        if bufnum >= self.numbufs:
            raise Exception("GwyVisible went out of range of number of buffers")
        GwyVis = self.convStringToByte("/%i/data/visible\0" % bufnum, format="ascii")
        GwyVis += self.convCharToByte("b")
        GwyVis += self.convBoolToByte(True)
        return GwyVis, len(GwyVis)

    def addGwyFilename(self):
        """Return the /filename object and its length in number of bytes"""
        GwyFn = self.convStringToByte("/filename\0", format="ascii")
        GwyFn += self.convCharToByte("s")
        GwyFn += self.convStringToByte(self.fp + ".gwy\0")
        return GwyFn, len(GwyFn)

    def addGwyPalette(self, bufnum):
        """Return base/palette object and its length in number of bytes"""
        GwyPalette = self.convStringToByte(
            "/%i/base/palette\0" % bufnum, format="ascii"
        )
        GwyPalette += self.convCharToByte("s")
        if self.buf_scan_modes[bufnum] == 1 or self.buf_scan_modes[bufnum] == 2:
            GwyPalette += self.convStringToByte("Gwyddion.net\0")
        else:
            GwyPalette += self.convStringToByte("Gray\0")
        return GwyPalette, len(GwyPalette)

    def addGwyHeader(self):
        """Return the header of the file"""
        GwyHeader = self.convCharToByte("G")
        GwyHeader += self.convCharToByte("W")
        GwyHeader += self.convCharToByte("Y")
        GwyHeader += self.convCharToByte("P")
        GwyHeader += self.convStringToByte("GwyContainer\0", format="ascii")
        return GwyHeader

    def toGwy(self):
        """Convert the STMfile class structure in a binary file of the gwyddion format"""
        # Write the magic header
        GwyHeader = self.addGwyHeader()
        GwyFn, GwyFn_size = self.addGwyFilename()  # the name
        GwyVises = []
        GwyTitles = []
        GwySelections = []
        GwyDataFields = []
        GwyBricks = []
        GwyBrickTitles = []
        GwySpectra = []
        #
        GwyPalettes = []
        for bufnum in range(self.numbufs):
            GwyVises.append(self.addGwyVisible(bufnum))
            GwySelections.append(self.addGwySelection(bufnum))
            GwyTitles.append(self.addGwyTitle(bufnum))
            GwyDataFields.append(self.addGwyDataField(bufnum))
            #
            GwyPalettes.append(self.addGwyPalette(bufnum))

        datasize = (
            sum([GwyVises[i][1] for i in range(len(GwyVises))])
            + sum([GwyDataFields[i][1] for i in range(len(GwyDataFields))])
            + sum([GwyTitles[i][1] for i in range(len(GwyTitles))])
            + sum([GwySelections[i][1] for i in range(len(GwySelections))])
            + sum([GwyPalettes[i][1] for i in range(len(GwyPalettes))])
            + +GwyFn_size
        )
        if self.valid_cits == 1:
            for bricknum in range(self.cits_num_blocks):
                GwyBricks.append(self.addGwyBrick(bricknum))
                GwyBrickTitles.append(self.addBrickTitle(bricknum))
            datasize += sum([GwyBricks[i][1] for i in range(len(GwyBricks))]) + sum(
                [GwyBrickTitles[i][1] for i in range(len(GwyBrickTitles))]
            )

        elif self.valid_spec == 1 and self.valid_cits == 0:
            for stsblocknum in range(self.spec_num_blocks):
                GwySpectra.append(self.addGwySpectra(stsblocknum))
            datasize += sum([GwySpectra[i][1] for i in range(len(GwySpectra))])

        with open(self.fp + ".gwy", "wb") as f:
            f.write(GwyHeader)
            f.write(
                self.convIntToByte(datasize, format="<I")
            )  # size of the entire data file
            f.write(GwyFn)
            for i in range(self.numbufs):
                f.write(GwySelections[i][0])
                f.write(GwyVises[i][0])
                f.write(GwyTitles[i][0])
                f.write(GwyDataFields[i][0])
                f.write(GwyPalettes[i][0])
            if self.valid_cits == 1:
                for j in range(self.cits_num_blocks):
                    f.write(GwyBricks[j][0])
                    f.write(GwyBrickTitles[j][0])
            elif self.valid_spec == 1 and self.valid_cits == 0:
                for stsblocknum in range(self.spec_num_blocks):
                    f.write(GwySpectra[stsblocknum][0])

    # def get_all_orig_buffers(self):
    #     """The returned buffers dictionary is like what was done in Scott's code"""
    #
    #     i = self.bin_data.find(b"IMG_BUF ")
    #     if i == -1:
    #         raise Exception("Missing data")
    #     buffer_starts = [
    #         i + 8 + 2 * b * self.dimensions[0] * self.dimensions[1]
    #         for b in range(self.numbufs)
    #     ]
    #     buffers = {}
    #     for idx, elem in enumerate(list(range(self.numbufs))):
    #         img_arr = np.zeros((self.ynum, self.xnum))
    #         if self.BufDataTypes[elem] == 0:
    #             for x in range(self.xnum):
    #                 for y in range(self.ynum):
    #                     img_arr[y, x] = struct.unpack(
    #                         "<h",
    #                         self.bin_data[buffer_starts[idx] : buffer_starts[idx] + 2],
    #                     )[0]
    #                     buffer_starts[idx] = buffer_starts[idx] + 2
    #         img_arr = np.reshape(img_arr, (self.ynum, self.xnum))
    #         img_arr = np.flip(img_arr, 0)
    #         buffers[elem] = img_arr
    #     return buffers

    def get_all_orig_buffers(self):
        """The returned buffers dictionary is like what was done in Scott's code"""

        i = self.bin_data.find(b"IMG_BUF ")
        if i == -1:
            raise Exception("Missing data")
        buffer_starts = [
            i + 8 + 2 * b * self.dimensions[0] * self.dimensions[1]
            for b in range(self.numbufs)
        ]
        buffers = {}
        for idx, elem in enumerate(list(range(self.numbufs))):
            if self.BufDataTypes[elem] == 0:
                # for x in range(self.xnum):
                #     for y in range(self.ynum):
                #         img_arr[y, x] = struct.unpack(
                #             "<h",
                #             self.bin_data[buffer_starts[idx] : buffer_starts[idx] + 2],
                #         )[0]
                #         buffer_starts[idx] = buffer_starts[idx] + 2
                points = self.dimensions[0] * self.dimensions[1]
                image_array = struct.unpack(
                    "<" + "h" * points,
                    self.bin_data[buffer_starts[idx] : buffer_starts[idx] + points * 2],
                )
                image_array = np.reshape(image_array, (self.ynum, self.xnum))
                image_array = np.flip(image_array, 0)
            buffers[elem + 1] = image_array
        return buffers

    def get_buffers(self, bufferList):
        """bufferList is a list of number(s) that indicates the buffer(s) to be extracted. return a dictionary
            with buffer number being the key and the corresponding image array being the value
        """
        for i in bufferList:
            if i > self.numbufs:
                raise Exception(
                    "One or more buffer number(s) are not within the total number of buffers"
                )
        i = self.bin_data.find(b"IMG_BUF ")
        if i == -1:
            raise Exception("Missing data")
        buffer_starts = [
            i + 8 + 2 * (b - 1) * self.dimensions[0] * self.dimensions[1]
            for b in bufferList
        ]
        # The 2 is because each int is 2-byte (2 indices)
        buffers = {}
        for idx, elem in enumerate(bufferList):
            if self.BufDataTypes[elem - 1] == 0:
                points = self.dimensions[0] * self.dimensions[1]
                image_array = struct.unpack(
                    "<" + "h" * points,
                    self.bin_data[buffer_starts[idx] : buffer_starts[idx] + points * 2],
                )
                image_array = np.reshape(image_array, (self.ynum, self.xnum))
                image_array = np.flip(image_array, 0)
                buffers[bufferList[idx]] = image_array
        return buffers

    def get_height_buffers(self, bufferList):
        buffers = self.get_buffers(bufferList)
        height_buffers = {}
        for key in buffers:
            # adjust data's multiplication factor
            mulfac = 1
            addfac = 0
            if self.BufDataTypes[key-1] == 0:
                if self.buf_scan_modes[key-1] == 0:
                    mulfac = (
                        self.dsp_atod_max_v / self.dtoa_max_value
                    ) / self.cur_ad_gain ** 2
                    if not self.bias_to_probe and self.atodsign:
                        mulfac *= -1

                elif self.buf_scan_modes[key-1] == 1 or self.buf_scan_modes[key-1] == 2:
                    mulfac = (self.dsp_dtoa_max_v / self.dtoa_max_value) * (
                        self.jj_z_gain * self.hv_gain * self.zcal * self.zver / 10
                    )

                elif self.buf_scan_modes[key-1] == 3:
                    mulfac = (
                        self.dsp_atod_max_v / self.dtoa_max_value
                    ) / self.cur_ad_gain ** 2

                elif self.buf_scan_modes[key-1] == 4 or self.buf_scan_modes[key-1] == 5:
                    mulfac = self.dsp_atod_max_v / self.dtoa_max_value
                    if not self.bias_to_probe and self.atodsign:
                        mulfac *= -1

            
            height_buffers[key] = buffers[key] * mulfac * 1e-10 + addfac
        return height_buffers

    def get_height_buffers_average(self, bufferList):
        height_buffers = self.get_height_buffers(bufferList)
        avg = {}
        for key in height_buffers:
            avg[key] = height_buffers[key].mean()
        return avg


def parseChunk(bindat, header, format, skip_steps=0):
    # if header == "CITS_2" or header == "CITS_4":
    #     pass
    i = bindat.find(header)
    if i == -1:
        raise Exception("missing data")
    return struct.unpack(
        format,
        bindat[i + 8 + skip_steps : i + 8 + skip_steps + struct.calcsize(format)],
    )  # There are 8 spaces from the header to the actual data
