import h5py
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings

from .imports import data_dir
from BTInvert import SensorInfo
from .parse import proc_group

def load_ordnance_dict(
    directory=data_dir,
    filenames=[
        "ordnance_DoD_UltraTEM_5F_APG.h5",
        "ordnance_DoD_UltraTEM_5F_ISOsmall.h5",
        "ordnance_DoD_UltraTEM_NATA_dyn_F_scale0o86.h5"
    ]
):
    """
    create a dictionary of ordnance object from h5 files
    """
    ord_dict = {}

    for file in filenames:
        ord_file = os.path.join(data_dir, file)
        f = h5py.File(ord_file, 'r')
        for i in f['ordnance']:
            ord_name = str(f[f'ordnance/{i}/Name'][()][0]).split("'")[1]

            L3 = f[f'ordnance/{i}/L1ref'][()].flatten()
            L2 = f[f'ordnance/{i}/L2ref'][()].flatten()
            L1 = f[f'ordnance/{i}/L3ref'][()].flatten()
            size_mm = int(f[f'ordnance/{i}/size_mm'][()].flatten())
            common_name = f[f'ordnance/{i}/h5_Common_Name'][()].flatten()[0]

            if isinstance(common_name, list):
                common_name = common_name[0]

            if ord_name not in ord_dict.keys():
                times = f[f'ordnance/{i}/time'][()].flatten()
                ord_dict[ord_name] = {
                    "L3": [L3],
                    "L2": [L2],
                    "L1": [L2],
                    "size mm": [size_mm],
                    "common name": [common_name],
                    "times": times,
                }
            else:
                for key, val in zip(
                    ["L3", "L2", "L1", "size mm", "common name"],
                    [L3, L2, L1, size_mm, common_name]
                ):
                    ord_dict[ord_name][key].append(val)

    return ord_dict


def load_sensor_info(sensor="UltraTEM"):

    if sensor.lower() == "ultratem":
        filename = os.path.join(
            data_dir, 'config', 'sensor_definitions', 'UltraTEMArrayNA___Default.yaml'
        )
    elif "subtem" in sensor.lower():
        if "rov" in sensor.lower():
            filename = os.path.join(
                data_dir, 'config', 'sensor_definitions', 'SubTEM_ROV___Default.yaml'
            )
        else:
            filename = os.path.join(
                data_dir, 'config', 'sensor_definitions', 'SubTEMSledge___Default.yaml'
            )
    elif sensor.lower() == "ultratema":
        filename = os.path.join(
            data_dir, 'config', 'sensor_definitions', 'UltraTEMA___Default.yaml'
        )


    return SensorInfo.fromYAML(filename)[0]


def load_h5_data(filepath):

    # load into a dict with h5py
    dfile = os.path.join(filepath)
    f = h5py.File(dfile, 'r')
    dictionary = proc_group(f)
    f.close()

    return dictionary

def rotate_survey(local_x, local_y):
    model = LinearRegression(fit_intercept=False)
    model.fit(local_x[:, np.newaxis], local_y)
    slope = model.coef_[0]
    theta = np.arctan(slope) + np.pi/2
    rotated_x = np.cos(theta) * local_x + np.sin(theta) * local_y
    rotated_y = -np.sin(theta) * local_x + np.cos(theta) * local_y
    return theta, slope, rotated_x, rotated_y

def create_survey_from_file(filepath, convert_degrees_to_radians=False):
    dic = load_h5_data(filepath)

    xyz_dict = dic["XYZ"]
    xyz_data = xyz_dict["Data"]

    times = np.array(dic['SensorTimes'].flatten())

    def get_index(key):
        return xyz_dict["Info"][key]["ChannelIndex"].flatten().astype(int)-1

    def get_values(key):
        index = get_index(key)
        return xyz_data[index, :].flatten()

    easting = get_values("Easting")
    northing = get_values("Northing")

    yaw = get_values("Yaw")
    pitch = get_values("Pitch")
    roll = get_values("Roll")

    if convert_degrees_to_radians:
        yaw = yaw*np.pi/180
        pitch = pitch*np.pi/180
        roll = roll*np.pi/180
    else:
        if not all(np.abs(yaw) < 2*np.pi):
            warnings.warn("There are yaw values above 2 pi. You might need to convert to radians `convert_degrees_to_radians=True`")

    mnum = get_values("MeasNum").astype(int) - 1 # index from zero in python
    line = get_values("Line").astype(int)
    rx_num = get_values("RxNum").astype(int) - 1
    tx_num = get_values("TxNum").astype(int) - 1
    rx_comp = get_values("RxCNum").astype(int) - 1
    data = xyz_data[get_index("Data"), :].T
    return Survey(times, easting, northing, pitch, roll, yaw, mnum, data, line, rx_num, tx_num, rx_comp)


class Survey:
    def __init__(
        self,
        times,
        easting,
        northing,
        pitch,
        roll,
        yaw,
        mnum,
        data=None,
        line=None,
        rx_num=None,
        tx_num=None,
        rx_comp=None,
    ):
        self._data = data
        self._time = times
        self._easting = easting
        self._northing = northing
        self._pitch = pitch
        self._roll = roll
        self._yaw = yaw
        self._mnum = mnum
        self._line = line
        self._rx_num = rx_num
        self._tx_num = tx_num
        self._rx_comp = rx_comp


    # Properties directly from data file
    @property
    def times(self):
        return self._times

    @property
    def easting(self):
        return self._easting

    @property
    def northing(self):
        return self._northing

    @property
    def yaw(self):
        return self._yaw

    @property
    def pitch(self):
        return self._pitch

    @property
    def roll(self):
        return self._roll

    @property
    def mnum(self):
        return self._mnum

    @property
    def line(self):
        return self._line

    @property
    def rx_num(self):
        return self._rx_num

    @property
    def tx_num(self):
        return self._tx_num

    @property
    def data(self):
        return self._data

    # Derived properties
    @property
    def x0(self):
        if getattr(self, "_x0", None) is None:
            self._x0 = np.r_[self.easting.mean(), self.northing.mean()]
        return self._x0

    @property
    def local_x(self):
        if getattr(self, "_local_x", None) is None:
            self._local_x = self.easting - self.x0[0]
        return self._local_x

    @property
    def local_y(self):
        if getattr(self, "_local_y", None) is None:
            self._local_y = self.northing - self.x0[1]
        return self._local_y

    def _rotate_survey(self):
        theta, slope, rotated_x, rotated_y = rotate_survey(self.local_x, self.local_y)
        self._theta = theta
        self._slope = slope
        self._rotated_x = rotated_x
        self._rotated_y = rotated_y
        self._rotated_yaw = self.yaw + theta

    @property
    def slope(self):
        if getattr(self, "_slope", None) is None:
            self._rotate_survey()
        return self._slope

    @property
    def rotated_x(self):
        if getattr(self, "_rotated_x", None) is None:
            self._rotate_survey()
        return self._rotated_x

    @property
    def rotated_y(self):
        if getattr(self, "_rotated_y", None) is None:
            self._rotate_survey()
        return self._rotated_y

    @property
    def rotated_yaw(self):
        if getattr(self, "_rotated_y", None) is None:
            self._rotate_survey()
        return self._rotated_yaw

    @property
    def unique_lines(self):
        if getattr(self, "_unique_lines", None) is None:
            self._unique_lines = np.unique(self.line)
        return self._unique_lines
