import numpy as np

from .data import load_ordnance_dict, Survey
from BTInvert import (
    sensorCoords2RxCoords, preCalcLoopCorners, FModParam, Model, forwardWithQ
)

def create_survey(
    sensorinfo,
    times,
    line_length=3,
    along_line_spacing=0.1,
    line_spacing=0.5,
    n_lines=1,
    starting_point=np.r_[0, 0],
    z=0.28,
    pitch=0,
    roll=0,
    yaw=0,
):

    ntx = len(sensorinfo.transmitters)
    dy = along_line_spacing / ntx
    n_along_line = int(np.ceil(line_length/dy))
    ncycles = int(n_along_line/ntx)

    x_line = starting_point[1] * np.ones(n_along_line)
    y_line = np.linspace(starting_point[0], line_length+starting_point[0]-dy, n_along_line)


    y = [y_line, np.flipud(y_line + dy)] * int(np.ceil(n_lines/2))
    y = np.hstack(y[:n_lines])

    x = np.kron(
        np.linspace(starting_point[1], n_lines*line_spacing, n_lines), np.ones(n_along_line)
    )
    lines = np.kron(
        np.arange(n_lines), np.ones(n_along_line, dtype=int)
    )

    # todo rotate x, y if yaw != 0

    z = z * np.ones(n_along_line*n_lines)
    xyz = np.vstack([x, y, z]).T

    pitch = pitch*np.ones(n_along_line*n_lines)
    roll = roll*np.ones(n_along_line*n_lines)

    yaw = [
        yaw * np.ones(n_along_line), (yaw + np.pi) * np.ones(n_along_line)
    ] * int(np.ceil(n_lines/2))
    yaw = np.hstack(yaw[:n_lines])

    txnum = np.kron(np.ones(ncycles*n_lines, dtype=int), np.arange(ntx, dtype=int))

    # Convert sensor location coordinates to Rx locations
    pos, mnum = sensorCoords2RxCoords(
        sensorinfo=sensorinfo,
        x = x,
        y = y,
        z = z,
        pitch = pitch,
        roll = roll,
        yaw = yaw,
        txnum = txnum if ntx > 1 else None
    )

    pitch = np.concatenate([np.tile(x,pos[i].shape[0]) for i, x in enumerate(pitch)])
    roll = np.concatenate([np.tile(x,pos[i].shape[0]) for i, x in enumerate(roll)])
    yaw = np.concatenate([np.tile(x,pos[i].shape[0]) for i, x in enumerate(yaw)])
    lines = np.concatenate([np.tile(x,pos[i].shape[0]) for i, x in enumerate(lines)])

    if len(pos.shape) > 2:
        pos = np.concatenate(pos, axis=0)
    else:
        pos = np.atleast_3d(pos).repeat(3, axis=-1)
        pos = np.concatenate(pos, axis=1).T

    if ntx == 1:
        mnum = np.kron(np.ones(ncycles*n_lines, int), mnum)

    return {
        "xyz": xyz,
        "pos": pos,
        "mnum": mnum,
        "pitch": pitch,
        "roll": roll,
        "yaw": yaw,
        "txnum": txnum,
        "line": lines,
    }



def create_forward_modelling_params(
    sensorinfo, times, mnum, pos, pitch, roll, yaw
):
    Tx_indices_rot, Rx_indices_rot = preCalcLoopCorners(
        sensorinfo=sensorinfo, mnum=mnum, rlist=pos,
        pitch=pitch, roll=roll, yaw=yaw
    )

    return FModParam(sensorinfo, pos, mnum, times, Tx_indices_rot, Rx_indices_rot)

def generate_random_variables(n, bounds, log_scaled=False):
    if log_scaled is True:
        if any(bounds == 0):
            return np.zeros(n)
        bounds = np.log(bounds)
        return np.exp(bounds.min() + (bounds.max() - bounds.min()) * np.random.rand(n))
    return bounds.min() + (bounds.max() - bounds.min()) * np.random.rand(n)

def noise_model(times, amplitude=0.1, slope=-1):
    return amplitude * np.exp(slope * np.log(times))

def simulate_object(L1, L2, L3, st, times, xyz, ypr):
    # run simulation
    mod = Model(xyz=xyz, gba=ypr, l3=L3, l2=L2, l1=L1, times=times)
    V = forwardWithQ(mod, st) # nT/s (some version of db/dt)
    V = V.reshape(-1, st.mnum.max()+1, len(times))
    V = np.swapaxes(V, 0, 1)

    return V



