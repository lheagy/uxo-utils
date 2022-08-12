import numpy as np

from BTSensor import (    #from BTInvert
    SensorInfo, Measurement, TxRxLoop
)


class Custom1TX3RXSensorInfo(SensorInfo):

    @classmethod
    def coincident_system(cls, z_tx=0, tx_width=1., orientation="z", rx_width=0.25):

        def make_square(center, length, orientation):
            half_length = length/2
            npoints = 4
            dim_1 = half_length * np.r_[-1, 1, 1, -1]
            dim_2 = half_length * np.r_[-1, -1, 1, 1]

            if orientation.lower() == "x":
                loop = np.vstack([
                    np.zeros(npoints) + center[0],
                    dim_1 + center[1],
                    dim_2 + center[2]
                ])
            elif orientation.lower() == "y":
                loop = np.vstack([
                    dim_2 + center[0],
                    np.zeros(npoints) + center[1],
                    dim_1 + center[2]
                ])
            elif orientation.lower() == "z":
                loop = np.vstack([
                    dim_1 + center[0],
                    dim_2 + center[1],
                    np.zeros(npoints) + center[2]
                ])
            return loop.T

        ntx = 1
        system_center = np.r_[0, 0, z_tx]
        tx_shape = make_square(system_center, tx_width, orientation)

        TxLoops = np.ndarray(shape=(ntx), dtype=TxRxLoop)
        TxLoops[0] = TxRxLoop(
            name = f"Tx {orientation.upper()}",
            shape = tx_shape,
            center = system_center,
            gain = 1.0,
            # component = 0 if orientation.lower() == "x" else (1 if orientation.lower() == "y" else 2)
        )
        # -------- Receivers ------------
        # 9 receivers
        nrx = 9

        rx_0RXx_shape = make_square(system_center-np.r_[1.5*rx_width, 0, 0], rx_width, "x")
        rx_0RXy_shape = make_square(system_center-np.r_[1.5*rx_width, 0, 0], rx_width, "y")
        rx_0RXz_shape = make_square(system_center-np.r_[1.5*rx_width, 0, 0], rx_width, "z")

        rx_1RXx_shape = make_square(system_center, rx_width, "x")
        rx_1RXy_shape = make_square(system_center, rx_width, "y")
        rx_1RXz_shape = make_square(system_center, rx_width, "z")

        rx_2RXx_shape = make_square(system_center+np.r_[1.5*rx_width, 0, 0], rx_width, "x")
        rx_2RXy_shape = make_square(system_center+np.r_[1.5*rx_width, 0, 0], rx_width, "y")
        rx_2RXz_shape = make_square(system_center+np.r_[1.5*rx_width, 0, 0], rx_width, "z")


        RxLoops = np.ndarray(shape = (9), dtype=TxRxLoop)

        RxLoops[0] = TxRxLoop(name = "0",
                                shape = rx_0RXx_shape,
                                center = system_center-1.5*rx_width,
                                gain = 1.0,
                                component = 0)

        RxLoops[1] = TxRxLoop(name = "1",
                                shape = rx_0RXy_shape,
                                center = system_center-1.5*rx_width,
                                gain = 1.0,
                                component = 1)

        RxLoops[2] = TxRxLoop(name = "2",
                                shape = rx_0RXz_shape,
                                center = system_center-1.5*rx_width,
                                gain = 1.0,
                                component = 2)

        RxLoops[3] = TxRxLoop(name = "3",
                                shape = rx_1RXx_shape,
                                center = system_center,
                                gain = 1.0,
                                component = 3)

        RxLoops[4] = TxRxLoop(name = "4",
                                shape = rx_1RXy_shape,
                                center = system_center,
                                gain = 1.0,
                                component = 4)

        RxLoops[5] = TxRxLoop(name = "5",
                                shape = rx_1RXz_shape,
                                center = system_center,
                                gain = 1.0,
                                component = 5)

        RxLoops[6] = TxRxLoop(name = "6",
                                shape = rx_2RXx_shape,
                                center = system_center+1.5*rx_width,
                                gain = 1.0,
                                component = 6)

        RxLoops[7] = TxRxLoop(name = "7",
                                shape = rx_2RXy_shape,
                                center = system_center+1.5*rx_width,
                                gain = 1.0,
                                component = 7)

        RxLoops[8] = TxRxLoop(name = "8",
                                shape = rx_2RXz_shape,
                                center = system_center+1.5*rx_width,
                                gain = 1.0,
                                component = 8)





        # put it all together
        mnum = ntx*nrx

        Measurements = np.ndarray(shape=(mnum), dtype=Measurement)

        for i in range(nrx):
            auxrx = np.zeros((1,2),dtype = int)
            auxrx[0,:] = [i,1]
            Measurements[i] = Measurement(
                name = "%d"%(i),
                transmitters = np.array([0]),
                receivers = auxrx
            )

        return SensorInfo(name = "coincident_system",
                          transmitters = TxLoops,
                          receivers = RxLoops,
                          measurements = Measurements)
