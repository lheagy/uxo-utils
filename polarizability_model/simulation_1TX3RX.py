import numpy as np
from scipy.constants import mu_0
import warnings

from SimPEG import maps
from SimPEG.simulation import BaseSimulation
from pymatsolver import Pardiso as Solver

from .survey_1TX3RX import component_dictionary

class SimulationPolarizabilityModel(BaseSimulation):

    def __init__(self, locations, survey, mapping=None):
        if locations.shape[1] != 3:
            raise ValueError(
                f"The location must be (npoints, 3), but the input shape is {locations.shape}"
            )
        self._locations = locations
        self._nC = locations.shape[0]

        self._survey = survey

        if mapping is None:
            mapping = maps.IdentityMap(nP=self._nC*3)
        self._mapping = mapping
        self._solver = Solver

    @property
    def locations(self):
        return self._locations

    @property
    def nC(self):
        return self._nC

    @property
    def mapping(self):
        return self._mapping

    @property
    def survey(self):
        return self._survey

    @property
    def solver(self):
        return self._solver

    @property
    def G(self):
        if getattr(self, "_G", None) is None:
            locations = self.locations

            G = []
            for src in self.survey.source_list:
                # NOTE: Hard-coded for only one type of receivers
                receivers = src.receiver_list[0]
                vector_distance = (
                    receivers.locations[None, :, :] - locations[:, None, :]
                )
                distance = np.linalg.norm(vector_distance, axis=2)
                rhat = vector_distance / distance[:, :, None]

                receiver_componentset = np.reshape(receivers.components,(-1,3))  # modified here

                for k in range(len(receiver_componentset)):
                    receiver_components = np.zeros((0, len(receiver_componentset[k])))

                    for comp in receiver_componentset[k]:
                        if comp in receiver_componentset[k]:
                            e = np.zeros(len(receiver_componentset[k]))  #len(receivers.components)  #3
                            e[component_dictionary[comp]-k*3] = 1  # modified here
                            receiver_components = np.vstack(
                                [receiver_components, e])

                    G.append(
                        receivers.area *
                        1 / (4*np.pi) * np.vstack([(
                            receiver_components @
                            np.hstack([
                                1/distance[j, i]**3 * (3*np.outer(rhat[j, i, :], rhat[j, i, :]) - np.eye(3))
                                for j in range(locations.shape[0])
                            ])
                        ) for i in range(rhat.shape[1])
                        ])
                    )
                self._G = G
        return self._G

    def magnetization(self, m, adjoint=False):
        polarizabilities = self.mapping * m  # modified here
        return [np.array([src.eval(self.locations) * polarizabilities]*3) for src in self.survey.source_list]

    def fields(self, m):
        # self.model = m
        G_list = self.G
        #m_list = self.magnetization(m)
        m_list = []   # modified here
        for j in range(15):
            for i in range(3):
                m_list.append(self.magnetization(m)[j][i, :])
        # modified here

        if len(G_list) != len(m_list):
            raise Exception(
                "Something is wrong, the length of the system matrix list and magnetizations don't match")

        return [G @ m for G, m in zip(G_list, m_list)]

    def dpred(self, m, f=None):
        # self.model = m
        return np.hstack(self.fields(m))

    def Jvec(self, m, v, f=None):
        # self.model = m
        G_list = self.G
        src_v = self.magnetization(v)
        return np.hstack([G @ vec for G, vec in zip(G_list, src_v)])

    def Jtvec(self, m, v, f=None):
        # self.model = m
        G_list = self.G
        v_list = [v[i*src.nD:(i+1)*src.nD]
                  for i, src in enumerate(self.survey.source_list)]
        return self.mapping * sum([
            src.eval(self.locations) * (G.T @ vi) for G, vi, src in zip(G_list, v_list, self.survey.source_list)
        ])
