from .simulation import SimulationPolarizabilityModel
from .survey import (
    Survey, MagneticControlledSource, MagneticUniformSource,
    MagneticFluxDensityReceiver, component_dictionary, inv_component_dictionary
)

from .simulation_1TX3RX import SimulationPolarizabilityModel
from .survey_1TX3RX import (
    Survey, MagneticControlledSource, MagneticUniformSource,
    MagneticFluxDensityReceiver, component_dictionary, inv_component_dictionary
)
from .inversion import Inversion
