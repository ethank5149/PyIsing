"""Physical constants used throughout PyIsing."""

import numpy as np

#: Boltzmann constant (natural units, dimensionless = 1)
KB: float = 1

#: Exact critical temperature of the 2D Ising model (Onsager, 1944)
T_CRIT: float = 2 / np.log(1 + np.sqrt(2))

#: Critical inverse temperature β_c = 1 / (k_B T_c)
BETA_CRIT: float = 1 / (KB * T_CRIT)
