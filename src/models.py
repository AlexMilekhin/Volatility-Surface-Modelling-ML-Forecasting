from dataclasses import dataclass
import numpy as np

@dataclass
class SVIParams:
    "Container for SVI parameters"
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    