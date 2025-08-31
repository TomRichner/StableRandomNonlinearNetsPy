from .activation import get_activation
from .params import Params, package_params
from .srnn import SRNN, Trajectory

# For advanced use, users can still access lower-level components
from .dependent import compute_dependent
from .dynamics import make_rhs
from .simulate import solve
from .state import unpack_state, unpack_trajectory


__all__ = [
    "SRNN",
    "Trajectory",
    "Params",
    "package_params",
    "get_activation",
    "compute_dependent",
    "make_rhs",
    "solve",
    "unpack_state",
    "unpack_trajectory",
]

