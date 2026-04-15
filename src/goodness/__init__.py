from .base import Goodness
from .squared_sum import SquaredSumGoodness
from .unsquared_sum import UnsquaredSumGoodness
from .spatial import SpatialBlockGoodness

__all__ = [
    "Goodness",
    "SquaredSumGoodness",
    "UnsquaredSumGoodness",
    "SpatialBlockGoodness",
]
