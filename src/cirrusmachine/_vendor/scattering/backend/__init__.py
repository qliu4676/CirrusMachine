"""PyTorch backend for CPU and CUDA, independent of external package settings."""

from .backend_common import *
from .backend_torch import *

BACKEND = "torch"
