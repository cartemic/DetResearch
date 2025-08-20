"""
It's the lines-to-deltas algorithm implemented in Rust and compiled for linux x86_64 (Kubuntu 20.04)

https://github.com/cartemic/get_px_deltas_from_lines
"""

from typing import Optional

import numpy as np

# noinspection PyUnresolvedReferences
from .get_px_deltas_from_lines import get_px_deltas_from_lines as get_px_deltas_from_lines_compiled


def _fast_get_deltas(img_path: str, mask_path: Optional[str] = None) -> np.array:
    return get_px_deltas_from_lines_compiled(img_path, mask_path)
