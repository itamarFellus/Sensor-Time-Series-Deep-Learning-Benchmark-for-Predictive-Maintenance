from __future__ import annotations

import numpy as np


def cap_rul_labels(
    labels: np.ndarray,
    rul_cap: int | float | None,
) -> np.ndarray:
    """Optionally cap RUL labels at a maximum value."""
    labels = labels.astype(np.float32)

    if rul_cap is None:
        return labels

    return np.minimum(labels, rul_cap).astype(np.float32)
