from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV


def fit_calibrator(
    fitted_model: Any,
    x_val,
    y_val: np.ndarray,
    method: str = "isotonic",
) -> Any:
    try:
        calibrator = CalibratedClassifierCV(fitted_model, cv="prefit", method=method)
        calibrator.fit(x_val, y_val)
        return calibrator
    except Exception:
        # Fallback for sklearn versions/environments where prefit behavior differs.
        calibrator = CalibratedClassifierCV(fitted_model, cv=3, method=method)
        calibrator.fit(x_val, y_val)
        return calibrator
