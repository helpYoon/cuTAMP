"""The virtual-base chart of a translated stance."""

from copy import deepcopy

import numpy as np
import pytest

from cutamp.robots.t1 import REAL_LEGS, t1_curobo_cfg
from cutamp.supported_posture import set_stance_base_limits


def test_transported_base_chart_intersects_both_leg_workspaces():
    cfg = deepcopy(t1_curobo_cfg(REAL_LEGS))
    feet = np.array([[0.1, 0.15, 0.3], [0.08, -0.14, -0.1]])
    limits = set_stance_base_limits(cfg, feet)
    shifted = set_stance_base_limits(cfg, feet + [0.5, -0.2, 0])
    np.testing.assert_allclose(shifted, limits + [[0.5], [-0.2]], atol=1e-9)
    with pytest.raises(ValueError, match="shared virtual-base"):
        set_stance_base_limits(cfg, np.array([[0, 3, 0], [0, -3, 0]]))
