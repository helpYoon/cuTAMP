"""Tests for joint-limit margin vectors and costs.

Spec: docs/superpowers/specs/2026-06-11-joint-limit-margin-design.md
"""
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest
import torch

from cutamp.robots.t1 import (
    BASE_INDICES,
    CUROBO_DOF,
    JOINT_NAMES_FULL,
    T1_ASSETS_DIR,
    T1_JOINT_LIMIT_MARGIN,
    T1_LIMIT_MARGIN_LOWER,
    T1_LIMIT_MARGIN_UPPER,
    t1_home,
)


class TestMarginVectors:
    def test_lengths(self):
        assert len(T1_LIMIT_MARGIN_LOWER) == CUROBO_DOF == 21
        assert len(T1_LIMIT_MARGIN_UPPER) == CUROBO_DOF == 21

    def test_base_unmargined(self):
        # Virtual planar-base DOFs are locked in IK/planner — no margins.
        assert all(m == 0.0 for m in T1_LIMIT_MARGIN_LOWER[BASE_INDICES])
        assert all(m == 0.0 for m in T1_LIMIT_MARGIN_UPPER[BASE_INDICES])

    def test_home_side_bounds_unmargined(self):
        # The standing home posture sits exactly ON these three bounds
        # (straight leg): margining them would penalize standing still.
        names = list(JOINT_NAMES_FULL)
        assert T1_LIMIT_MARGIN_UPPER[names.index("ankle_pitch")] == 0.0
        assert T1_LIMIT_MARGIN_LOWER[names.index("knee_pitch")] == 0.0
        assert T1_LIMIT_MARGIN_UPPER[names.index("Torso_Pitch")] == 0.0

    def test_protected_sides_use_master_margin(self):
        names = list(JOINT_NAMES_FULL)
        # Far-side bounds of the leg/torso-pitch joints are protected.
        assert T1_LIMIT_MARGIN_LOWER[names.index("ankle_pitch")] == T1_JOINT_LIMIT_MARGIN
        assert T1_LIMIT_MARGIN_UPPER[names.index("knee_pitch")] == T1_JOINT_LIMIT_MARGIN
        assert T1_LIMIT_MARGIN_LOWER[names.index("Torso_Pitch")] == T1_JOINT_LIMIT_MARGIN
        # Waist_Yaw (idx 6) + all 14 arm joints (7..20): both sides protected.
        for i in range(6, 21):
            assert T1_LIMIT_MARGIN_LOWER[i] == T1_JOINT_LIMIT_MARGIN, JOINT_NAMES_FULL[i]
            assert T1_LIMIT_MARGIN_UPPER[i] == T1_JOINT_LIMIT_MARGIN, JOINT_NAMES_FULL[i]


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _t1_position_limits() -> torch.Tensor:
    """[2, 21] limits in JOINT_NAMES_FULL order (row 0 = lower).

    URDF-parsed for the 18 body joints; virtual base DOFs use the
    t1_planar_base.yml extra_links values.
    """
    base_limits = {
        "base_j_x": (-2.0, 2.0),
        "base_j_y": (-2.0, 2.0),
        "base_j_yaw": (-3.14159, 3.14159),
    }
    tree = ET.parse(T1_ASSETS_DIR / "t1_simplified.urdf")
    urdf_limits = {}
    for joint in tree.getroot().iter("joint"):
        limit = joint.find("limit")
        if joint.get("type") == "revolute" and limit is not None:
            urdf_limits[joint.get("name")] = (
                float(limit.get("lower")), float(limit.get("upper")),
            )
    rows = [base_limits.get(n) or urdf_limits[n] for n in JOINT_NAMES_FULL]
    return torch.tensor(rows, dtype=torch.float32).T.contiguous()  # [2, 21]


class TestShrunkenBoundsPenalty:
    def test_zero_strictly_inside(self):
        from cutamp.joint_limit_cost import shrunken_bounds_penalty
        lo = torch.tensor([-1.0, 0.0])
        hi = torch.tensor([1.0, 2.0])
        q = torch.tensor([[0.0, 1.0], [-0.99, 1.99]])
        assert shrunken_bounds_penalty(q, lo, hi).sum() == 0.0

    def test_quadratic_in_penetration(self):
        from cutamp.joint_limit_cost import shrunken_bounds_penalty
        lo = torch.tensor([0.0]); hi = torch.tensor([1.0])
        # 0.2 below the shrunk lower bound -> 0.04; doubling depth quadruples.
        p1 = shrunken_bounds_penalty(torch.tensor([[-0.2]]), lo, hi)
        p2 = shrunken_bounds_penalty(torch.tensor([[-0.4]]), lo, hi)
        assert torch.allclose(p1, torch.tensor([[0.04]]))
        assert torch.allclose(p2, 4.0 * p1)

    def test_both_sides_penalized(self):
        from cutamp.joint_limit_cost import shrunken_bounds_penalty
        lo = torch.tensor([0.0]); hi = torch.tensor([1.0])
        below = shrunken_bounds_penalty(torch.tensor([[-0.1]]), lo, hi)
        above = shrunken_bounds_penalty(torch.tensor([[1.1]]), lo, hi)
        assert below.item() > 0.0 and above.item() > 0.0
        assert torch.allclose(below, above)


class TestSoftCostHelper:
    def _margins(self):
        m_lo = torch.tensor(T1_LIMIT_MARGIN_LOWER, dtype=torch.float32)
        m_hi = torch.tensor(T1_LIMIT_MARGIN_UPPER, dtype=torch.float32)
        return m_lo, m_hi

    def test_home_posture_costs_exactly_zero(self):
        # The whole point of per-side margins: standing at home (which sits ON
        # three bounds) must cost nothing.
        from cutamp.joint_limit_cost import joint_limit_margin_soft_cost
        limits = _t1_position_limits()
        m_lo, m_hi = self._margins()
        home = torch.tensor(t1_home, dtype=torch.float32).expand(4, -1)
        particles = {"q_retract": home.clone(), "left_q1": home.clone()}
        cost = joint_limit_margin_soft_cost(
            particles, limits, m_lo, m_hi, num_particles=4, device=torch.device("cpu"),
        )
        assert cost.shape == (4,)
        assert torch.all(cost == 0.0)

    def test_near_limit_conf_penalized(self):
        # Torso_Pitch (idx 5) at -1.68: 0.02 rad from the -1.7 lower limit,
        # 0.08 inside the 0.1 band -> penalty 0.08^2 = 6.4e-3.
        from cutamp.joint_limit_cost import joint_limit_margin_soft_cost
        limits = _t1_position_limits()
        m_lo, m_hi = self._margins()
        q = torch.tensor(t1_home, dtype=torch.float32).expand(2, -1).clone()
        q[:, 5] = -1.68
        cost = joint_limit_margin_soft_cost(
            {"q_pick": q}, limits, m_lo, m_hi, num_particles=2, device=torch.device("cpu"),
        )
        assert torch.allclose(cost, torch.full((2,), 0.08 ** 2), atol=1e-6)

    def test_q0_excluded(self):
        # Violating config under every initial-state name + clean home under
        # q_pick: total must be exactly 0 (exclusion works inside the stacked
        # path, not just via the empty-dict fallback). Initial states must
        # never acquire margin gradients — Phase-2 LBFGS re-leafs every
        # cloned particle with requires_grad_(True).
        from cutamp.joint_limit_cost import joint_limit_margin_soft_cost
        limits = _t1_position_limits()
        m_lo, m_hi = self._margins()
        bad = torch.tensor(t1_home, dtype=torch.float32).expand(2, -1).clone()
        bad[:, 5] = -1.68  # Torso_Pitch inside the margin band
        home = torch.tensor(t1_home, dtype=torch.float32).expand(2, -1).clone()
        cost = joint_limit_margin_soft_cost(
            {"q0": bad, "left_q0": bad.clone(), "right_q0": bad.clone(), "q_pick": home},
            limits, m_lo, m_hi, num_particles=2, device=torch.device("cpu"),
        )
        assert torch.all(cost == 0.0)

    def test_no_confs_returns_zeros(self):
        from cutamp.joint_limit_cost import joint_limit_margin_soft_cost
        limits = _t1_position_limits()
        m_lo, m_hi = self._margins()
        cost = joint_limit_margin_soft_cost(
            {"pose1": torch.zeros(3, 4)}, limits, m_lo, m_hi,
            num_particles=3, device=torch.device("cpu"),
        )
        assert cost.shape == (3,) and torch.all(cost == 0.0)

    def test_gradients_restoring_and_q0_grad_free(self):
        # (a) Gradient through the hinge² points back toward the interior;
        # (b) excluded initial-state confs get NO gradient even when they
        #     participate in the dict — the safety property the exclusion
        #     exists for.
        from cutamp.joint_limit_cost import joint_limit_margin_soft_cost
        limits = _t1_position_limits()
        m_lo, m_hi = self._margins()
        bad = torch.tensor(t1_home, dtype=torch.float32).expand(2, -1).clone()
        bad[:, 5] = -1.68  # 0.08 into the lower-side band
        q_pick = bad.clone().requires_grad_(True)
        left_q0 = bad.clone().requires_grad_(True)
        cost = joint_limit_margin_soft_cost(
            {"q_pick": q_pick, "left_q0": left_q0},
            limits, m_lo, m_hi, num_particles=2, device=torch.device("cpu"),
        )
        cost.sum().backward()
        assert left_q0.grad is None or torch.all(left_q0.grad == 0.0)
        # d/dq of relu(shrunk_lo - q)^2 is negative of 2*relu(...): pushing q
        # UP (toward interior) decreases cost -> grad at the violated DOF < 0.
        assert q_pick.grad is not None
        assert q_pick.grad[0, 5] < 0.0
        assert torch.all(q_pick.grad[:, :5] == 0.0)  # untouched DOFs grad-free


class TestTrajoptCost:
    @needs_cuda
    def test_forward_shape_zero_inside_weight_applied(self):
        from curobo.types import DeviceCfg
        from cutamp.joint_limit_cost import JointLimitMarginCost, JointLimitMarginCostCfg

        dc = DeviceCfg()
        lo = torch.tensor([-1.0, -1.0, -1.0])
        hi = torch.tensor([1.0, 1.0, 1.0])
        cfg = JointLimitMarginCostCfg(
            weight=[2.0], device_cfg=dc, shrunk_lower=lo, shrunk_upper=hi,
        )
        cost_fn = JointLimitMarginCost(cfg)
        b, h, dof = 2, 5, 3
        q = torch.zeros(b, h, dof, device=dc.device, dtype=dc.dtype)
        q[1, 3, 0] = 1.5  # 0.5 past the shrunk upper bound
        state = SimpleNamespace(joint_state=SimpleNamespace(position=q))
        out = cost_fn.forward(state)
        assert out.shape == (b, h, 1)
        assert out[0].sum() == 0.0                      # inside everywhere
        assert torch.allclose(out[1, 3, 0], torch.tensor(2.0 * 0.25, device=dc.device, dtype=dc.dtype))


class TestSoftCostWiring:
    def test_registered_default_on_and_weighted(self):
        from cutamp.config import SUPPORTED_SOFT_COSTS, TAMPConfiguration, validate_tamp_config
        from cutamp.scripts.utils import default_constraint_to_mult

        assert "joint_limit_margin" in SUPPORTED_SOFT_COSTS
        cfg = TAMPConfiguration()
        assert set(cfg.soft_cost) == {"place_close_to_base", "joint_limit_margin"}  # default-on, nothing dropped
        validate_tamp_config(cfg)                              # whitelist accepts it
        # Unregistered names pass through UNWEIGHTED in CostReducer — the
        # weight entry is mandatory.
        assert default_constraint_to_mult["soft"]["joint_limit_margin"] == 1e3


class TestCheckPlanMargins:
    def test_urdf_parser_and_joint_order(self):
        from cutamp.scripts.check_plan_margins import STORED_JOINTS, parse_urdf_limits

        limits = parse_urdf_limits(T1_ASSETS_DIR / "t1_simplified.urdf")
        assert limits["ankle_pitch"] == (-0.87, 0.0)
        assert limits["knee_pitch"] == (0.0, 2.34)
        assert limits["Torso_Pitch"] == (-1.7, 0.0)
        assert limits["Waist_Yaw"] == (-1.47, 1.47)
        # Plan schema v3 stores the 18 non-base joints, in cspace order.
        assert STORED_JOINTS == JOINT_NAMES_FULL[3:]
