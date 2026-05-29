"""Unit tests for plan_processor — covers the joint-name alignment guard
and the world-frame schema invariants.

The former quat sign-canonicalization test exercised the FD-based
``_angular_velocity_from_quat`` helper, which was retired when Cartesian
velocity moved to the analytic Jacobian (``J·q̇``). World-frame angular
velocity / acceleration is now covered in ``test_plan_processor_derivatives``.
"""


def test_to_active_cspace_falls_through_on_name_mismatch():
    """Same DOF count but DIFFERENT joint name order must not short-circuit.
    Otherwise the same-shape positional gather mis-indexes every column."""
    import torch
    from cutamp.utils.plan_processor import _to_active_cspace, _build_processing_kinematics

    kin = _build_processing_kinematics()
    active = list(kin.joint_names)
    n = len(active)
    src_names = list(reversed(active))
    tensor = torch.zeros(5, n, device=kin.device_cfg.device)
    tensor[0, 0] = 1.0
    out = _to_active_cspace(tensor, src_names, kin)
    assert not torch.equal(out, tensor), (
        "_to_active_cspace short-circuited despite joint-name mismatch — "
        "downstream gathers would mis-index."
    )
