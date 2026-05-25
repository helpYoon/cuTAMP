# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Per-conf gradient masking for the particle optimizer.

Adam's bias-corrected first step takes magnitude ``lr`` along every
non-zero-gradient DOF, regardless of gradient size. Once the IK initialization
already produces sub-tolerance kinematic residuals, that uniform-magnitude step
overshoots the optimum — every active DOF moves by ``lr`` and the EE pose
displaces well past the satisfaction tolerance.

The fix: hard-zero the gradient on DOFs that should not move during a given
operator. This is a *physical* lock, not a soft penalty: even with Adam's
per-parameter normalization, a zero gradient produces a zero step.

Per-operator locking policy for T1 (single 21-DOF planner with planar base):

* Arm action / arm motion (Pick, Place, Push, MoveFree, MoveHolding,
  Retract): lock base (slice 0:3) + inactive arm (the 7 DOFs of the
  non-active arm). Active arm + lift/torso remain free.
* Navigate (MoveBaseTo): lock lift/torso + both arms (slice 3:21). Only
  the 3 base DOFs remain free.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from cutamp.task_planning import PlanSkeleton


def get_conf_locked_dofs(
    plan_skeleton: PlanSkeleton, conf_name: str
) -> List[int]:
    """Return the DOF indices that should be gradient-locked for ``conf_name``.

    Locking is determined by the metadata of the first operator in the plan
    skeleton that references the conf as one of its parameters.
    """
    from cutamp.robots.t1 import (
        BASE_DOF,
        CUROBO_DOF,
        LEFT_ARM_INDICES,
        RIGHT_ARM_INDICES,
    )

    base = list(range(0, BASE_DOF))
    left_arm = list(range(LEFT_ARM_INDICES.start, LEFT_ARM_INDICES.stop))
    right_arm = list(range(RIGHT_ARM_INDICES.start, RIGHT_ARM_INDICES.stop))

    for ground_op in plan_skeleton:
        if conf_name not in ground_op.values:
            continue
        meta = ground_op.operator.metadata
        if meta.action_type == "navigate":
            return list(range(BASE_DOF, CUROBO_DOF))
        if meta.arm == "left":
            return base + right_arm
        if meta.arm == "right":
            return base + left_arm
        return []
    return []


def build_grad_masks(
    plan_skeleton: PlanSkeleton,
    particles: Dict[str, torch.Tensor],
    param_to_type: Dict[str, str],
    conf_type: str,
) -> Dict[int, torch.Tensor]:
    """Build a tensor mask per Conf parameter; index = ``id(tensor)``.

    The mask has the same shape as the parameter and contains 1.0 at free
    DOFs and 0.0 at locked DOFs. Apply by ``param.grad.mul_(mask)`` after
    ``loss.backward()``.
    """
    masks: Dict[int, torch.Tensor] = {}
    for param_name, tensor in particles.items():
        if param_to_type.get(param_name) != conf_type:
            continue
        locked = get_conf_locked_dofs(plan_skeleton, param_name)
        if not locked:
            continue
        dof = tensor.shape[-1]
        mask = torch.ones(dof, device=tensor.device, dtype=tensor.dtype)
        mask[locked] = 0.0
        masks[id(tensor)] = mask
    return masks


def apply_grad_masks(
    param_groups: list, masks: Dict[int, torch.Tensor]
) -> None:
    """Zero out grads on locked DOFs in-place."""
    if not masks:
        return
    for group in param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            mask = masks.get(id(p))
            if mask is not None:
                p.grad.mul_(mask)
