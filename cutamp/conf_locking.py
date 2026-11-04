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

Per-operator locking policy for T1, on the planning model's slices
(``world.model``):

* Arm action / arm motion (Pick, Place, MoveFree, MoveHolding,
  Retract): lock the inactive arm (the 7 DOFs of the non-active arm), and
  the base on a model that locks it (the simplified model's planar base,
  ``model.lock_base``). Active arm + legs/torso remain free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import torch

from cutamp.task_planning import PlanSkeleton

if TYPE_CHECKING:
    from cutamp.robots.t1 import T1Model


def _indices(s: slice) -> List[int]:
    return list(range(s.start, s.stop))


def get_conf_locked_dofs(
    plan_skeleton: PlanSkeleton, conf_name: str, model: "T1Model"
) -> List[int]:
    """Return the DOF indices that should be gradient-locked for ``conf_name``.

    Locking is determined by the metadata of the first operator in the plan
    skeleton that references the conf as one of its parameters. Indices are
    into ``model``'s cspace.
    """
    base = _indices(model.base) if model.lock_base else []
    left_arm = _indices(model.left_arm)
    right_arm = _indices(model.right_arm)

    for ground_op in plan_skeleton:
        if conf_name not in ground_op.values:
            continue
        meta = ground_op.operator.metadata
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
    model: "T1Model",
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
        locked = get_conf_locked_dofs(plan_skeleton, param_name, model)
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
