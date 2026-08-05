"""Flag plumbing for the standard-cuTAMP 2x2 ablation baseline."""
import pytest

from cutamp.config import TAMPConfiguration, validate_tamp_config


def test_new_flags_default_to_current_behavior():
    c = TAMPConfiguration()
    assert c.enable_com_hard_constraint is True
    assert c.enable_com_init_gate is True
    assert c.force_adam is False
    assert c.soft_in_adam == "auto"
    validate_tamp_config(c)  # defaults must validate


def test_soft_in_adam_rejects_unknown_value():
    c = TAMPConfiguration(soft_in_adam="sometimes")
    with pytest.raises(ValueError, match="soft_in_adam"):
        validate_tamp_config(c)


def test_baseline_cell_config_validates():
    # The "standard cuTAMP" cell from the spec.
    c = TAMPConfiguration(
        enable_com_aware_ik=False,
        enable_com_init_gate=False,
        enable_com_hard_constraint=False,
        enable_com_polygon=False,
        coupled_reik=False,
        force_adam=True,
        soft_in_adam="always",
    )
    validate_tamp_config(c)


import torch

from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


def _tiny_cost_dict(enable_com_hard_constraint: bool):
    from cutamp.config import TAMPConfiguration
    from cutamp.cost_function import CostFunction
    from cutamp.rollout import RolloutFunction
    from cutamp.particle_initialization import ParticleInitializer
    from cutamp.t1_domain import all_t1_operators
    from cutamp.task_planning import task_plan_generator

    world = make_blocks_t1_world()
    config = TAMPConfiguration(
        num_particles=8,
        enable_com_hard_constraint=enable_com_hard_constraint,
    )
    plan_skeleton = next(task_plan_generator(
        world.initial_state, world.goal_state, all_t1_operators,
    ))
    # Mirrors cutamp.algorithm.sample_plan_skeleton: particle init needs
    # autograd for curobo's seed IK solver internals, so no_grad() wraps
    # only the rollout + cost computation, not particle initialization.
    particles = ParticleInitializer(world, config)(plan_skeleton)
    with torch.no_grad():
        rollout = RolloutFunction(plan_skeleton, world, config)(particles)
        return CostFunction(plan_skeleton, world, config)(rollout, particles)


@needs_cuda
def test_com_hard_constraint_flag_controls_cost_dict():
    import gc

    from cutamp.task_planning.constraints import ComPolygon
    on = _tiny_cost_dict(True)
    # Each _tiny_cost_dict() call builds a full TAMPWorld + curobo IK-solver
    # stack (several GiB). Reclaim before building the second one so two
    # heavy stacks don't coexist and OOM a 24 GiB card (see conftest's
    # _free_cuda_between_tests, which does the same thing between tests).
    gc.collect()
    torch.cuda.empty_cache()
    off = _tiny_cost_dict(False)
    assert ComPolygon.type in on
    assert ComPolygon.type not in off
