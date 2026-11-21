import torch
import pytest


def test_particle_adam_updates_coupled_commands_and_placement_without_row_mixing():
    from cutamp.step_adam import optimize_particles

    foot = torch.zeros(2, 1)
    place = torch.zeros_like(foot)
    grasp = torch.tensor([[0.35], [-0.23]])

    def evaluate(values):
        u, p = values["foot"], values["place"]
        pick = (u - grasp).square().flatten()
        transfer = (p - u - grasp).square().flatten()
        return (u.abs() - 0.22).clamp_min(0).flatten(), pick + transfer

    best, report = optimize_particles({"foot": foot, "place": place}, evaluate, 25, 0.025)
    assert torch.all(best["foot"].abs() <= 0.22)
    assert (best["foot"].sign() == grasp.sign()).all()
    assert (best["place"].sign() == grasp.sign()).all()
    assert (report["iteration"] > 0).all()
    assert torch.equal(foot, torch.zeros_like(foot))


def test_center_plateau_tracks_complete_feasible_particles_and_cumulative_gain():
    from cutamp.step_adam import optimize_particles, PlateauStop

    distances_m = [0.030, 0.020, 0.010, 0.0094, 0.0087, 0.0086, 0.009, 0.0082]
    calls = []

    def evaluate(values):
        i = len(calls)
        calls.append(i)
        # Particle zero has ideal centering but always violates a hard constraint.
        x = values["foot"][:, 0]
        violation = x * 0 + x.new_tensor([1., float(i < 2)])
        centers = x * 0 + x.new_tensor([0., 4. * distances_m[i] ** 2])
        return violation, centers

    _, report = optimize_particles({"foot": torch.zeros(2, 1)}, evaluate, 20, .001,
                                   plateau=PlateauStop(.001, 3), center_weight=4.)
    assert report["updates_completed"] == 7 and report["stop_reason"] == "plateau"
    assert report["iteration"].tolist() == [0, 7]


@pytest.mark.parametrize("feasible,minimum,expected,reason", [
    (False, 0, 6, "max_updates"), (True, 5, 5, "plateau"),
])
def test_plateau_needs_a_feasible_particle_and_respects_minimum(feasible, minimum, expected, reason):
    from cutamp.step_adam import optimize_particles, PlateauStop

    def evaluate(values):
        x = values["x"].flatten()
        return x * 0 + float(not feasible), x * 0 + .01

    _, report = optimize_particles({"x": torch.zeros(1, 1)}, evaluate, 6, .01,
                                   plateau=PlateauStop(.001, 2, minimum))
    assert report["updates_completed"] == expected and report["stop_reason"] == reason


@pytest.mark.parametrize("options", [
    {"step_adam_plateau_mm": 0.},
    {"step_adam_plateau_mm": 1., "step_adam_plateau_patience": 0},
    {"step_adam_plateau_mm": 1., "step_adam_plateau_min_updates": -1},
    {"step_adam_plateau_mm": 1., "com_center_weight": 0},
    {"step_adam_plateau_mm": 1., "step_mode": "off"},
])
def test_invalid_plateau_controls(options):
    from cutamp.config import TAMPConfiguration, validate_tamp_config

    with pytest.raises(ValueError, match="CoM plateau"):
        validate_tamp_config(TAMPConfiguration(**options))
