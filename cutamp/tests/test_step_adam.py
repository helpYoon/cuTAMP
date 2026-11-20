import torch


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
