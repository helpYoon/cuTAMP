import torch

from cutamp import step_ik


def test_implicit_conditioning_checks_absolute_and_relative_thresholds(monkeypatch):
    # Equality in either of the two strict tests must fail. A small negative
    # eigenvalue must fail even when all unrotated diagonal entries are positive.
    eigs = torch.tensor([
        [0.5, 2.0], [1e-10, 0.5], [1.01e-10, 0.5],
        [2e-10, 2.0], [2.1e-10, 2.0], [-1e-5, 10.0],
    ], dtype=torch.float64)
    matrices = eigs.diag_embed()
    rotation = torch.tensor([[0.8, -0.6], [0.6, 0.8]], dtype=eigs.dtype)
    matrices[-1] = rotation @ matrices[-1] @ rotation.T
    batch_sizes = []
    eigenvalues = torch.linalg.eigvalsh

    def record_eigenvalues(batch):
        batch_sizes.append(len(batch))
        return eigenvalues(batch)

    monkeypatch.setattr(torch.linalg, "eigvalsh", record_eigenvalues)
    ok = step_ik._well_conditioned(matrices)
    assert ok.tolist() == [True, False, True, False, True, False]
    assert sum(batch_sizes) < len(matrices)
