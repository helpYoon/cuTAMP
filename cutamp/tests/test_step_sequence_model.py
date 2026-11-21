import hashlib

import pytest
import torch

from cutamp.step_sequence_model import DEFAULT_STEP_SEQUENCE_MODEL, load_step_sequence_model


def test_shipped_step_sequence_reaches_both_step_commands_through_the_frozen_model():
    # The default must be the exact export used for the verified plateau benchmark.
    digest = hashlib.sha256(DEFAULT_STEP_SEQUENCE_MODEL.read_bytes()).hexdigest()
    assert digest == "e6558a4779e89c0ed68ae29a87a3668e2db95d214355741eb0d8e9d1526615fb"
    model = load_step_sequence_model(DEFAULT_STEP_SEQUENCE_MODEL, "cpu")
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert model is load_step_sequence_model(DEFAULT_STEP_SEQUENCE_MODEL, "cpu")

    zero3 = torch.zeros(1, 3)
    joints = torch.zeros(1, 29)
    history = torch.zeros(1, 10, 93, requires_grad=True)
    state = (zero3, torch.eye(3)[None], zero3, zero3, joints, joints, torch.zeros(1, 58), history, torch.zeros(1, 8))
    start_feet = torch.tensor([[[0.0, 0.10, 0.0], [0.0, -0.10, 0.0]]])
    commands = torch.tensor([[[0.12, 0.21, 0.04], [0.09, -0.22, -0.05]]], requires_grad=True)
    body, soles = model(state, torch.zeros(1, 38), start_feet, torch.tensor([[0, 1]]), commands)
    assert body.shape == (1, 2, 6, 76) and soles.shape == (1, 2, 6, 2, 3)
    dc, dh = torch.autograd.grad(soles[:, 1, -1].square().sum(), (commands, history))
    assert dc.isfinite().all() and (dc.abs().sum(-1) > 0).all()
    assert dh.isfinite().all() and dh.abs().sum() > 0


def test_cached_step_sequence_checks_the_native_deployment():
    load_step_sequence_model(DEFAULT_STEP_SEQUENCE_MODEL, "cpu")
    with pytest.raises(ValueError, match="different landing deployment"):
        load_step_sequence_model(DEFAULT_STEP_SEQUENCE_MODEL, "cpu", expected_deployment="different")
