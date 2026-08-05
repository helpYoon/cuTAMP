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
