import copy
import json
import os

import numpy as np
import pytest
import torch
from ase.build import bulk
from metatomic.torch import ModelMetadata, load_atomistic_model
from metatomic_ase import MetatomicCalculator

from upet import get_upet, save_upet
from upet._models import _resolve_and_download_checkpoint
from upet.ase import UPETCalculator


# "energy/triple" uses "energy/double", declared before it, and an integer coefficient
WEIGHTED_SUMS = {
    "energy/double": {"energy": 2.0},
    "energy/triple": {"energy/double": 1.0, "energy": 1},
}


@pytest.fixture(scope="module")
def plain_checkpoint():
    """Plain PET checkpoint, and the name of the file it was published as."""
    _, _, path = _resolve_and_download_checkpoint("pet-mad", "xs", "1.6.0")
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    # the published checkpoint wraps the PET model in an uncertainty (LLPR) model,
    # only the PET model is used here
    return checkpoint["wrapped_model_checkpoint"], os.path.basename(path)


def _tagged_checkpoint(tmp_path, plain_checkpoint, declaration):
    """Save the plain checkpoint, declaring ``declaration`` in its metadata."""
    checkpoint, filename = plain_checkpoint
    checkpoint = copy.copy(checkpoint)
    # no name, so that upet fills in the metadata and has to keep the extra entry
    checkpoint["metadata"] = ModelMetadata(extra={"weighted_sums": declaration})
    # keep the standard name, so that upet recognizes the model
    tagged_path = tmp_path / filename
    torch.save(checkpoint, tagged_path)
    return tagged_path


def test_weighted_sums_from_checkpoint(tmp_path, plain_checkpoint):
    path = _tagged_checkpoint(tmp_path, plain_checkpoint, json.dumps(WEIGHTED_SUMS))
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.rattle(0.05, seed=0)

    energies = {}
    forces = {}
    for variant in [None, "double", "triple"]:
        atoms.calc = UPETCalculator(
            checkpoint_path=path, variants={"energy": variant}, device="cpu"
        )
        energies[variant] = atoms.get_potential_energy()
        forces[variant] = atoms.get_forces()

    assert np.isclose(energies["double"], 2.0 * energies[None])
    assert np.isclose(energies["triple"], 3.0 * energies[None])
    np.testing.assert_allclose(forces["triple"], 3.0 * forces[None], atol=1e-5)


def test_save_upet_keeps_weighted_sums(tmp_path, plain_checkpoint):
    declaration = json.dumps(WEIGHTED_SUMS)
    path = _tagged_checkpoint(tmp_path, plain_checkpoint, declaration)
    output = tmp_path / "model.pt"
    save_upet(checkpoint_path=path, output=output)

    # the exported model has the weighted sums, and tells how they are defined
    model = load_atomistic_model(str(output))
    assert "energy/double" in model.capabilities().outputs
    assert "energy/triple" in model.capabilities().outputs
    assert json.loads(model.metadata().extra["weighted_sums"]) == WEIGHTED_SUMS

    # and runs without upet
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    energies = {}
    for variant in [None, "triple"]:
        atoms.calc = MetatomicCalculator(
            str(output), variants={"energy": variant}, device="cpu"
        )
        energies[variant] = atoms.get_potential_energy()
    assert np.isclose(energies["triple"], 3.0 * energies[None])


@pytest.mark.parametrize(
    "declaration, message",
    [
        ('{"energy/mix": {"energy": 0.5}', "is not valid JSON"),
        ('{"energy/mix": 0.5}', "must map the name of each weighted sum"),
        ('{"energy/mix": {"energy": "0.5"}}', "must map the name of each weighted sum"),
        (
            '{"energy/mix": {"energy/pbe": 0.5}}',
            "unable to add the 'energy/mix' weighted sum .* "
            "does not have a 'energy/pbe' output",
        ),
    ],
)
def test_invalid_weighted_sums(tmp_path, plain_checkpoint, declaration, message):
    path = _tagged_checkpoint(tmp_path, plain_checkpoint, declaration)
    with pytest.raises(ValueError, match=message):
        get_upet(checkpoint_path=path)
