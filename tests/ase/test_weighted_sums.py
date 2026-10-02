import json
import os

import numpy as np
import torch
from ase.build import bulk
from metatomic.torch import ModelMetadata

from upet._models import _resolve_and_download_checkpoint
from upet.ase import UPETCalculator


def _tagged_checkpoint(tmp_path, weighted_sums):
    """Plain PET checkpoint declaring ``weighted_sums`` in its metadata."""
    _, _, path = _resolve_and_download_checkpoint("pet-mad", "xs", "1.6.0")
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    # the published checkpoint wraps the PET model in an uncertainty (LLPR) model,
    # only the PET model is used here
    checkpoint = checkpoint["wrapped_model_checkpoint"]
    # no name, so that upet fills in the metadata and has to keep the extra entry
    checkpoint["metadata"] = ModelMetadata(
        extra={"weighted_sums": json.dumps(weighted_sums)}
    )
    # keep the standard name, so that upet recognizes the model
    tagged_path = tmp_path / os.path.basename(path)
    torch.save(checkpoint, tagged_path)
    return tagged_path


def test_weighted_sums_from_checkpoint(tmp_path):
    path = _tagged_checkpoint(tmp_path, {"energy/double": {"energy": 2.0}})
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)

    energies = {}
    for variant in [None, "double"]:
        atoms.calc = UPETCalculator(
            checkpoint_path=path, variants={"energy": variant}, device="cpu"
        )
        energies[variant] = atoms.get_potential_energy()

    assert np.isclose(energies["double"], 2.0 * energies[None])
