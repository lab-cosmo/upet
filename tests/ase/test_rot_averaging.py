import re

import numpy as np
import pytest
from _utils import non_conservative_error_message, supports_non_conservative
from ase.build import bulk
from scipy.spatial.transform import Rotation

from upet._models import get_versions_for_model
from upet._version import UPET_AVAILABLE_MODELS
from upet.ase import UPETCalculator


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_calc_rot_averaging(model_name):
    if "-xl" in model_name or "-l" in model_name:
        pytest.skip("Skipping XL models and L models due to large size.")
    atoms = bulk("C", cubic=True, a=3.57, crystalstructure="diamond")
    atoms.rattle(0.05)
    atoms.calc = UPETCalculator(model=model_name)

    target_energy = atoms.get_potential_energy()
    target_forces = atoms.get_forces()
    target_stress = atoms.get_stress()

    atoms.calc = UPETCalculator(
        model=model_name,
        rotational_average_order=3,
        rotational_average_batch_size=8,
    )
    averaged_energy = atoms.get_potential_energy()
    averaged_forces = atoms.get_forces()
    averaged_stress = atoms.get_stress()
    np.testing.assert_allclose(averaged_energy, target_energy, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(averaged_forces, target_forces, atol=1e-1, rtol=1e-1)
    np.testing.assert_allclose(averaged_stress, target_stress, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_calc_rot_averaging_non_conservative(model_name):
    non_conservative = "forces"
    if "-xl" in model_name or "-l" in model_name:
        pytest.skip("Skipping XL models and L models due to large size.")
    version = max(get_versions_for_model(*model_name.rsplit("-", 1)))
    if not supports_non_conservative(model_name, version):
        message = non_conservative_error_message(model_name, version, non_conservative)
        with pytest.raises(NotImplementedError, match=re.escape(message)):
            _ = UPETCalculator(model=model_name, non_conservative=non_conservative)
    else:
        atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
        atoms.rattle(0.05)
        atoms.calc = UPETCalculator(model=model_name, non_conservative=non_conservative)

        target_energy = atoms.get_potential_energy()
        target_forces = atoms.get_forces()
        target_stress = atoms.get_stress()

        atoms.calc = UPETCalculator(
            model=model_name,
            non_conservative=non_conservative,
            rotational_average_order=3,
            rotational_average_batch_size=8,
        )
        averaged_energy = atoms.get_potential_energy()
        averaged_forces = atoms.get_forces()
        averaged_stress = atoms.get_stress()
        assert "energy_rot_std" in atoms.calc.results
        np.testing.assert_allclose(averaged_energy, target_energy, atol=5e-2)
        np.testing.assert_allclose(averaged_forces, target_forces, atol=1e-1)
        np.testing.assert_allclose(averaged_stress, target_stress, atol=5e-2)


def test_batched_calc_rot_averaging():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.rattle(0.05)
    calc = UPETCalculator(model="pet-mad-s", rotational_average_order=3)
    batched_calc = UPETCalculator(
        model="pet-mad-s",
        rotational_average_order=3,
        rotational_average_batch_size=5,
    )
    atoms.calc = calc
    target_energy = atoms.get_potential_energy()
    target_forces = atoms.get_forces()
    target_stress = atoms.get_stress()

    atoms.calc = batched_calc
    batched_energy = atoms.get_potential_energy()
    batched_forces = atoms.get_forces()
    batched_stress = atoms.get_stress()

    np.testing.assert_allclose(batched_energy, target_energy, atol=1e-6)
    np.testing.assert_allclose(batched_forces, target_forces, atol=1e-6)
    np.testing.assert_allclose(batched_stress, target_stress, atol=1e-6)


def test_energy_rot_std():
    # the standard deviation of the energy over the quadrature matches the one over
    # random rotations
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.rattle(0.05, seed=0)
    atoms.calc = UPETCalculator(model="pet-mad-s", rotational_average_order=3)
    atoms.get_potential_energy()

    calculator = UPETCalculator(model="pet-mad-s")
    energies = []
    for rotation in Rotation.random(100, random_state=0).as_matrix():
        rotated = atoms.copy()
        rotated.set_cell(atoms.cell.array @ rotation.T)
        rotated.positions = atoms.positions @ rotation.T
        rotated.calc = calculator
        energies.append(rotated.get_potential_energy())

    np.testing.assert_allclose(
        atoms.calc.results["energy_rot_std"], np.std(energies), rtol=0.2
    )
