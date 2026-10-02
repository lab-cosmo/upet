import pytest
from ase.build import bulk, molecule

from upet._models import get_versions_for_model
from upet._version import UPET_AVAILABLE_MODELS
from upet.ase import UPETCalculator


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_basic_usage(model_name):
    if "-xl" in model_name or "-l" in model_name:
        pytest.skip("Skipping XL models and L models due to large size.")
    atoms = (
        molecule("H2O")
        if any(name in model_name for name in ("spice", "mols"))
        else bulk("C", cubic=True, a=5.43, crystalstructure="diamond")
    )

    model, size = model_name.rsplit("-", 1)
    all_model_versions = get_versions_for_model(model, size)

    for version in all_model_versions:
        calc = UPETCalculator(model=model_name, version=version)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        virial = atoms.get_stress()
        assert isinstance(energy, float)
        assert forces.shape == (len(atoms), 3)
        assert virial.shape == (6,)


def test_info_change_invalidates_cache():
    atoms = molecule("H2O", vacuum=5.0)
    atoms.calc = UPETCalculator(model="pet-omol-s", device="cpu")
    reference = UPETCalculator(model="pet-omol-s", device="cpu")

    for charge, spin in [(0, 1), (1, 2), (0, 3)]:
        atoms.info = {"charge": charge, "spin": spin}
        reference.reset()
        expected = reference.get_potential_energy(atoms)
        assert atoms.get_potential_energy() == pytest.approx(expected, abs=1e-5)
