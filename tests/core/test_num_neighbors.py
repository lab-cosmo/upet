import pytest
import torch

from upet._models import _set_num_neighbors, check_num_neighbors, get_upet


def _hypers(num_neighbors_adaptive=16, variable_num_neighbors_adaptive=(8, 32)):
    return {
        "num_neighbors_adaptive": num_neighbors_adaptive,
        "variable_num_neighbors_adaptive": variable_num_neighbors_adaptive,
    }


@pytest.mark.parametrize("num_neighbors", [8, 16, 32])
def test_check_num_neighbors_in_range(num_neighbors):
    check_num_neighbors(_hypers(), num_neighbors)


@pytest.mark.parametrize("num_neighbors", [0, 7, 33])
def test_check_num_neighbors_out_of_range(num_neighbors):
    with pytest.raises(ValueError, match=r"must be within \[8, 32\]"):
        check_num_neighbors(_hypers(), num_neighbors)


@pytest.mark.parametrize("num_neighbors", [16.0, "16", True])
def test_check_num_neighbors_type(num_neighbors):
    with pytest.raises(TypeError, match="must be an integer"):
        check_num_neighbors(_hypers(), num_neighbors)


def test_check_num_neighbors_no_adaptive_cutoff():
    hypers = _hypers(num_neighbors_adaptive=None, variable_num_neighbors_adaptive=None)
    with pytest.raises(ValueError, match="does not use the adaptive cutoff"):
        check_num_neighbors(hypers, 16)


def test_check_num_neighbors_fixed():
    hypers = _hypers(variable_num_neighbors_adaptive=None)
    # setting the default value is a no-op, and always allowed
    check_num_neighbors(hypers, 16)
    with pytest.raises(ValueError, match="fixed number of neighbors"):
        check_num_neighbors(hypers, 20)


class _FakePET(torch.nn.Module):
    """Stands in for ``metatrain.pet.PET``, recording calls to the setter."""

    def __init__(self, hypers):
        super().__init__()
        self.hypers = hypers
        self.calls = []

    def _set_num_neighbors_adaptive(self, num_neighbors, variable_range):
        self.calls.append((num_neighbors, variable_range))


class _FakeMetatrainModel(torch.nn.Module):
    """Stands in for ``metatrain.utils.wrapper.MetatrainModel``."""

    def __init__(self, core):
        super().__init__()
        self.core = core


class _FakeLLPR(torch.nn.Module):
    """Stands in for ``metatrain.llpr.LLPRUncertaintyModel``."""

    def __init__(self, model):
        super().__init__()
        self.model = model


@pytest.mark.parametrize("with_llpr", [False, True])
def test_set_num_neighbors(with_llpr):
    pet = _FakePET(_hypers())
    model = _FakeMetatrainModel(pet)
    if with_llpr:
        model = _FakeMetatrainModel(_FakeLLPR(model))

    _set_num_neighbors(model, 24)
    assert pet.calls == [(24, (8, 32))]

    # the default value does not touch the model
    _set_num_neighbors(model, 16)
    assert pet.calls == [(24, (8, 32))]

    with pytest.raises(ValueError):
        _set_num_neighbors(model, 64)
    assert len(pet.calls) == 1


def test_set_num_neighbors_not_pet():
    with pytest.raises(ValueError, match="does not wrap a PET model"):
        _set_num_neighbors(_FakeMetatrainModel(torch.nn.Linear(1, 1)), 16)


def test_get_upet_num_neighbors_out_of_range():
    # no published model is trained with a range reaching this far
    with pytest.raises(ValueError, match="num_neighbors"):
        get_upet(model="pet-mad", size="xs", num_neighbors=10_000)


# TODO: once a checkpoint trained with `variable_num_neighbors_adaptive` is
# published, check that `get_upet(..., num_neighbors=n)` changes the predictions
# for n within the range, and that the exported model keeps the value after
# `torch.jit.script` and `save_upet`.
