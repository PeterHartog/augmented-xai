import pytest
import torch
from torch import Tensor
from torchmetrics import Accuracy

from representation.src.metrics.metrics.accuracy import (
    CharacterAccuracy,
    SequenceAccuracy,
)


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    ([torch.tensor([1, 2, 3, 3, 0]), torch.tensor([1, 2, 3, 4, 0]), 0.75],),
)
def test_base_metric(y_true: Tensor, y_pred: Tensor, expected: float) -> None:
    accuracy = CharacterAccuracy()
    assert accuracy(y_true, y_pred) == expected
    assert accuracy.compute() == expected
    accuracy.reset()
    assert accuracy(y_true, y_pred) == expected


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    (
        [torch.tensor([1, 2, 3, 4, 0]), torch.tensor([1, 2, 3, 4, 0]), 1.0],
        [torch.tensor([1, 2, 3, 4, 0]), torch.tensor([1, 3, 3, 4, 0]), 0.75],
        [torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 2, 1, 3, 4]), 0.5],
        [torch.tensor([1, 2, 3, 4, 0]), torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0]), 1.0],
    ),
)
def test_char_accuracy(y_true: Tensor, y_pred: Tensor, expected: float) -> None:
    accuracy = CharacterAccuracy()
    accuracy.update(y_true, y_pred)
    assert accuracy.compute() == expected


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    (
        [torch.tensor([[1, 2, 3, 0], [2, 3, 1, 0]]), torch.tensor([[1, 2, 3, 0], [2, 3, 1, 0]]), 1.0],
        [torch.tensor([[1, 2, 3, 0], [2, 3, 1, 0]]), torch.tensor([[1, 2, 3, 0], [2, 3, 2, 0]]), 0.5],
        [torch.tensor([[1, 2, 3, 0], [2, 3, 1, 0]]), torch.tensor([[1, 2, 3, 0], [2, 3, 1, 2]]), 0.5],
    ),
)
def test_seq_accuracy(y_true: Tensor, y_pred: Tensor, expected: float) -> None:
    accuracy = SequenceAccuracy()
    accuracy.update(y_true, y_pred)
    assert accuracy.compute() == expected


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    (
        [torch.tensor([1, 2, 3, 4, 0]), torch.tensor([1, 2, 3, 4, 0]), 1.0],
        [torch.tensor([1, 2, 3, 4, 0]), torch.tensor([1, 3, 3, 4, 0]), 0.75],
        [torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 2, 1, 3, 4]), 0.5],
        [torch.tensor([1, 2, 3, 4, 0]), torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0]), 1.0],
    ),
)
def test_base_difference(y_true: Tensor, y_pred: Tensor, expected: float) -> None:
    torch_acc = Accuracy(task="multiclass", num_classes=5, ignore_index=0)
    torch_acc.update(y_true, y_pred)
    accuracy = CharacterAccuracy()
    accuracy.update(y_true, y_pred)
    assert accuracy.compute() == torch_acc.compute() == expected
