from abc import ABC, abstractmethod
from typing import Union

from torch import Tensor


class AbstractCollate(ABC):
    @abstractmethod
    def collate_fn(self, batch: dict[str, Union[list[Tensor], Tensor]]) -> dict[str, Union[list[Tensor], Tensor]]:
        """Collates a batch"""
