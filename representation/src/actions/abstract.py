from abc import ABC, abstractmethod
from math import ceil
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


class AbstractAction(ABC):
    input_column: Optional[Union[list[str], str]]
    output_column: Optional[Union[list[str], str]]

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the action."""

    def batchify(self, df: pd.DataFrame, batch_size: int = 1024):
        chunks = [(i + 1) * batch_size for i in range(ceil(len(df) / batch_size))]
        outcomes = []
        for chunk in tqdm(np.array_split(df, chunks)):
            outcomes.append(self(chunk))  # type: ignore
        return pd.concat(outcomes)
