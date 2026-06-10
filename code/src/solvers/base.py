from abc import ABC, abstractmethod
from typing import Any, Mapping

from dimod import SampleSet

Qubo = Mapping[tuple[Any, Any], float]
SampleConfig = Mapping[str, Any]


class SolverBase(ABC):
    default_config: dict = {"num_reads": 10}

    @abstractmethod
    def solve(self, Q: Qubo, sample_config: SampleConfig | None = None) -> SampleSet:
        pass

    def _merged_config(self, sample_config: SampleConfig | None) -> dict:
        if sample_config is None:
            return dict(self.default_config)
        return {**self.default_config, **sample_config}
