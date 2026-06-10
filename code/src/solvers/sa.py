from typing import cast

import openjij as oj
from dimod import SampleSet

from src.solvers.base import Qubo, SampleConfig, SolverBase


class SASolver(SolverBase):
    def solve(self, Q: Qubo, sample_config: SampleConfig | None = None) -> SampleSet:
        cfg = self._merged_config(sample_config)
        sampler = oj.SASampler()
        sampleset = sampler.sample_qubo(Q, **cfg)
        return cast(SampleSet, sampleset)
