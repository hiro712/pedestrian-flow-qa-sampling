from typing import cast

import openjij as oj
from dimod import BinaryQuadraticModel, SampleSet
from dwave.samplers import SteepestDescentSampler

from src.solvers.base import Qubo, SampleConfig, SolverBase


class SQASolver(SolverBase):
    def __init__(self, use_greedy: bool = False):
        self.use_greedy = use_greedy

    def solve(self, Q: Qubo, sample_config: SampleConfig | None = None) -> SampleSet:
        cfg = self._merged_config(sample_config)
        sampler = oj.SQASampler()
        sampleset = sampler.sample_qubo(Q, **cfg)
        sampleset = cast(SampleSet, sampleset)

        if self.use_greedy:
            bqm = BinaryQuadraticModel.from_qubo(Q)
            sampleset = SteepestDescentSampler().sample(bqm, initial_states=sampleset)

        return sampleset
