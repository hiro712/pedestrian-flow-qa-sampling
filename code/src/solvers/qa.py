import os
from typing import Mapping, cast

from dimod import BinaryQuadraticModel, SampleSet, concatenate
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.embedding.chain_breaks import MinimizeEnergy
from dwave.samplers import SteepestDescentSampler
from dwave.system import DWaveSampler
from minorminer import find_embedding

from src.solvers.base import Qubo, SampleConfig, SolverBase

MAX_READS_PER_CALL = 5000


class QASolver(SolverBase):
    """D-Wave Advantage hardware solver. Reads credentials from .env.local."""

    def __init__(self, use_greedy: bool = False):
        solver_name = os.getenv("DWAVE_SOLVER_NAME")
        token = os.getenv("DWAVE_API_TOKEN")
        self.hw_sampler = DWaveSampler(solver=solver_name, token=token)
        self.use_greedy = use_greedy

    def solve(self, Q: Qubo, sample_config: SampleConfig | None = None) -> SampleSet:
        cfg = self._merged_config(sample_config)
        bqm = BinaryQuadraticModel.from_qubo(Q)
        embedding = self._find_embedding(bqm)
        embedded = self._embed(bqm, embedding)

        total_reads = cfg.get("num_reads", 10)
        run_cfg = {k: v for k, v in cfg.items() if k != "num_reads"}

        if total_reads <= MAX_READS_PER_CALL:
            response = self.hw_sampler.sample(embedded, num_reads=total_reads, **run_cfg)
            sampleset = self._unembed(response, embedding, bqm)
        else:
            chunks = self._split_reads(total_reads)
            sets = []
            for n in chunks:
                response = self.hw_sampler.sample(embedded, num_reads=n, **run_cfg)
                sets.append(self._unembed(response, embedding, bqm))
            sampleset = concatenate(sets)

        if self.use_greedy:
            sampleset = SteepestDescentSampler().sample(bqm, initial_states=sampleset)

        return sampleset

    def _find_embedding(self, bqm: BinaryQuadraticModel) -> Mapping:
        logical_edges = list(bqm.quadratic.keys())
        emb = find_embedding(logical_edges, self.hw_sampler.edgelist, timeout=1000, verbose=0)
        if not emb:
            raise ValueError("No embedding found.")
        emb = cast(Mapping, emb)
        if set(emb.keys()) != set(bqm.variables):
            raise ValueError("Embedding does not cover all logical variables.")
        return emb

    def _embed(self, bqm: BinaryQuadraticModel, embedding: Mapping, chain_strength: float | None = None) -> BinaryQuadraticModel:
        embedded = embed_bqm(bqm, embedding, self.hw_sampler.adjacency, chain_strength)
        if not isinstance(embedded, BinaryQuadraticModel):
            raise ValueError("embed_bqm returned unexpected type.")
        return embedded

    def _unembed(self, response, embedding: Mapping, bqm: BinaryQuadraticModel) -> SampleSet:
        ss = unembed_sampleset(
            response,
            embedding,
            bqm,
            chain_break_method=MinimizeEnergy(bqm, embedding),
            chain_break_fraction=True,
        )
        if not isinstance(ss, SampleSet):
            raise ValueError("unembed_sampleset returned unexpected type.")
        return ss

    @staticmethod
    def _split_reads(total: int) -> list[int]:
        chunks = []
        remaining = total
        while remaining > 0:
            n = min(remaining, MAX_READS_PER_CALL)
            chunks.append(n)
            remaining -= n
        return chunks
