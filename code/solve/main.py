# Load environment variables
import os

# Type-related
from typing import Mapping, Optional, Any, cast
from abc import ABC, abstractmethod
from dimod import SampleSet

# SA
import openjij as oj
import neal

# QA
from dimod import BinaryQuadraticModel, concatenate
from dwave.system import DWaveSampler
from dwave.samplers import SteepestDescentSampler
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.embedding.chain_breaks import MinimizeEnergy
from minorminer import find_embedding

Qubo = Mapping[tuple[Any, Any], float]
SampleConfig = Mapping[str, Any]


class SolverBase(ABC):
    def __init__(self):
        self.sample_config = {"num_reads": 10}

    @abstractmethod
    def solve(self, Q: Qubo, sample_config: Optional[SampleConfig] = None) -> SampleSet:
        pass


class SASolver(SolverBase):
    def __init__(self, is_oj: bool = True):
        self.is_oj = is_oj
        super().__init__()

    def solve(self, Q: Qubo, sample_config: Optional[SampleConfig] = None) -> SampleSet:
        if self.is_oj:
            return self._solve_with_oj(Q, sample_config)
        else:
            return self._solve_with_neal(Q, sample_config)

    def _solve_with_oj(
        self, Q: Qubo, sample_config: Optional[SampleConfig] = None
    ) -> SampleSet:
        if sample_config is None:
            sample_config = self.sample_config
        else:
            sample_config = {**self.sample_config, **sample_config}

        sampler = oj.SASampler()
        sampleset = sampler.sample_qubo(Q, **sample_config)
        sampleset = cast(SampleSet, sampleset)  # type cast
        return sampleset

    def _solve_with_neal(
        self, Q: Qubo, sample_config: Optional[SampleConfig] = None
    ) -> SampleSet:
        if sample_config is None:
            sample_config = self.sample_config
        else:
            sample_config = {**self.sample_config, **sample_config}

        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(Q, **sample_config)
        return sampleset


class QASolver(SolverBase):
    def __init__(self, use_greedy: bool = False):
        SOLVER_NAME = os.getenv("DWAVE_SOLVER_NAME")
        TOKEN = os.getenv("DWAVE_API_TOKEN")
        self.sampler = DWaveSampler(solver=SOLVER_NAME, token=TOKEN)
        self.use_greedy = use_greedy
        self.max_num_reads = 5000  # D-Wave's per-call sampling limit
        super().__init__()

    def solve(self, Q: Qubo, sample_config: Optional[SampleConfig] = None) -> SampleSet:
        return self._solve_with_advantage(Q, sample_config)

    def _solve_with_advantage(
        self, Q: Qubo, sample_config: Optional[SampleConfig] = None
    ) -> SampleSet:
        if sample_config is None:
            sample_config = self.sample_config
        else:
            sample_config = {**self.sample_config, **sample_config}

        # Check whether num_reads was explicitly specified
        total_num_reads = sample_config.get("num_reads")

        # Check whether splitting into multiple calls is needed (num_reads is specified and exceeds the limit)
        if total_num_reads is not None and total_num_reads > self.max_num_reads:
            # Compute the number of chunks
            num_iterations = (
                total_num_reads + self.max_num_reads - 1
            ) // self.max_num_reads
            reads_per_iteration = []

            for i in range(num_iterations):
                if i < num_iterations - 1:
                    reads_per_iteration.append(self.max_num_reads)
                else:
                    # Handle the remainder on the last iteration
                    remaining_reads = total_num_reads - (
                        self.max_num_reads * (num_iterations - 1)
                    )
                    reads_per_iteration.append(remaining_reads)

            # Sample on each iteration and combine the results
            combined_sampleset = None
            bqm = self._convert_bqm_from_qubo(Q)
            embedding = self._find_embedding(bqm)
            embed_bqm = self._embed_bqm(bqm, embedding)

            for num_reads in reads_per_iteration:
                # Update the config
                current_config = {**sample_config, "num_reads": num_reads}

                # Sample
                response = self.sampler.sample(embed_bqm, **current_config)

                # Unembed
                sampleset = unembed_sampleset(
                    response,
                    embedding,
                    bqm,
                    chain_break_method=MinimizeEnergy(bqm, embedding),
                    chain_break_fraction=True,
                )
                if not isinstance(sampleset, SampleSet):
                    raise ValueError(
                        "Unembedding failed, resulting object is not a SampleSet."
                    )

                # Combine the results
                if combined_sampleset is None:
                    combined_sampleset = sampleset
                else:
                    combined_sampleset = combined_sampleset.aggregate()
                    new_sampleset = sampleset.aggregate()
                    # Concatenate the SampleSets
                    combined_sampleset = concatenate(
                        [combined_sampleset, new_sampleset]
                    )

            # Confirm combined_sampleset is not None
            if combined_sampleset is None:
                raise ValueError("Sampling failed, no valid sampleset was generated.")

            sampleset = combined_sampleset
        else:
            # Normal sampling (10000 or fewer)
            bqm = self._convert_bqm_from_qubo(Q)
            embedding = self._find_embedding(bqm)
            embed_bqm = self._embed_bqm(bqm, embedding)

            # Sample
            response = self.sampler.sample(embed_bqm, **sample_config)

            # Unembed
            sampleset = unembed_sampleset(
                response,
                embedding,
                bqm,
                chain_break_method=MinimizeEnergy(bqm, embedding),
                chain_break_fraction=True,
            )
            if not isinstance(sampleset, SampleSet):
                raise ValueError(
                    "Unembedding failed, resulting object is not a SampleSet."
                )

        # Improve the solution with Greedy (optional)
        if self.use_greedy:
            bqm = self._convert_bqm_from_qubo(Q)
            sampleset = SteepestDescentSampler().sample(bqm, initial_states=sampleset)

        return sampleset

    def _find_embedding(self, bqm: BinaryQuadraticModel, timeoutsec: int = 1000):
        logical_edges = list(bqm.quadratic.keys())
        hardware_edges = self.sampler.edgelist

        emb = find_embedding(
            logical_edges,
            hardware_edges,
            timeout=timeoutsec,
            verbose=0,  # When set to 1, it prints out information about the embedding process.
        )

        # Check whether embedding failed
        if not emb:
            raise ValueError("No embedding found.")

        # Type cast (find_embedding's return type is Any)
        emb = cast(Mapping[int, list[int]], emb)

        # Check whether the embedding covers all logical variables (should never hit this branch)
        logical_vars = set(bqm.variables)
        if set(emb.keys()) != logical_vars:
            raise ValueError("Embedding does not cover all logical variables.")

        return emb

    def _convert_bqm_from_qubo(self, Q: Qubo) -> BinaryQuadraticModel:
        bqm = BinaryQuadraticModel.from_qubo(Q)
        return bqm

    def _embed_bqm(
        self,
        bqm: BinaryQuadraticModel,
        embedding: Mapping[int, list[int]],
        chain_strength: Optional[float] = None,
    ) -> BinaryQuadraticModel:
        embedded_bqm = embed_bqm(bqm, embedding, self.sampler.adjacency, chain_strength)
        if not isinstance(embedded_bqm, BinaryQuadraticModel):
            raise ValueError(
                "Embedding failed, resulting object is not a BinaryQuadraticModel."
            )
        return embedded_bqm


class SQASolver(SolverBase):
    """
    Simulated Quantum Annealing (SQA) solver using OpenJij.

    Parameters
    ----------
    use_greedy : bool, optional
        Whether to locally improve the obtained samples with SteepestDescent
        on the BinaryQuadraticModel. Default False.
    """

    def __init__(self, use_greedy: bool = False):
        self.use_greedy = use_greedy
        super().__init__()

    def solve(self, Q: Qubo, sample_config: Optional[SampleConfig] = None) -> SampleSet:
        if sample_config is None:
            sample_config = self.sample_config
        else:
            sample_config = {**self.sample_config, **sample_config}

        sampler = oj.SQASampler()
        sampleset = sampler.sample_qubo(Q, **sample_config)
        sampleset = cast(SampleSet, sampleset)  # type cast

        # Improve the solution with Greedy (optional)
        if self.use_greedy:
            bqm = BinaryQuadraticModel.from_qubo(Q)
            sampleset = SteepestDescentSampler().sample(bqm, initial_states=sampleset)

        return sampleset


if __name__ == "__main__":
    # Example QUBO
    Q = {
        (0, 0): -1.0,
        (1, 1): -1.0,
        (2, 2): -1.0,
        (0, 1): 2.0,
        (1, 2): 2.0,
        (0, 2): 2.0,
    }

    # SA
    solver = SASolver()
    ss = solver.solve(Q)

    print("=== SA Result ===")
    print(ss)
    print()
    # SQA
    solver = SQASolver()
    ss = solver.solve(Q)

    print("=== SQA Result ===")
    print(ss)
    print()
    # QA
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env.local")  # load environment variables from .env.local
    print("DWAVE_SOLVER_NAME:", os.getenv("DWAVE_SOLVER_NAME"))
    solver = QASolver()
    ss = solver.solve(Q)
    print("=== QA Result ===")
    print(ss)
    print()
