# 環境変数の読み込み
import os

# 型関連
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
        sampleset = cast(SampleSet, sampleset)  # 型キャスト
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
        self.max_num_reads = 5000  # D-Waveの1回のサンプリング上限
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

        # num_readsが明示的に指定されているかチェック
        total_num_reads = sample_config.get("num_reads")

        # 複数回に分割する必要があるかチェック（num_readsが指定されており、かつ上限を超える場合）
        if total_num_reads is not None and total_num_reads > self.max_num_reads:
            # 分割回数を計算
            num_iterations = (
                total_num_reads + self.max_num_reads - 1
            ) // self.max_num_reads
            reads_per_iteration = []

            for i in range(num_iterations):
                if i < num_iterations - 1:
                    reads_per_iteration.append(self.max_num_reads)
                else:
                    # 最後のイテレーションで残りを処理
                    remaining_reads = total_num_reads - (
                        self.max_num_reads * (num_iterations - 1)
                    )
                    reads_per_iteration.append(remaining_reads)

            # 各イテレーションでサンプリングして結果を結合
            combined_sampleset = None
            bqm = self._convert_bqm_from_qubo(Q)
            embedding = self._find_embedding(bqm)
            embed_bqm = self._embed_bqm(bqm, embedding)

            for num_reads in reads_per_iteration:
                # configを更新
                current_config = {**sample_config, "num_reads": num_reads}

                # サンプリング
                response = self.sampler.sample(embed_bqm, **current_config)

                # 埋め込みの解除
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

                # 結果を結合
                if combined_sampleset is None:
                    combined_sampleset = sampleset
                else:
                    combined_sampleset = combined_sampleset.aggregate()
                    new_sampleset = sampleset.aggregate()
                    # SampleSetを結合
                    combined_sampleset = concatenate(
                        [combined_sampleset, new_sampleset]
                    )

            # combined_samplesetがNoneでないことを確認
            if combined_sampleset is None:
                raise ValueError("Sampling failed, no valid sampleset was generated.")

            sampleset = combined_sampleset
        else:
            # 通常のサンプリング（10000以下）
            bqm = self._convert_bqm_from_qubo(Q)
            embedding = self._find_embedding(bqm)
            embed_bqm = self._embed_bqm(bqm, embedding)

            # サンプリング
            response = self.sampler.sample(embed_bqm, **sample_config)

            # 埋め込みの解除
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

        # Greedy で解を改善する (Optional)
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

        # 埋め込み失敗のチェック
        if not emb:
            raise ValueError("No embedding found.")

        # 型キャスト　(find_embeddingの戻り値がAnyなので)
        emb = cast(Mapping[int, list[int]], emb)

        # 埋め込みが全ての論理変数をカバーしているかのチェック（基本的にはこのif文にひっかからないはず）
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
        取得したサンプルをBinaryQuadraticModel上でSteepestDescentで
        局所改善するかどうか。デフォルト False。
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
        sampleset = cast(SampleSet, sampleset)  # 型キャスト

        # Greedy で解を改善する (Optional)
        if self.use_greedy:
            bqm = BinaryQuadraticModel.from_qubo(Q)
            sampleset = SteepestDescentSampler().sample(bqm, initial_states=sampleset)

        return sampleset


if __name__ == "__main__":
    # 例のQUBO
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

    load_dotenv(dotenv_path=".env.local")  # .env.local から環境変数を読み込む
    print("DWAVE_SOLVER_NAME:", os.getenv("DWAVE_SOLVER_NAME"))
    solver = QASolver()
    ss = solver.solve(Q)
    print("=== QA Result ===")
    print(ss)
    print()
