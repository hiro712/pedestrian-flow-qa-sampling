from src.solvers.base import SolverBase
from src.solvers.sa import SASolver
from src.solvers.sqa import SQASolver
from src.solvers.qa import QASolver


def get_solver(solver_type: str) -> SolverBase:
    """Return the solver instance corresponding to the given solver name."""
    if solver_type == "sa":
        return SASolver()
    elif solver_type == "sqa":
        return SQASolver()
    elif solver_type == "qa":
        return QASolver()
    else:
        raise ValueError(f"Unknown solver_type: {solver_type!r}. Choose from 'sa', 'sqa', 'qa'.")
