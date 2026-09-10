"""Goodput / throughput helpers for constrained DSBD evaluation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class GoodputStats:
    tokens_proposed: int = 0
    tokens_accepted: int = 0
    tokens_committed: int = 0
    tokens_constraint_rejected: int = 0
    xgrammar_rejects: int = 0
    z3_rejects: int = 0
    xgrammar_time_ns: int = 0
    z3_time_ns: int = 0
    wall_time_ns: int = 0
    n_useful: int = 0
    n_useful_correct: int = 0
    executable: bool = False
    exec_correct: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def throughput(self) -> float:
        if self.wall_time_ns <= 0:
            return 0.0
        return self.tokens_proposed / (self.wall_time_ns / 1e9)

    @property
    def goodput(self) -> float:
        if self.wall_time_ns <= 0:
            return 0.0
        return self.n_useful / (self.wall_time_ns / 1e9)

    @property
    def goodput_correct(self) -> float:
        if self.wall_time_ns <= 0:
            return 0.0
        return self.n_useful_correct / (self.wall_time_ns / 1e9)

    @property
    def useful_frac(self) -> float:
        if self.tokens_proposed <= 0:
            return 0.0
        return self.n_useful / self.tokens_proposed


def is_sql_executable(db: str, pred: str, db_root: str = "./spider/database") -> bool:
    """Return True if pred executes on the Spider sqlite DB (syntax+schema runnable)."""
    from sampling.utils import try_execute_sql

    ok, _ = try_execute_sql(db, pred, db_root)
    return ok


def compute_goodput_row(
    *,
    method: str,
    constraints: str,
    example_idx: int,
    db_id: Optional[str],
    pred_sql: str,
    reference: Optional[str],
    committed_tokens: int,
    tokens_proposed: int,
    tokens_accepted: int,
    tokens_constraint_rejected: int,
    xgrammar_rejects: int,
    z3_rejects: int,
    wall_time_ns: int,
    xgrammar_time_ns: int = 0,
    z3_time_ns: int = 0,
    acc_len_mean: float = 0.0,
    acc_rate: float = 0.0,
    exec_acc: Optional[float] = None,
    constraint_site: str = "draft",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one metrics row; N_useful counts committed tokens only if SQL is executable."""
    from sampling.utils import extract_sql, try_execute_sql

    pred_sql = extract_sql(pred_sql or "")
    executable = False
    exec_correct = False
    exec_error = ""
    if db_id is not None and pred_sql:
        executable, exec_error = try_execute_sql(db_id, pred_sql)
        if exec_acc is not None:
            exec_correct = exec_acc >= 1.0
        elif reference is not None and "[SQL]" in reference:
            from sampling.utils import execution_accuracy

            gt = reference.split("[SQL]", 1)[1]
            acc = execution_accuracy(db_id, pred_sql, gt)
            exec_correct = acc >= 1.0
            exec_acc = float(max(acc, 0))
    elif not pred_sql:
        exec_error = "empty_sql"

    n_useful = committed_tokens if executable else 0
    n_useful_correct = committed_tokens if exec_correct else 0
    wall_s = wall_time_ns / 1e9 if wall_time_ns > 0 else 0.0
    row = {
        "method": method,
        "constraints": constraints,
        "constraint_site": constraint_site,
        "example_idx": example_idx,
        "db_id": db_id,
        "pred_sql": pred_sql,
        "committed_tokens": committed_tokens,
        "tokens_proposed": tokens_proposed,
        "tokens_accepted": tokens_accepted,
        "tokens_constraint_rejected": tokens_constraint_rejected,
        "xgrammar_rejects": xgrammar_rejects,
        "z3_rejects": z3_rejects,
        "wall_time_s": wall_s,
        "xgrammar_time_s": xgrammar_time_ns / 1e9,
        "z3_time_s": z3_time_ns / 1e9,
        # throughput: all proposed tokens (draft beams + extras) / wall
        "throughput": (tokens_proposed / wall_s) if wall_s > 0 else 0.0,
        # main_throughput: final committed tokens only (target output) / wall
        # For AR this equals throughput; for DSBD it excludes rejected draft tokens.
        "main_throughput": (committed_tokens / wall_s) if wall_s > 0 else 0.0,
        "goodput": (n_useful / wall_s) if wall_s > 0 else 0.0,
        "goodput_correct": (n_useful_correct / wall_s) if wall_s > 0 else 0.0,
        "useful_frac": (n_useful / tokens_proposed) if tokens_proposed > 0 else 0.0,
        "n_useful": n_useful,
        "n_useful_correct": n_useful_correct,
        "executable": int(executable),
        "exec_error": exec_error,
        "exec_correct": int(exec_correct),
        "exec_acc": exec_acc if exec_acc is not None else (1.0 if exec_correct else 0.0),
        "acc_len_mean": acc_len_mean,
        "acc_rate": acc_rate,
    }
    if extra:
        row.update(extra)
    return row
