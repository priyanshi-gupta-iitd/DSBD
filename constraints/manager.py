"""Unified constraint manager for DSBD / AR / speculative sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from constraints.xgrammar_sql import XGrammarSQLConstraint
from constraints.z3_schema import SchemaFacts, Z3SchemaChecker


@dataclass
class ConstraintConfig:
    mode: str = "none"  # none | xgrammar | z3 | both

    @property
    def use_xgrammar(self) -> bool:
        return self.mode in ("xgrammar", "both")

    @property
    def use_z3(self) -> bool:
        return self.mode in ("z3", "both")


class ConstraintManager:
    """
    Owns optional xgrammar matchers (per beam) and a Z3 schema checker.
    Tracks reject counts and timing for goodput instrumentation.
    """

    def __init__(
        self,
        config: ConstraintConfig,
        tokenizer=None,
        schema_facts: Optional[SchemaFacts] = None,
        vocab_size: Optional[int] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.xgrammar: Optional[XGrammarSQLConstraint] = None
        self.z3: Optional[Z3SchemaChecker] = None
        self.beam_sql: List[str] = []  # decoded generation so far per beam

        if config.use_xgrammar:
            if tokenizer is None:
                raise ValueError("tokenizer required for xgrammar constraints")
            self.xgrammar = XGrammarSQLConstraint(tokenizer, vocab_size=vocab_size)
        if config.use_z3:
            if schema_facts is None:
                raise ValueError("schema_facts required for z3 constraints")
            self.z3 = Z3SchemaChecker(schema_facts)

    def reset(self, num_beams: int = 1):
        self.beam_sql = [""] * num_beams
        if self.xgrammar is not None:
            self.xgrammar.reset(num_beams)

    def sync_from_generated_ids(self, gen_token_ids: Sequence[int], num_beams: int = 1):
        """Reset matchers/SQL to the committed generation (prompt excluded)."""
        self.reset(1)
        sql = ""
        if self.xgrammar is not None:
            for tid in gen_token_ids:
                self.xgrammar.accept_token_on_beam(0, int(tid))
                sql += self.decode_token(tid)
        elif self.tokenizer is not None and gen_token_ids:
            sql = self.tokenizer.decode(list(gen_token_ids), skip_special_tokens=True)
        self.beam_sql = [sql]
        if num_beams > 1:
            self.ensure_beams(num_beams)
            if self.xgrammar is not None:
                self.xgrammar.fork_from([0] * num_beams)
            self.beam_sql = [sql] * num_beams

    def ensure_beams(self, num_beams: int):
        while len(self.beam_sql) < num_beams:
            self.beam_sql.append(self.beam_sql[-1] if self.beam_sql else "")
        self.beam_sql = self.beam_sql[:num_beams]
        if self.xgrammar is not None:
            self.xgrammar.ensure_beams(num_beams)

    def reorder(self, beam_idx: Sequence[int]):
        """Reorder beam state after beam selection (like cache reorder)."""
        idxs = [int(i) for i in beam_idx]
        self.beam_sql = [self.beam_sql[i] for i in idxs]
        if self.xgrammar is not None:
            self.xgrammar.fork_from(idxs)

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply syntactic bitmask. logits shape [num_beams, vocab]."""
        if self.xgrammar is None:
            return logits
        return self.xgrammar.apply_to_logits(logits)

    def mask_flat_beam_logits(self, flat_logits: torch.Tensor, num_beams: int, vocab_size: int) -> torch.Tensor:
        """
        Mask flattened [1, num_beams * vocab] or [num_beams * vocab] scores used by beam sampling.
        Operates on probability/log-prob tensors by reshaping.
        """
        if self.xgrammar is None:
            return flat_logits
        squeeze = False
        if flat_logits.dim() == 1:
            flat_logits = flat_logits.unsqueeze(0)
            squeeze = True
        # [1, B*V] -> [B, V]
        reshaped = flat_logits.view(num_beams, vocab_size).clone()
        self.mask_logits(reshaped)
        out = reshaped.view(1, num_beams * vocab_size)
        return out.squeeze(0) if squeeze else out

    def decode_token(self, token_id: int) -> str:
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode([int(token_id)], skip_special_tokens=True)

    def on_tokens_sampled(self, token_ids: Sequence[int], parent_beam_idx: Optional[Sequence[int]] = None):
        """Update matcher + decoded SQL after draft/AR sampling a token per beam."""
        if parent_beam_idx is not None:
            self.reorder(parent_beam_idx)
        self.ensure_beams(len(token_ids))
        if self.xgrammar is not None:
            self.xgrammar.accept_tokens(token_ids)
        for i, tid in enumerate(token_ids):
            self.beam_sql[i] = self.beam_sql[i] + self.decode_token(tid)

    def z3_allows(self, beam_idx: int, token_id: int) -> bool:
        if self.z3 is None:
            return True
        piece = self.decode_token(token_id)
        prefix = self.beam_sql[beam_idx] if beam_idx < len(self.beam_sql) else ""
        return self.z3.would_accept_token(prefix, piece)

    def z3_resample_flat(
        self,
        flat_ids: torch.Tensor,
        probs: torch.Tensor,
        vocab_size: int,
        max_tries: int = 8,
    ) -> torch.Tensor:
        """
        Resample draft flat indices (parent*vocab + tok) that fail the sound Z3 gate.
        Operates on the draft model only; call before committing tokens / on_tokens_sampled.
        ``beam_sql[parent]`` must still be the prefix for that parent beam.
        """
        if self.z3 is None:
            return flat_ids
        flat_ids = flat_ids.clone()
        squeeze = False
        if flat_ids.dim() == 1:
            flat_ids = flat_ids.unsqueeze(0)
            squeeze = True
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        probs = probs.clone()
        batch, k = flat_ids.shape
        for b in range(batch):
            for i in range(k):
                for _ in range(max_tries):
                    fid = int(flat_ids[b, i].item())
                    parent = fid // vocab_size
                    tok = fid % vocab_size
                    if self.z3_allows(parent, tok):
                        break
                    # force-reject this draft token and resample
                    probs[b, fid] = 0
                    s = probs[b].sum()
                    if s <= 0:
                        break
                    probs[b] = probs[b] / s
                    # sample a single replacement for this beam slot
                    new_fid = torch.multinomial(probs[b], num_samples=1)
                    flat_ids[b, i] = new_fid
        return flat_ids.squeeze(0) if squeeze else flat_ids

    def commit_token(self, beam_idx: int, token_id: int):
        """Commit an accepted token onto a beam (after verify)."""
        self.ensure_beams(max(beam_idx + 1, len(self.beam_sql)))
        if self.xgrammar is not None and beam_idx < len(self.xgrammar.matchers):
            # Matcher may already have accepted during draft; for target-only paths accept here.
            pass
        self.beam_sql[beam_idx] = self.beam_sql[beam_idx] + self.decode_token(token_id)

    def replace_beam_sql(self, beam_idx: int, sql: str):
        self.ensure_beams(beam_idx + 1)
        self.beam_sql[beam_idx] = sql

    def set_all_sql_from_token_ids(self, sequences: torch.Tensor, prompt_len: int):
        """Rebuild beam_sql from full sequences (prompt stripped)."""
        self.ensure_beams(sequences.size(0))
        for i in range(sequences.size(0)):
            gen = sequences[i, prompt_len:].tolist()
            self.beam_sql[i] = self.tokenizer.decode(gen, skip_special_tokens=True) if self.tokenizer else ""

    @property
    def xgrammar_time_ns(self) -> int:
        return 0 if self.xgrammar is None else self.xgrammar.time_ns

    @property
    def z3_time_ns(self) -> int:
        return 0 if self.z3 is None else self.z3.time_ns

    @property
    def xgrammar_rejects(self) -> int:
        return 0 if self.xgrammar is None else self.xgrammar.reject_count

    @property
    def z3_rejects(self) -> int:
        return 0 if self.z3 is None else self.z3.reject_count

    def stats_dict(self) -> Dict[str, Any]:
        return {
            "xgrammar_time_ns": self.xgrammar_time_ns,
            "z3_time_ns": self.z3_time_ns,
            "xgrammar_rejects": self.xgrammar_rejects,
            "z3_rejects": self.z3_rejects,
            "tokens_constraint_rejected": self.xgrammar_rejects + self.z3_rejects,
        }


def build_constraint_manager(
    mode: str,
    tokenizer=None,
    schema_facts: Optional[SchemaFacts] = None,
    vocab_size: Optional[int] = None,
) -> Optional[ConstraintManager]:
    mode = (mode or "none").lower()
    if mode == "none":
        return None
    return ConstraintManager(
        ConstraintConfig(mode=mode),
        tokenizer=tokenizer,
        schema_facts=schema_facts,
        vocab_size=vocab_size,
    )
