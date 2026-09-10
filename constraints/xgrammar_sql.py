"""xgrammar-backed SQL syntactic constraint helpers."""

from __future__ import annotations

import os
from pathlib import Path
from time import process_time_ns
from typing import List, Optional, Sequence

import torch

_GRAMMAR_PATH = Path(__file__).with_name("sql_grammar.ebnf")


def load_sql_ebnf(path: Optional[str] = None) -> str:
    p = Path(path) if path else _GRAMMAR_PATH
    return p.read_text(encoding="utf-8")


class XGrammarSQLConstraint:
    """Per-request syntactic constraint using one GrammarMatcher per beam."""

    def __init__(self, tokenizer, vocab_size: Optional[int] = None, ebnf: Optional[str] = None):
        import xgrammar as xgr

        self.xgr = xgr
        self.tokenizer = tokenizer
        ebnf_str = ebnf or load_sql_ebnf()
        # Strip comment lines for from_ebnf
        cleaned = "\n".join(
            line for line in ebnf_str.splitlines() if line.strip() and not line.strip().startswith("#")
        )
        tok_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=vocab_size or getattr(tokenizer, "vocab_size", None),
        )
        compiler = xgr.GrammarCompiler(tok_info)
        self.compiled = compiler.compile_grammar(cleaned)
        self.vocab_size = tok_info.vocab_size
        self.matchers: List = []
        self.bitmask = None
        self.time_ns = 0
        self.reject_count = 0

    def reset(self, num_beams: int = 1):
        self.matchers = [self.xgr.GrammarMatcher(self.compiled) for _ in range(num_beams)]
        self.bitmask = self.xgr.allocate_token_bitmask(num_beams, self.vocab_size)

    def ensure_beams(self, num_beams: int):
        if len(self.matchers) == num_beams:
            return
        if not self.matchers:
            self.reset(num_beams)
            return
        # Grow/shrink by cloning root matcher state via fork from first when possible
        root = self.matchers[0]
        self.matchers = [root.fork() for _ in range(num_beams)]
        self.bitmask = self.xgr.allocate_token_bitmask(num_beams, self.vocab_size)

    def fork_from(self, parent_indices: Sequence[int]):
        """Replace matchers so matcher[i] continues from parent_indices[i]."""
        new_matchers = []
        for p in parent_indices:
            p = int(p)
            new_matchers.append(self.matchers[p].fork())
        self.matchers = new_matchers
        self.bitmask = self.xgr.allocate_token_bitmask(len(self.matchers), self.vocab_size)

    def fill_bitmasks(self) -> torch.Tensor:
        """Fill CPU bitmask for all beams; return tensor on CPU."""
        t0 = process_time_ns()
        for i, matcher in enumerate(self.matchers):
            matcher.fill_next_token_bitmask(self.bitmask, i)
        self.time_ns += process_time_ns() - t0
        return self.bitmask

    def apply_to_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Mask logits in-place for beams. logits: [num_beams, vocab].
        If the grammar allows nothing (or vocab sizes disagree), leave that row unmasked
        so sampling cannot emit an invalid CUDA index.
        """
        assert logits.dim() == 2
        num_beams = logits.size(0)
        self.ensure_beams(num_beams)
        bitmask = self.fill_bitmasks()
        t0 = process_time_ns()
        backup = logits.detach().clone()
        vocab = logits.size(-1)
        # apply only on the overlapping vocab prefix to avoid device-side OOB asserts
        apply_vocab = min(vocab, self.vocab_size)
        try:
            sl = logits[:, :apply_vocab].contiguous()
            self.xgr.apply_token_bitmask_inplace(sl, bitmask.to(logits.device) if logits.is_cuda else bitmask)
            logits[:, :apply_vocab] = sl
            if apply_vocab < vocab:
                logits[:, apply_vocab:] = torch.finfo(logits.dtype).min
        except Exception:
            logits.copy_(backup)
            self.time_ns += process_time_ns() - t0
            return logits
        # rows that became all -inf / nan → restore (empty grammar state)
        finite = torch.isfinite(logits)
        has_mass = finite.any(dim=-1)
        if not has_mass.all():
            logits[~has_mass] = backup[~has_mass]
        self.time_ns += process_time_ns() - t0
        return logits

    def accept_tokens(self, token_ids: Sequence[int]) -> List[bool]:
        """Advance each matcher by one token. Returns per-beam accept success."""
        t0 = process_time_ns()
        ok = []
        for matcher, tid in zip(self.matchers, token_ids):
            accepted = matcher.accept_token(int(tid))
            if not accepted:
                self.reject_count += 1
            ok.append(bool(accepted))
        self.time_ns += process_time_ns() - t0
        return ok

    def accept_token_on_beam(self, beam_idx: int, token_id: int) -> bool:
        t0 = process_time_ns()
        ok = self.matchers[beam_idx].accept_token(int(token_id))
        if not ok:
            self.reject_count += 1
        self.time_ns += process_time_ns() - t0
        return bool(ok)

    def would_accept_after(self, prefix_token_ids: Sequence[int], new_token_id: int) -> bool:
        """
        True iff replaying ``prefix_token_ids`` then ``new_token_id`` is legal.
        Used for main/verify-side checks without mutating live draft matchers.
        """
        t0 = process_time_ns()
        try:
            matcher = self.xgr.GrammarMatcher(self.compiled)
            for tid in prefix_token_ids:
                if not matcher.accept_token(int(tid)):
                    self.reject_count += 1
                    return False
            ok = bool(matcher.accept_token(int(new_token_id)))
            if not ok:
                self.reject_count += 1
            return ok
        finally:
            self.time_ns += process_time_ns() - t0

    def rollback_beams(self, num_tokens: int = 1):
        t0 = process_time_ns()
        for matcher in self.matchers:
            matcher.rollback(num_tokens)
        self.time_ns += process_time_ns() - t0

    def rollback_beam(self, beam_idx: int, num_tokens: int = 1):
        t0 = process_time_ns()
        self.matchers[beam_idx].rollback(num_tokens)
        self.time_ns += process_time_ns() - t0
