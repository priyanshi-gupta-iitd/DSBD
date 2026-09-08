"""Sound Z3 schema gate for (partial) Spider SQL.

Only reject when a *completed* identifier is impossible given the schema.
That is the smallest set that must not knock out a still-valid beam:

* completed FROM/JOIN table name that is not a schema table
* completed ``table.col`` / ``alias.col`` whose qualifier already resolves
  and whose column is not on that table

Never reject: incomplete BPE pieces, unknown qualifiers (alias may appear
later in FROM), bare names (aliases / AS), punctuation, keywords.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from time import process_time_ns
from typing import Dict, Iterator, List, Optional, Set, Tuple

from z3 import Bool, Solver, BoolVal, sat


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FROM_JOIN = re.compile(r"\b(?:from|join)\b", re.I)
_REGION_STOP = re.compile(
    r"\b(?:where|group|order|limit|having|union|except|intersect|on)\b", re.I
)
_KEYWORDS = {
    "select", "from", "where", "join", "inner", "left", "right", "outer", "on",
    "group", "by", "order", "asc", "desc", "limit", "as", "and", "or", "not",
    "in", "like", "is", "null", "count", "sum", "avg", "min", "max", "distinct",
    "having", "union", "all", "between", "exists", "case", "when", "then", "else",
    "end", "cast", "except", "intersect", "true", "false", "with",
}


def _ident_completed(sql: str, end: int) -> bool:
    """True iff the ident ending at ``end`` is closed by a delimiter (not EOS)."""
    if end >= len(sql):
        return False
    ch = sql[end]
    return not (ch.isalnum() or ch == "_")


@dataclass
class SchemaFacts:
    db_id: str
    tables: Set[str] = field(default_factory=set)
    columns: Dict[str, Set[str]] = field(default_factory=dict)
    column_to_tables: Dict[str, Set[str]] = field(default_factory=dict)
    types: Dict[Tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def from_spider_frames(cls, db_id: str, spider_schema, spider_primary=None, spider_foreign=None):
        """Build from DataFrames produced by evaluation.creatiing_schema."""
        facts = cls(db_id=db_id)
        df = spider_schema[spider_schema["Database name"] == db_id]
        for _, row in df.iterrows():
            table = str(row[" Table Name"]).strip()
            col = str(row[" Field Name"]).strip()
            typ = str(row[" Type"]).strip().lower() if " Type" in row else "text"
            if not table or table == "nan":
                continue
            tkey = table.lower()
            facts.tables.add(tkey)
            facts.columns.setdefault(tkey, set())
            ckey = col.lower()
            if ckey == "*":
                continue
            facts.columns[tkey].add(ckey)
            facts.column_to_tables.setdefault(ckey, set()).add(tkey)
            facts.types[(tkey, ckey)] = typ
        return facts


class Z3SchemaChecker:
    """
    Sound incremental semantic gate (see module docstring).
    Uses Z3 only to test the completed-identifier axioms.
    """

    def __init__(self, facts: SchemaFacts):
        self.facts = facts
        self.time_ns = 0
        self.reject_count = 0
        self._table_vars = {t: Bool(f"table_{t}") for t in facts.tables}
        self._col_vars = {}
        for t, cols in facts.columns.items():
            for c in cols:
                self._col_vars[(t, c)] = Bool(f"col_{t}__{c}")

    @staticmethod
    def _strip_literals(sql: str) -> str:
        return re.sub(r"'[^']*'", "''", sql)

    def _from_join_regions(self, sql: str) -> Iterator[Tuple[int, int]]:
        for m in _FROM_JOIN.finditer(sql):
            lo = m.end()
            stop = _REGION_STOP.search(sql, lo)
            hi = stop.start() if stop else len(sql)
            yield lo, hi

    def _split_commas(self, sql: str, lo: int, hi: int) -> List[Tuple[int, int]]:
        parts = []
        start = lo
        depth = 0
        for i in range(lo, hi):
            ch = sql[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                parts.append((start, i))
                start = i + 1
        parts.append((start, hi))
        return parts

    def completed_from_tables(self, sql: str) -> List[str]:
        """Completed table names in FROM/JOIN position (lowercased)."""
        out: List[str] = []
        for lo, hi in self._from_join_regions(sql):
            for a, b in self._split_commas(sql, lo, hi):
                idents = list(_IDENT.finditer(sql, a, b))
                if not idents:
                    continue
                tm = idents[0]
                name = tm.group(0).lower()
                if name in _KEYWORDS:
                    continue
                if _ident_completed(sql, tm.end()):
                    out.append(name)
        return out

    def alias_map(self, sql: str) -> Dict[str, str]:
        """Completed aliases in FROM/JOIN → table. Incomplete aliases omitted."""
        mapping: Dict[str, str] = {}
        for lo, hi in self._from_join_regions(sql):
            for a, b in self._split_commas(sql, lo, hi):
                idents = list(_IDENT.finditer(sql, a, b))
                if not idents:
                    continue
                tm = idents[0]
                table = tm.group(0).lower()
                if table in _KEYWORDS or not _ident_completed(sql, tm.end()):
                    continue
                rest = idents[1:]
                if rest and rest[0].group(0).lower() == "as":
                    rest = rest[1:]
                if not rest:
                    continue
                am = rest[0]
                alias = am.group(0).lower()
                if alias in _KEYWORDS:
                    continue
                if _ident_completed(sql, am.end()):
                    mapping[alias] = table
        return mapping

    def completed_qualified(self, sql: str) -> List[Tuple[str, str]]:
        """Completed ``qual.col`` pairs (both sides closed by a delimiter)."""
        refs = []
        for m in re.finditer(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)", sql
        ):
            if not _ident_completed(sql, m.end()):
                continue
            refs.append((m.group(1).lower(), m.group(2).lower()))
        return refs

    def check_partial_sql(self, sql: str) -> bool:
        t0 = process_time_ns()
        try:
            ok = self._check(sql)
            if not ok:
                self.reject_count += 1
            return ok
        finally:
            self.time_ns += process_time_ns() - t0

    def _check(self, sql: str) -> bool:
        if not sql or not sql.strip():
            return True
        sql = self._strip_literals(sql)
        solver = Solver()
        for t, v in self._table_vars.items():
            solver.add(v == True)  # noqa: E712
        for (_t, _c), v in self._col_vars.items():
            solver.add(v == True)  # noqa: E712

        for table in self.completed_from_tables(sql):
            if table not in self._table_vars:
                solver.add(BoolVal(False))
            else:
                solver.add(self._table_vars[table])

        aliases = self.alias_map(sql)
        for qual, col in self.completed_qualified(sql):
            if col in _KEYWORDS:
                continue
            if qual in aliases:
                table = aliases[qual]
            elif qual in self.facts.tables:
                table = qual
            else:
                # qualifier not yet bound (SELECT alias before FROM) — do not reject
                continue
            if (table, col) not in self._col_vars:
                solver.add(BoolVal(False))
            else:
                solver.add(self._col_vars[(table, col)])

        return solver.check() == sat

    def would_accept_token(self, prefix_sql: str, new_piece: str) -> bool:
        return self.check_partial_sql((prefix_sql or "") + (new_piece or ""))
