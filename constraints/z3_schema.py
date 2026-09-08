"""Z3-backed schema semantic checks for (partial) Spider SQL."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from time import process_time_ns
from typing import Dict, Iterable, List, Optional, Set, Tuple

from z3 import Bool, Solver, And, Or, Not, BoolVal, sat


_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_KEYWORDS = {
    "select", "from", "where", "join", "inner", "left", "right", "outer", "on",
    "group", "by", "order", "asc", "desc", "limit", "as", "and", "or", "not",
    "in", "like", "is", "null", "count", "sum", "avg", "min", "max", "distinct",
    "having", "union", "all", "between", "exists", "case", "when", "then", "else",
    "end", "cast", "except", "intersect", "true", "false",
}


@dataclass
class SchemaFacts:
    db_id: str
    tables: Set[str] = field(default_factory=set)
    # lowercased table -> set of lowercased columns
    columns: Dict[str, Set[str]] = field(default_factory=dict)
    # column -> set of tables containing it
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
    Incremental semantic gate: completed identifiers in FROM/JOIN must be tables;
    completed column refs must exist in schema (and in-scope tables when known).
    Uses Z3 for the satisfiability check over schema facts.
    """

    def __init__(self, facts: SchemaFacts):
        self.facts = facts
        self.time_ns = 0
        self.reject_count = 0
        # Prebuild Z3 atoms
        self._table_vars = {t: Bool(f"table_{t}") for t in facts.tables}
        self._col_vars = {}
        for t, cols in facts.columns.items():
            for c in cols:
                self._col_vars[(t, c)] = Bool(f"col_{t}__{c}")

    @staticmethod
    def _strip_literals(sql: str) -> str:
        """Replace quoted strings so literal words are not treated as identifiers."""
        return re.sub(r"'[^']*'", "''", sql)

    def _completed_tokens(self, sql: str) -> List[str]:
        """Return identifier-like tokens; drop trailing incomplete fragment."""
        sql = self._strip_literals(sql)
        # If ends with alphanumeric without delimiter, last ident may be incomplete —
        # still validate fully delimited prior idents only.
        ends_incomplete = bool(sql) and (sql[-1].isalnum() or sql[-1] == "_")
        idents = _IDENT.findall(sql)
        if ends_incomplete and idents:
            idents = idents[:-1]
        return idents

    def _extract_from_tables(self, sql: str) -> Set[str]:
        """Heuristic: names after FROM / JOIN until WHERE/GROUP/ORDER/LIMIT/ON."""
        low = self._strip_literals(sql).lower()
        tables: Set[str] = set()
        for m in re.finditer(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", low):
            tables.add(m.group(1))
        return tables

    def _extract_column_refs(self, sql: str) -> List[Tuple[Optional[str], str]]:
        """Return (table_or_None, column) for qualified and bare column mentions outside keywords."""
        refs = []
        low = self._strip_literals(sql)
        # qualified
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", low):
            refs.append((m.group(1).lower(), m.group(2).lower()))
        # bare names that are not keywords / known tables — checked softly
        for m in _IDENT.finditer(low):
            name = m.group(1).lower()
            if name in _KEYWORDS:
                continue
            # skip if this match is the table side of qualified already handled
            if m.end() < len(low) and low[m.end()] == ".":
                continue
            # skip if previous char was '.'
            if m.start() > 0 and low[m.start() - 1] == ".":
                continue
            refs.append((None, name))
        return refs

    def check_partial_sql(self, sql: str) -> bool:
        """
        Return True if partial SQL does not yet violate schema constraints.
        Incomplete trailing identifiers are ignored.
        """
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

        solver = Solver()
        # Schema axioms: existing tables/columns are True; others False via absence
        for t, v in self._table_vars.items():
            solver.add(v == True)  # noqa: E712 — Z3 Bool
        for (t, c), v in self._col_vars.items():
            solver.add(v == True)  # noqa: E712

        completed = {x.lower() for x in self._completed_tokens(sql)}
        from_tables = self._extract_from_tables(sql)
        # Only validate FROM/JOIN tables that are completed identifiers
        for t in from_tables:
            if t not in completed and t not in self.facts.tables:
                # still check — extract_from_tables only gets full matches
                pass
            if t in _KEYWORDS:
                continue
            if t not in self._table_vars:
                # Unknown table: unsat
                solver.add(BoolVal(False))
            else:
                solver.add(self._table_vars[t])

        # In-scope tables for column checks
        scope = {t for t in from_tables if t in self.facts.tables}
        # If no FROM yet, only reject clearly unknown qualified refs; bare names deferred
        refs = self._extract_column_refs(sql)
        # Only check refs whose column token is completed
        for table, col in refs:
            if col not in completed and table is None:
                continue
            if col in _KEYWORDS:
                continue
            if col in self.facts.tables and table is None:
                # bare table name appearing elsewhere is ok
                continue
            if table is not None:
                if table not in self._table_vars:
                    solver.add(BoolVal(False))
                    continue
                if (table, col) not in self._col_vars:
                    solver.add(BoolVal(False))
                else:
                    solver.add(self._col_vars[(table, col)])
            else:
                # bare column: must exist in some in-scope table, or any table if scope empty
                candidates = []
                search_tables = scope if scope else self.facts.tables
                for t in search_tables:
                    if (t, col) in self._col_vars:
                        candidates.append(self._col_vars[(t, col)])
                if not candidates:
                    # Unknown column name
                    if col not in self.facts.column_to_tables:
                        solver.add(BoolVal(False))
                else:
                    solver.add(Or(*candidates) if len(candidates) > 1 else candidates[0])

        return solver.check() == sat

    def would_accept_token(self, prefix_sql: str, new_piece: str) -> bool:
        return self.check_partial_sql(prefix_sql + new_piece)
