#!/usr/bin/env python3
"""Soundness checks for the reduced Z3 schema gate (no GPU)."""

from constraints.z3_schema import SchemaFacts, Z3SchemaChecker


def _concert_facts() -> SchemaFacts:
    facts = SchemaFacts(db_id="concert_singer")
    facts.tables = {"singer", "stadium", "concert", "singer_in_concert"}
    facts.columns = {
        "singer": {"singer_id", "name", "country", "song_name", "song_release_year", "age"},
        "stadium": {"stadium_id", "location", "name", "capacity", "highest", "lowest", "average"},
        "concert": {"concert_id", "concert_name", "theme", "stadium_id", "year"},
        "singer_in_concert": {"concert_id", "singer_id"},
    }
    for t, cols in facts.columns.items():
        for c in cols:
            facts.column_to_tables.setdefault(c, set()).add(t)
    return facts


def _ok(z: Z3SchemaChecker, sql: str, expect: bool, label: str):
    got = z.check_partial_sql(sql)
    assert got is expect, f"{label}: {sql!r} -> {got} (want {expect})"


def main():
    z = Z3SchemaChecker(_concert_facts())

    # incomplete BPE / mid-name must pass (old gate failed here)
    _ok(z, "SELECT stadium.Name FROM concert JOIN stadium ON concert.Stadium", True, "mid Stadium_ID")
    _ok(z, "SELECT * FROM singer_in", True, "mid table name")
    _ok(z, "SELECT * FROM Has", True, "incomplete unknown table")
    _ok(z, "SELECT singer.Name,singer.S", True, "incomplete qualified col")

    # completed illegal table
    _ok(z, "SELECT * FROM nosuch ", False, "unknown completed table")
    _ok(z, "SELECT * FROM singer ", True, "known completed table")
    _ok(z, "SELECT * FROM singer_in_concert ", True, "underscore table")
    _ok(z, "SELECT * FROM concert, stadium ", True, "comma join")
    _ok(z, "SELECT * FROM concert, nosuch ", False, "comma unknown table")
    _ok(z, "SELECT * FROM concert JOIN stadium ON stadium.Capacity ", True, "join on col")

    # completed illegal / legal qualified column
    _ok(z, "SELECT * FROM concert WHERE concert.Stadium ", False, "Stadium is not a column")
    _ok(z, "SELECT * FROM concert WHERE concert.Stadium_ID ", True, "Stadium_ID ok")
    _ok(z, "SELECT * FROM concert WHERE concert.year ", True, "year ok")

    # alias: do not reject AS / short alias; resolve only when completed
    _ok(z, "SELECT AVG(Age) AS average_age ", True, "AS alias")
    _ok(z, "SELECT s.Name FROM singer s ", True, "alias.col after FROM")
    _ok(z, "SELECT s.Name FROM singer s WHERE s.Foo ", False, "bad alias.col")
    _ok(z, "SELECT s.Name FROM singer", True, "alias before FROM completes")

    # punctuation must not be treated as a schema fact
    _ok(z, "SELECT * FROM singer!", True, "bang after valid table")
    _ok(z, "SELECT name FROM singer WHERE age > 20", True, "plain valid")

    # token API: space completes, underscore does not
    assert z.would_accept_token("SELECT * FROM concert WHERE concert.Stadium", " ") is False
    assert z.would_accept_token("SELECT * FROM concert WHERE concert.Stadium", "_") is True
    assert z.would_accept_token("SELECT * FROM singer", " ") is True
    assert z.would_accept_token("SELECT * FROM nosuch", " ") is False

    print("test_z3_sound: ok")


if __name__ == "__main__":
    main()
