"""Export Freebuff Desktop chat history (semantic-6g project) to a readable file.

Usage:
    python tools/export_freebuff_chat.py [project-dir] [out-file]

Defaults to the semantic-6g project's desktop-v2.db and writes
results/freebuff_chat_semantic-6g.txt next to the repo root.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJ = Path.home() / ".config" / "freebuff-desktop" / "projects"
DB = PROJ / "semantic-6g-91b0ccaf-86a2-497f-8224-d02ca9472eae" / "desktop-v2.db"
OUT = Path(__file__).resolve().parents[1] / "results" / "freebuff_chat_semantic-6g.txt"


def main() -> None:
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else DB
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    print("tables:", tables)

    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# Freebuff chat export — {db}", ""]

    for t in tables:
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
        rows = list(con.execute(f'SELECT * FROM "{t}"'))
        lines.append(f"## table {t} ({len(rows)} rows) cols={cols}")
        for row in rows:
            rec = dict(zip(cols, row))
            lines.append(json.dumps(rec, default=str, ensure_ascii=False))
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
