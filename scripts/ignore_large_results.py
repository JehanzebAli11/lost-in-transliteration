"""Append results/*.jsonl files larger than a threshold to the managed block in .gitignore.

Raw results are committed by default (they back every number in the write-up). GitHub
rejects files over 100 MB, so anything over 50 MB is git-ignored instead and should be
compressed or attached to a release by the owner.

    python scripts/ignore_large_results.py [--mb 50] [--root .]
"""
from __future__ import annotations

import argparse
from pathlib import Path

START = "# >>> large-results (managed by scripts/ignore_large_results.py) >>>"
END = "# <<< large-results <<<"


def update(root: Path, mb: float = 50.0) -> list[str]:
    gi = root / ".gitignore"
    text = gi.read_text(encoding="utf-8")
    if START not in text or END not in text:
        text = text.rstrip("\n") + f"\n{START}\n{END}\n"
    head, rest = text.split(START, 1)
    block, tail = rest.split(END, 1)
    current = {l.strip() for l in block.splitlines() if l.strip()}
    big = sorted(str(p.relative_to(root)) for p in (root / "results").rglob("*.jsonl")
                 if p.stat().st_size > mb * 1024 * 1024)
    entries = sorted(current | set(big))
    gi.write_text(head + START + "\n" + "".join(e + "\n" for e in entries) + END + tail, encoding="utf-8")
    return [b for b in big if b not in current]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mb", type=float, default=50.0)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args()
    added = update(Path(a.root), a.mb)
    print("newly ignored: " + (", ".join(added) if added else "none"))
