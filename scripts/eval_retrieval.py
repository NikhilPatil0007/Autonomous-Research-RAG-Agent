"""Compare dense, BM25, hybrid RRF, BM25 rerank, and chunk size.

  python scripts/eval_retrieval.py
  python scripts/eval_retrieval.py --ir-only
  python scripts/eval_retrieval.py --gold data/eval/gold.jsonl --k 5

Gold file: one JSON object per line with "question" and "reference".
Optional "reference_contexts": list of passages that count as relevant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.eval.run import run_eval
from app.observability import setup_logging
from app.tools import ensure_qdrant


def _print_table(rows: list[dict]) -> None:
    skip = {"n_relevant_in_corpus"}
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and key not in skip:
                keys.append(key)
    def cell(key: str, value: object) -> str:
        if isinstance(value, float):
            return f"{value:8.3f}"
        text = str(value)
        return f"{text:<22}" if key == "name" else f"{text:>12}"

    header = " | ".join(cell(key, key) for key in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(cell(key, row.get(key, "")) for key in keys))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval, rerank, and chunking")
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "data" / "eval" / "gold.jsonl",
        help="JSONL with question + reference",
    )
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument(
        "--ir-only",
        action="store_true",
        help="Precision/recall/hit@k only. Skip RAGAS (no extra Gemini judge calls).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "eval" / "results.json",
    )
    args = parser.parse_args()
    if not args.gold.exists():
        raise SystemExit(
            f"Missing {args.gold}. Copy data/eval/gold.example.jsonl to data/eval/gold.jsonl "
            "and fill reference answers from your PDFs."
        )
    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=settings.log_dir)
    ensure_qdrant()
    rows = run_eval(args.gold, k=args.k, use_ragas=not args.ir_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nk={args.k}  gold={args.gold}  ragas={not args.ir_only}\n")
    _print_table(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
