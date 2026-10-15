"""Re-score an existing run from its saved replies (no model, CPU only).

    python -m rag_audit.rescore results/<run>
"""
import json
import sys
from pathlib import Path

from .report import summarise, to_text
from .scoring import score_item

KEEP = ("qid", "condition", "question", "answer", "target", "type", "level", "gold_ids", "n_passages", "reply")


def main():
    run = Path(sys.argv[1])
    rows_in = [json.loads(l) for l in (run / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for r in rows_in:
        item = {k: r[k] for k in ("condition", "answer", "target")}
        item["passages"] = [{"gold": i in r["gold_ids"]} for i in range(1, r["n_passages"] + 1)]
        row = {k: r[k] for k in KEEP}
        if "retrieval_recall" in r:
            row["retrieval_recall"] = r["retrieval_recall"]
        row.update(score_item(item, r["reply"]))
        rows.append(row)
    with open(run / "items.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    old = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary = summarise(rows)
    summary["model"] = old.get("model")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = to_text(summary, header=f"model={summary['model']} (rescored)")
    (run / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
