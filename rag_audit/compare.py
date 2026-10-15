"""Combine per-model summaries into one comparison table (CPU only).

    python -m rag_audit.compare results
"""
import json
import sys
from pathlib import Path

from .stats import mcnemar_exact


def _items(path, condition):
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["qid"]: r for r in rows if r["condition"] == condition}


def _pct(d):
    return f"{d['rate']:.1%} [{d['ci95'][0]:.0%}, {d['ci95'][1]:.0%}]" if d and d.get("n") else "n/a"


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = []
    for main_run in sorted(root.glob("full_*_n500")):
        key = main_run.name[len("full_"):-len("_n500")]
        s = json.loads((main_run / "summary.json").read_text(encoding="utf-8"))
        conflict_path = root / f"full_{key}_conflict_all" / "summary.json"
        c = json.loads(conflict_path.read_text(encoding="utf-8")) if conflict_path.exists() else None
        cond, paired = s["conditions"], s["paired"]
        cn = c["conditions"]["conflict"] if c else {}
        ret_path = root / f"full_{key}_retrieved" / "summary.json"
        ret = json.loads(ret_path.read_text(encoding="utf-8"))["conditions"]["retrieved"] if ret_path.exists() else {}
        clean = _items(main_run / "items.jsonl", "clean")
        retrieved = _items(root / f"full_{key}_retrieved" / "items.jsonl", "retrieved")
        qids = sorted(set(clean) & set(retrieved))
        ret_p = mcnemar_exact([clean[q]["correct"] for q in qids], [retrieved[q]["correct"] for q in qids])[2] \
            if qids else float("nan")
        rows.append({
            "model": s.get("model", key),
            "clean_correct": cond["clean"]["correct"],
            "noisy_correct": cond["noisy"]["correct"],
            "clean_vs_noisy_p": paired["clean_vs_noisy"]["mcnemar_p"],
            "order_changed": paired["order_sensitivity"]["answer_changed_with_order"],
            "missing_answered": cond["missing"]["answered_without_evidence"],
            "missing_wrong": cond["missing"]["unsupported_and_wrong"],
            "conflict_context": cn.get("follows_context"),
            "conflict_memory": cn.get("follows_memory"),
            "conflict_short_context": cn.get("short_follows_context"),
            "noisy_cite_precision": cond["noisy"]["cite_precision"]["mean"],
            "noisy_cite_recall": cond["noisy"]["cite_recall"]["mean"],
            "retrieved_correct": ret.get("correct"),
            "clean_vs_retrieved_p": ret_p,
            "all_gold_retrieved": ret.get("all_gold_retrieved"),
        })
    header = ("| Model | Clean correct | Noisy correct | Clean vs noisy p | BM25-retrieved correct | "
              "Clean vs retrieved p | Both gold passages retrieved | Answer changed by order | "
              "Answered with evidence missing | ...and wrong | Conflict: follows passage | "
              "Conflict: follows memory | Conflict, short answers: follows passage | Noisy citation P / R |")
    lines = [header, "|" + "---|" * 14]
    for r in rows:
        lines.append(
            f"| {r['model']} | {_pct(r['clean_correct'])} | {_pct(r['noisy_correct'])} | "
            f"{r['clean_vs_noisy_p']:.3g} | {_pct(r['retrieved_correct'])} | {r['clean_vs_retrieved_p']:.3g} | "
            f"{_pct(r['all_gold_retrieved'])} | {_pct(r['order_changed'])} | {_pct(r['missing_answered'])} | "
            f"{_pct(r['missing_wrong'])} | {_pct(r['conflict_context'])} | {_pct(r['conflict_memory'])} | "
            f"{_pct(r['conflict_short_context'])} | "
            f"{r['noisy_cite_precision']:.2f} / {r['noisy_cite_recall']:.2f} |"
        )
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
