"""Aggregate scored rows into a summary with confidence intervals and paired tests."""
from collections import defaultdict

from .scoring import F1_CORRECT, f1, normalize, numbers
from .stats import mcnemar_exact, wilson


def same_answer(a, b):
    """Same answer: both abstained, or token F1 >= 0.6 with identical numbers."""
    if a["abstained"] or b["abstained"]:
        return a["abstained"] and b["abstained"]
    return f1(a["pred"], b["pred"]) >= F1_CORRECT and numbers(a["pred"]) == numbers(b["pred"])


def _rate(rows, key):
    k = sum(1 for r in rows if r.get(key))
    p, lo, hi = wilson(k, len(rows))
    return {"k": k, "n": len(rows), "rate": p, "ci95": [lo, hi]}


def _mean(rows, key):
    vals = [r[key] for r in rows if key in r]
    return {"n": len(vals), "mean": sum(vals) / len(vals) if vals else float("nan")}


def summarise(rows):
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    summary = {"conditions": {}, "paired": {}}
    for cond, rs in by_cond.items():
        s = {"n": len(rs), "correct": _rate(rs, "correct"), "abstained": _rate(rs, "abstained")}
        if cond in ("clean", "noisy", "noisy_reversed", "retrieved"):
            s.update(em=_mean(rs, "em"), f1=_mean(rs, "f1"),
                     cite_precision=_mean(rs, "cite_precision"), cite_recall=_mean(rs, "cite_recall"))
        if cond == "retrieved":
            full = [r for r in rs if r.get("retrieval_recall") == 1.0]
            s["retrieval_recall"] = _mean(rs, "retrieval_recall")
            s["all_gold_retrieved"] = _rate([dict(r, g=r.get("retrieval_recall") == 1.0) for r in rs], "g")
            s["correct_when_all_gold_retrieved"] = _rate(full, "correct")
        if cond == "missing":
            answered = [r for r in rs if not r["abstained"]]
            s["answered_without_evidence"] = _rate([dict(r, h=not r["abstained"]) for r in rs], "h")
            s["unsupported_and_wrong"] = _rate(
                [dict(r, w=not r["abstained"] and not r.get("answered_from_memory_correct")) for r in rs], "w")
            s["memory_correct_when_answered"] = _rate(answered, "answered_from_memory_correct")
        if cond == "conflict":
            s.update(follows_context=_rate(rs, "follows_context"), follows_memory=_rate(rs, "follows_memory"))
            # Robustness subset: short answers (<= 3 tokens), where the edited number is the answer itself
            # rather than one number inside a longer phrase or sentence.
            short = [r for r in rs if r.get("answer") and len(normalize(r["answer"]).split()) <= 3]
            s.update(short_answer_n=len(short),
                     short_follows_context=_rate(short, "follows_context"),
                     short_follows_memory=_rate(short, "follows_memory"))
        summary["conditions"][cond] = s

    def paired(a, b, key="correct"):
        ra = {r["qid"]: r for r in by_cond.get(a, [])}
        rb = {r["qid"]: r for r in by_cond.get(b, [])}
        qids = sorted(set(ra) & set(rb))
        a_only, b_only, p = mcnemar_exact([ra[q][key] for q in qids], [rb[q][key] for q in qids])
        return qids, ra, rb, {"n_pairs": len(qids), f"{a}_only": a_only, f"{b}_only": b_only, "mcnemar_p": p}

    if "clean" in by_cond and "noisy" in by_cond:
        summary["paired"]["clean_vs_noisy"] = paired("clean", "noisy")[3]
    if "noisy" in by_cond and "noisy_reversed" in by_cond:
        qids, ra, rb, test = paired("noisy", "noisy_reversed")
        same = sum(1 for q in qids if same_answer(ra[q], rb[q]))
        p, lo, hi = wilson(len(qids) - same, len(qids))
        test["answer_changed_with_order"] = {"k": len(qids) - same, "n": len(qids), "rate": p, "ci95": [lo, hi]}
        summary["paired"]["order_sensitivity"] = test
    return summary


def _fmt(d):
    return f"{d['rate']:.1%} [{d['ci95'][0]:.1%}, {d['ci95'][1]:.1%}] ({d['k']}/{d['n']})"


def to_text(summary, header=""):
    lines = [header] if header else []
    for cond, s in summary["conditions"].items():
        lines.append(f"\n== {cond} (n={s['n']})")
        lines.append(f"  correct     {_fmt(s['correct'])}")
        lines.append(f"  abstained   {_fmt(s['abstained'])}")
        for key in ("em", "f1", "cite_precision", "cite_recall", "retrieval_recall"):
            if key in s:
                lines.append(f"  {key:<11} {s[key]['mean']:.3f} (n={s[key]['n']})")
        for key in ("answered_without_evidence", "unsupported_and_wrong", "memory_correct_when_answered",
                    "follows_context", "follows_memory", "short_follows_context", "short_follows_memory",
                    "all_gold_retrieved", "correct_when_all_gold_retrieved"):
            if key in s:
                lines.append(f"  {key:<15} {_fmt(s[key])}")
    for name, t in summary["paired"].items():
        lines.append(f"\n== paired: {name}")
        for k, v in t.items():
            lines.append(f"  {k:<26} {_fmt(v) if isinstance(v, dict) else v}")
    return "\n".join(lines)
