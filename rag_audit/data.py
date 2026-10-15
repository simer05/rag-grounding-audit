"""Load HotpotQA (distractor setting) and build paired audit conditions.

Every question is shown to the model under several conditions that differ only in the
evidence it receives, so behaviour can be compared within the same question:

    clean           only the gold (supporting) passages
    noisy           gold passages mixed with the dataset's 8 distractor passages, shuffled
    noisy_reversed  the same noisy passages in reverse order (order sensitivity)
    missing         the distractors only; the correct behaviour is to abstain
    conflict        gold passages with a numeric answer (date, year, count, ordinal) changed to a
                    different value of the same type; the grounded behaviour is to follow the
                    passages, not parametric memory. Only questions with numeric answers qualify.
"""
import random
import re

import pandas as pd

ABSTAIN = "NOT IN CONTEXT"
CONDITIONS = ("clean", "noisy", "noisy_reversed", "missing", "conflict")


def load_hotpot(path, n, seed):
    """Return (records, answer_pool). Records are a fixed random sample of n questions."""
    df = pd.read_parquet(path)
    pool = [str(a) for a in df["answer"].tolist()]
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(df)), n)) if n < len(df) else list(range(len(df)))
    records = []
    for i in idx:
        row = df.iloc[i]
        titles = [str(t) for t in row["context"]["title"]]
        sentences = [[str(s) for s in sents] for sents in row["context"]["sentences"]]
        gold_titles = {str(t) for t in row["supporting_facts"]["title"]}
        passages = [
            {"title": t, "text": "".join(s).strip(), "gold": t in gold_titles}
            for t, s in zip(titles, sentences)
        ]
        if not any(p["gold"] for p in passages):
            continue
        records.append({
            "qid": str(row["id"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "type": str(row["type"]),
            "level": str(row["level"]),
            "passages": passages,
        })
    return records, pool


def _rng(seed, qid, tag):
    return random.Random(f"{seed}:{qid}:{tag}")


MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december")
_NUM = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(st|nd|rd|th)?\b", re.IGNORECASE)
_FACTORS = (0.6, 0.7, 1.3, 1.5, 1.8)


def _ordinal(n):
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def perturb_number(answer, rng):
    """Change one number in a numeric answer while keeping its type and format.

    A year moves by 3-15 years, a day of the month stays a valid day, an ordinal keeps a correct
    suffix, and other quantities are scaled. Substitution is therefore type-safe by construction
    (a date stays a date, a count stays a count), which random answer swapping cannot guarantee.
    """
    found = list(_NUM.finditer(answer))
    if not found:
        return None
    years = [m for m in found if re.fullmatch(r"1\d{3}|20\d{2}", m.group(1))]
    m = years[-1] if years else found[0]
    raw, suffix = m.group(1), m.group(2)
    if years:
        new = str(int(raw) + rng.choice((-1, 1)) * rng.randint(3, 15))
    elif "." in raw:
        decimals = len(raw.split(".")[1])
        new = f"{float(raw) * rng.choice(_FACTORS):.{decimals}f}"
    else:
        v = int(raw.replace(",", ""))
        if any(mo in answer.lower() for mo in MONTHS) and v <= 31:
            nv = (v + rng.randint(3, 12)) % 28 + 1
        elif v < 10:
            nv = v + rng.randint(1, 5)
        else:
            nv = int(round(v * rng.choice(_FACTORS)))
        if nv == v:
            nv = v + 1
        new = f"{nv:,}" if "," in raw else str(nv)
    if suffix:
        new += _ordinal(int(new.replace(",", "").split(".")[0]))
    out = answer[:m.start()] + new + answer[m.end():]
    return out if out != answer else None


def pick_substitute(answer, passages, rng, tries=5):
    """A type-safe numeric substitute that does not already occur in any passage."""
    if answer.strip().lower() in ("yes", "no") or not re.search(r"\d", answer):
        return None
    joined = " ".join(p["text"] for p in passages).lower()
    for _ in range(tries):
        sub = perturb_number(answer, rng)
        if sub and sub.lower() not in joined:
            return sub
    return None


def _conflict_passages(record, pool, seed):
    gold = [p for p in record["passages"] if p["gold"]]
    answer = record["answer"]
    pattern = re.compile(re.escape(answer), re.IGNORECASE)
    if not any(pattern.search(p["text"]) for p in gold):
        return None, None
    sub = pick_substitute(answer, record["passages"], _rng(seed, record["qid"], "sub"))
    if sub is None:
        return None, None
    # Titles are edited too: in HotpotQA the answer is often the title of a gold passage.
    edited = [dict(p, title=pattern.sub(sub, p["title"]), text=pattern.sub(sub, p["text"])) for p in gold]
    return edited, sub


def build_items(records, pool, seed, conditions=CONDITIONS):
    items = []
    for r in records:
        base = {k: r[k] for k in ("qid", "question", "answer", "type", "level")}
        gold = [p for p in r["passages"] if p["gold"]]
        noisy = list(r["passages"])
        _rng(seed, r["qid"], "order").shuffle(noisy)
        built = {
            "clean": (gold, r["answer"]),
            "noisy": (noisy, r["answer"]),
            "noisy_reversed": (noisy[::-1], r["answer"]),
            "missing": ([p for p in noisy if not p["gold"]], ABSTAIN),
        }
        edited, sub = _conflict_passages(r, pool, seed)
        if edited is not None:
            built["conflict"] = (edited, sub)
        for cond in conditions:
            if cond in built:
                passages, target = built[cond]
                items.append(dict(base, condition=cond, passages=passages, target=target))
    return items
