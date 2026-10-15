"""Per-item scoring. Answer normalisation follows the SQuAD / HotpotQA evaluation scripts."""
import re
import string
from collections import Counter

from .data import ABSTAIN
from .prompting import parse_output


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


F1_CORRECT = 0.6
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(s):
    """Numbers in a string, with thousands separators removed: '1,200 and 16th' -> {'1200', '16'}."""
    return {n.replace(",", "") for n in _NUMBER.findall(s)}


def matches(pred, gold):
    """Exact match, the full gold answer appearing as whole words in the prediction,
    or token F1 >= 0.6 with no number in the prediction that the gold answer lacks.
    Accepts 'Brett Scallions' for 'Brett Allen Scallions' and 'June 12' for 'June 12, 2017';
    rejects 'BBC News' for 'Sky News' and 'June 12, 2016' for 'June 12, 2017'."""
    np_, ng = normalize(pred), normalize(gold)
    if not ng:
        return False
    if np_ == ng or f" {ng} " in f" {np_} ":
        return True
    return f1(pred, gold) >= F1_CORRECT and numbers(pred) <= numbers(gold)


# Normalised phrases that signal the model declined to answer from the passages
# (normalisation lower-cases, strips punctuation and drops articles: "doesn't" -> "doesnt").
ABSTAIN_PHRASES = (
    normalize(ABSTAIN), "not mentioned", "not provided", "not specified", "not stated", "not given",
    "not available", "does not contain", "do not contain", "doesnt contain", "dont contain",
    "does not mention", "do not mention", "doesnt mention", "no information", "not enough information",
    "cannot be determined", "cant be determined", "cannot determine", "unable to determine",
)


def abstained(pred):
    p = normalize(pred)
    return any(phrase in p for phrase in ABSTAIN_PHRASES)


def score_item(item, output):
    pred, cites = parse_output(output)
    gold_ids = [i for i, p in enumerate(item["passages"], 1) if p["gold"]]
    out = {
        "pred": pred,
        "cites": cites,
        "abstained": abstained(pred),
        "em": exact_match(pred, item["answer"]),
        "f1": f1(pred, item["answer"]),
    }
    cond = item["condition"]
    if cond == "missing":
        # Answering here is ungrounded; it may still be right from parametric memory.
        out["correct"] = out["abstained"]
        out["answered_from_memory_correct"] = not out["abstained"] and matches(pred, item["answer"])
    elif cond == "conflict":
        # The conflict edit changes one number, so compare that number directly.
        new_vals = numbers(item["target"]) - numbers(item["answer"])
        old_vals = numbers(item["answer"]) - numbers(item["target"])
        got = numbers(pred)
        ctx = not out["abstained"] and bool(new_vals) and new_vals <= got
        mem = not out["abstained"] and not ctx and bool(old_vals) and old_vals <= got
        out.update(correct=ctx, follows_context=ctx, follows_memory=mem)
    else:
        out["correct"] = not out["abstained"] and matches(pred, item["answer"])
    if cites and gold_ids:
        hit = len(set(cites) & set(gold_ids))
        out["cite_precision"] = hit / len(cites)
        out["cite_recall"] = hit / len(gold_ids)
    return out
