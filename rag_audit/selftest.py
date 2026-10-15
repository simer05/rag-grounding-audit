"""Quick CPU checks run at the start of every job, before any model is loaded."""
import random

from .data import ABSTAIN, build_items, perturb_number
from .prompting import build_prompt, parse_output
from .report import summarise
from .retrieve import BM25, build_retrieved_items
from .scoring import abstained, f1, matches, normalize, numbers, score_item
from .stats import mcnemar_exact, wilson


def _paris():
    passages = [
        {"title": "Paris", "text": "Paris is the capital of France.", "gold": True},
        {"title": "Berlin", "text": "Berlin is the capital of Germany.", "gold": False},
        {"title": "Rome", "text": "Rome is the capital of Italy.", "gold": False},
    ]
    return {"qid": "q1", "question": "What is the capital of France?", "answer": "Paris",
            "type": "bridge", "level": "easy", "passages": passages}


def _eiffel():
    passages = [
        {"title": "Eiffel Tower", "text": "The Eiffel Tower was completed in 1889.", "gold": True},
        {"title": "Big Ben", "text": "Big Ben's tower was completed in 1859.", "gold": False},
    ]
    return {"qid": "q2", "question": "When was the Eiffel Tower completed?", "answer": "1889",
            "type": "bridge", "level": "easy", "passages": passages}


def main():
    assert normalize("The  Eiffel-Tower!") == "eiffeltower"
    assert f1("the capital Paris", "Paris") > 0.6 and f1("Rome", "Paris") == 0.0
    assert matches("It is Paris.", "Paris") and not matches("Parisian", "Paris")
    assert matches("Brett Scallions", "Brett Allen Scallions")
    assert matches("June 12", "June 12, 2017") and not matches("June 12, 2016", "June 12, 2017")
    assert not matches("BBC News", "Sky News") and not matches("55%", "45%")
    assert numbers("1,200 people in the 16th year") == {"1200", "16"}
    assert abstained("NOT IN CONTEXT") and abstained("not in context.") and not abstained("Paris")
    assert abstained("Appenzeller Sennenhund is not mentioned, but the other is not mentioned either.")
    assert abstained("The passages don't contain this.") and not abstained("Shock cavalry")

    ans, cites = parse_output("Answer: Paris\nEvidence: [1, 3]")
    assert ans == "Paris" and cites == [1, 3]
    ans, cites = parse_output("Paris")
    assert ans == "Paris" and cites == []

    rng = random.Random(0)
    for _ in range(50):
        o = perturb_number("16th", rng)
        assert o and o != "16th" and o[-2:] in ("st", "nd", "rd", "th")
        d = perturb_number("June 12, 2017", rng)
        assert d.startswith("June 12, ") and 3 <= abs(int(d[-4:]) - 2017) <= 15
        c = perturb_number("1,200", rng)
        assert c != "1,200" and c.replace(",", "").isdigit()
    assert perturb_number("Paris", rng) is None

    p, lo, hi = wilson(5, 10)
    assert abs(p - 0.5) < 1e-9 and 0.18 < lo < 0.25 and 0.75 < hi < 0.82
    assert mcnemar_exact([True, True, False], [False, True, False])[:2] == (1, 0)

    bm25 = BM25(["Paris. Paris is the capital of France.", "Berlin. Berlin is the capital of Germany.",
                 "Eiffel Tower. The Eiffel Tower was completed in 1889."])
    assert bm25.search("What is the capital of France?", 1) == [0]
    assert bm25.search("When was the Eiffel Tower completed?", 1) == [2]
    ret = build_retrieved_items([_paris(), _eiffel()], k=2)
    assert [it["condition"] for it in ret] == ["retrieved", "retrieved"]
    assert ret[0]["passages"][0]["title"] == "Paris" and ret[0]["retrieval_recall"] == 1.0
    assert ret[1]["passages"][0]["title"] == "Eiffel Tower" and ret[1]["retrieval_recall"] == 1.0

    items = build_items([_paris(), _eiffel()], pool=[], seed=1)
    paris = {it["condition"]: it for it in items if it["qid"] == "q1"}
    eiffel = {it["condition"]: it for it in items if it["qid"] == "q2"}
    assert set(paris) == {"clean", "noisy", "noisy_reversed", "missing"}
    assert set(eiffel) == {"clean", "noisy", "noisy_reversed", "missing", "conflict"}
    assert len(paris["clean"]["passages"]) == 1 and len(paris["noisy"]["passages"]) == 3
    assert all(not p["gold"] for p in paris["missing"]["passages"])
    assert paris["missing"]["target"] == ABSTAIN
    assert [p["title"] for p in paris["noisy_reversed"]["passages"]] == \
        [p["title"] for p in paris["noisy"]["passages"]][::-1]
    sub = eiffel["conflict"]["target"]
    assert sub != "1889" and 3 <= abs(int(sub) - 1889) <= 15
    assert sub in eiffel["conflict"]["passages"][0]["text"]
    assert "1889" not in eiffel["conflict"]["passages"][0]["text"]
    assert "[1] Paris:" in build_prompt(paris["clean"])

    s_ctx = score_item(eiffel["conflict"], f"Answer: {sub}\nEvidence: [1]")
    s_mem = score_item(eiffel["conflict"], "Answer: 1889\nEvidence: [1]")
    assert s_ctx["follows_context"] and not s_ctx["follows_memory"]
    assert s_mem["follows_memory"] and not s_mem["follows_context"]

    rows = []
    for it, reply in [(paris["clean"], "Answer: Paris\nEvidence: [1]"),
                      (paris["noisy"], "Answer: Paris\nEvidence: [2]"),
                      (paris["noisy_reversed"], "Answer: Rome\nEvidence: [1]"),
                      (paris["missing"], "Answer: NOT IN CONTEXT\nEvidence: []")]:
        rows.append(dict(qid=it["qid"], condition=it["condition"], **score_item(it, reply)))
    by = {r["condition"]: r for r in rows}
    assert by["clean"]["correct"] and by["clean"]["cite_recall"] == 1.0
    assert by["missing"]["correct"] and not by["noisy_reversed"]["correct"]
    mem = score_item(paris["missing"], "Answer: Paris\nEvidence: []")
    assert not mem["correct"] and mem["answered_from_memory_correct"]
    s = summarise(rows)
    assert s["conditions"]["missing"]["answered_without_evidence"]["k"] == 0
    assert s["paired"]["order_sensitivity"]["answer_changed_with_order"]["k"] == 1
    print("selftest passed")


if __name__ == "__main__":
    main()
