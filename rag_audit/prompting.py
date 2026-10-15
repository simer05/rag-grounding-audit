"""Prompt construction and output parsing."""
import re

INSTRUCTION = (
    "Answer the question using only the numbered passages below. "
    "If the passages do not contain the answer, reply exactly: NOT IN CONTEXT.\n"
    "Reply in exactly two lines:\n"
    "Answer: <short answer, a few words>\n"
    "Evidence: <numbers of the passages you used, for example [1, 3]>"
)

_ANSWER = re.compile(r"answer\s*:\s*(.*)", re.IGNORECASE)
_EVIDENCE = re.compile(r"evidence\s*:\s*(.*)", re.IGNORECASE)


def build_prompt(item):
    context = "\n\n".join(
        f"[{i}] {p['title']}: {p['text']}" for i, p in enumerate(item["passages"], 1)
    )
    return f"{INSTRUCTION}\n\nPassages:\n{context}\n\nQuestion: {item['question']}"


def parse_output(text):
    """Return (answer, cited passage numbers) from a model reply."""
    answer, cites = None, []
    for line in text.splitlines():
        if answer is None:
            m = _ANSWER.search(line)
            if m:
                answer = m.group(1).strip()
                continue
        m = _EVIDENCE.search(line)
        if m:
            cites = sorted({int(x) for x in re.findall(r"\d+", m.group(1))})
            break
    if answer is None:
        lines = [l for l in text.strip().splitlines() if l.strip()]
        answer = lines[0].strip() if lines else ""
    return answer, cites
