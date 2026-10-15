"""BM25 retrieval over a pooled passage corpus (pure Python, no extra packages).

The `retrieved` condition replaces the dataset's given passages with what a real retriever finds:
all passages of all sampled questions are pooled into one corpus and each question retrieves its
top-k passages. This measures the full RAG pipeline: retrieval recall and answer quality together.
"""
import math
import re
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the of in on at to for and or is was were are be been by with as from that this which who whom "
    "what when where how did does do it its his her their they he she than then also into".split()
)


def tokenize(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        tfs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(tf.values()) for tf in tfs]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter()
        for tf in tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.postings = defaultdict(list)
        for i, tf in enumerate(tfs):
            for t, f in tf.items():
                self.postings[t].append((i, f))
        self.k1, self.b = k1, b

    def search(self, query, k):
        scores = defaultdict(float)
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                norm = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                scores[i] += idf * f * (self.k1 + 1) / norm
        return sorted(scores, key=lambda i: (-scores[i], i))[:k]


def build_retrieved_items(records, k=5):
    """One `retrieved` item per question, using BM25 top-k over the pooled corpus."""
    corpus, seen = [], set()
    for r in records:
        for p in r["passages"]:
            key = (p["title"], p["text"])
            if key not in seen:
                seen.add(key)
                corpus.append({"title": p["title"], "text": p["text"]})
    bm25 = BM25([f"{p['title']}. {p['text']}" for p in corpus])
    items = []
    for r in records:
        gold_titles = {p["title"] for p in r["passages"] if p["gold"]}
        hits = [corpus[i] for i in bm25.search(r["question"], k)]
        passages = [dict(p, gold=p["title"] in gold_titles) for p in hits]
        found = {p["title"] for p in passages if p["gold"]}
        items.append({
            **{key: r[key] for key in ("qid", "question", "answer", "type", "level")},
            "condition": "retrieved",
            "passages": passages,
            "target": r["answer"],
            "retrieval_recall": len(found) / len(gold_titles),
        })
    return items
