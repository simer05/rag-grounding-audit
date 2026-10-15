# Behavioural Auditing of Grounding in Retrieval-Augmented LLMs

Retrieval-augmented generation (RAG) is meant to make LLM answers depend on evidence. This project measures
whether they do, with a paired, controlled-truth design: each question is posed several times with different
evidence, so every behaviour is compared within the same question.

## Conditions

| Condition | Evidence shown | Grounded behaviour |
|---|---|---|
| `clean` | gold supporting passages only | correct answer |
| `noisy` | gold passages + 8 distractors (shuffled) | correct answer |
| `noisy_reversed` | the same passages in reverse order | same answer as `noisy` |
| `missing` | distractors only | abstain (`NOT IN CONTEXT`) |
| `conflict` | gold passages with a numeric answer (date, year, count, ordinal) changed to another value of the same type | follow the passages |
| `retrieved` | BM25 top-5 over the pooled corpus of all sampled questions' passages (end-to-end RAG) | correct answer |

Conflict edits are type-safe by construction (a year moves by 3–15 years, a day stays a valid day, an ordinal
keeps a correct suffix), so only questions with numeric answers qualify; the conflict condition is therefore run
over every validation question with a numeric answer rather than the 500-question sample.

## Key findings (500 HotpotQA questions per model; 95% Wilson intervals in `results/comparison.md`)

1. **Retrieval is the bottleneck.** With the gold passages given, the models answer 70–78% correctly. With BM25
   top-5 retrieval, accuracy falls to 44–53% (exact McNemar p < 1e-18 for every model), and both gold passages are
   retrieved for only 53% of questions.
2. **Accuracy hides instability.** Reversing the order of the same passages leaves accuracy unchanged (Llama:
   61.4% vs 62.2%, McNemar p = 0.72), yet it changes 11–23% of individual answers.
3. **Distractors cost 7–13 points.** Adding 8 distractor passages lowers accuracy for all three models
   (p ≤ 2e-4).
4. **Models answer without evidence.** With only distractors shown, the models still answer instead of
   abstaining for 26–40% of questions, and 17–28% of all such questions get a wrong answer.
5. **Under conflict, models mostly follow the passage.** When a numeric fact in the gold passage is changed, the
   models follow the passage in 65–74% of cases and fall back on memorised knowledge in only 4–9%.
6. **Citations are precise but incomplete.** Cited passages are relevant (precision 0.91–0.93), but the models
   cite only 64–70% of the passages they need (recall).

| Model | Gold passages | + distractors | BM25 retrieved | Order changes answer | Answers with evidence missing | Conflict: follows passage |
|---|---|---|---|---|---|---|
| Gemma-2-9B | 78.2% | 70.8% | 53.2% | 10.8% | 40.0% | 74.3% |
| Llama-3.1-8B | 74.6% | 61.4% | 44.4% | 23.0% | 26.2% | 67.1% |
| Mistral-Nemo-12B | 69.8% | 62.4% | 48.2% | 17.6% | 39.6% | 64.7% |

## Measures

Exact match and F1 (SQuAD normalisation); an answer counts as correct on exact match, whole-word containment
of the gold answer, or token F1 >= 0.6 with no number absent from the gold answer. Also: abstention, answering
without evidence (and how often such answers are wrong), context-vs-memory following under conflict, answer
changes caused by passage order alone, citation precision/recall of the passages the model says it used, and
retrieval recall for the BM25 pipeline. Rates are reported with 95% Wilson intervals; paired differences use an
exact McNemar test.

## Data and models

- HotpotQA, distractor setting, validation split (CC BY-SA 4.0).
- Open-weight instruction-tuned models run locally with greedy decoding: Llama-3.1-8B-Instruct,
  Gemma-2-9B-it, Mistral-Nemo-Instruct-2407.

## Layout

```
rag_audit/   data.py (conditions) · retrieve.py (BM25) · prompting.py · generate.py · scoring.py · stats.py
             report.py · cli.py · rescore.py (re-score saved replies) · compare.py (cross-model table) · selftest.py
slurm/       batch scripts
```

Each run writes `items.jsonl` (every reply with its scores), `summary.json`, `summary.txt` and `run_info.json`.

## How to run

```bash
pip install -r requirements.txt
pytest -q                                   # CPU checks, no model needed

# Save the HotpotQA distractor validation split as data/hotpot_distractor_validation.parquet, then:
python -m rag_audit.cli --data data/hotpot_distractor_validation.parquet --n 50 --seed 13 \
    --model-path <local model folder> --model-name llama-3.1-8b --out-dir results/demo

python -m rag_audit.cli ... --dry-run       # build items and prompts only, no model
python -m rag_audit.compare results         # cross-model table -> results/comparison.md
```

`slurm/` holds the batch scripts used for the full runs (one GPU per model).
