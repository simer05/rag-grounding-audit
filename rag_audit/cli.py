"""Run one audit: one model, a fixed sample of questions, the selected conditions.

    python -m rag_audit.cli --data data/hotpot_distractor_validation.parquet --n 50 \
        --model-path <local snapshot dir> --model-name llama-3.1-8b --out-dir results/<run>
"""
import argparse
import json
import platform
import time
from pathlib import Path

from .data import CONDITIONS, build_items, load_hotpot
from .retrieve import build_retrieved_items
from .prompting import build_prompt
from .report import summarise, to_text
from .scoring import score_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--model-path")
    ap.add_argument("--model-name")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=5, help="passages retrieved per question (retrieved condition)")
    ap.add_argument("--dry-run", action="store_true", help="build items and prompts only, no model")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conditions = tuple(c for c in args.conditions.split(",") if c)

    records, pool = load_hotpot(args.data, args.n, args.seed)
    items = build_items(records, pool, args.seed, conditions)
    if "retrieved" in conditions:
        items += build_retrieved_items(records, args.top_k)
    prompts = [build_prompt(it) for it in items]
    counts = {c: sum(1 for it in items if it["condition"] == c) for c in conditions}
    print(f"questions={len(records)} items={len(items)} per_condition={counts}", flush=True)
    print(f"longest prompt chars={max(len(p) for p in prompts)}", flush=True)

    info = {"args": vars(args), "questions": len(records), "items_per_condition": counts,
            "python": platform.python_version()}
    if args.dry_run:
        (out / "prompt_example.txt").write_text(prompts[0], encoding="utf-8")
        (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return

    import torch
    import transformers
    from .generate import generate, load_model

    info.update(torch=torch.__version__, transformers=transformers.__version__,
                gpu=torch.cuda.get_device_name(0))
    print(f"torch={info['torch']} transformers={info['transformers']} gpu={info['gpu']}", flush=True)

    t0 = time.time()
    tok, model = load_model(args.model_path)
    info["load_seconds"] = round(time.time() - t0, 1)
    t0 = time.time()
    replies = generate(tok, model, prompts, args.batch_size, args.max_new_tokens)
    info["generate_seconds"] = round(time.time() - t0, 1)
    info["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(f"generated {len(replies)} replies in {info['generate_seconds']}s, "
          f"peak GPU {info['peak_gpu_gb']} GB", flush=True)

    rows = []
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for it, reply in zip(items, replies):
            row = {k: it[k] for k in ("qid", "condition", "question", "answer", "target", "type", "level")}
            row["gold_ids"] = [i for i, p in enumerate(it["passages"], 1) if p["gold"]]
            row["n_passages"] = len(it["passages"])
            if "retrieval_recall" in it:
                row["retrieval_recall"] = it["retrieval_recall"]
            row["reply"] = reply
            row.update(score_item(it, reply))
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarise(rows)
    summary["model"] = args.model_name
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = to_text(summary, header=f"model={args.model_name} n_questions={len(records)} seed={args.seed}")
    (out / "summary.txt").write_text(text, encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
