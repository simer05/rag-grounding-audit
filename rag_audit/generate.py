"""Greedy batched generation with Hugging Face transformers (local weights only)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path):
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"local_files_only": True, "torch_dtype": torch.bfloat16, "device_map": "cuda"}
    if "gemma" in model_path.lower():
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return tok, model


@torch.inference_mode()
def generate(tok, model, prompts, batch_size=8, max_new_tokens=64):
    """Return one reply per prompt, in input order. Longest prompts run first."""
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    replies = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        enc = tok([texts[i] for i in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.pad_token_id,
        )
        decoded = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, text in zip(batch, decoded):
            replies[i] = text.strip()
    return replies
