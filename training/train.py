import argparse
import json
import random
import sys
from pathlib import Path

import torch
from jsonschema import Draft202012Validator
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from value_generator import INSTRUCTIONS, parse_value


def load_examples(path):
    examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not examples:
        raise ValueError(f"No examples in {path}")
    for example in examples:
        request = example["request"]
        schema = request["field_schema"]
        Draft202012Validator.check_schema(schema)
        value = parse_value(example["target"], request["target_type"])
        Draft202012Validator(schema).validate(value)
    return examples


def encode(tokenizer, example):
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": json.dumps(example["request"], ensure_ascii=False) + "\n/no_think"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answer = json.dumps(example["target"], ensure_ascii=False)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    return prefix, target


def sequence_loss(model, prefix, target, device):
    tokens = torch.tensor([prefix + target], device=device)
    labels = torch.tensor([[-100] * len(prefix) + target], device=device)
    if device != "xpu":
        return model(input_ids=tokens, labels=labels).loss

    # Build the causal mask without the advanced indexing used by Transformers.
    positions = torch.arange(tokens.shape[-1])
    mask = torch.zeros((tokens.shape[-1], tokens.shape[-1]), dtype=torch.float32)
    mask.masked_fill_(positions[None, :] > positions[:, None], torch.finfo(mask.dtype).min)
    return model(input_ids=tokens, labels=labels, attention_mask=mask[None, None].to(device)).loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=PROJECT_DIR / ".training/base")
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "training/train/values.jsonl")
    parser.add_argument("--test-data", type=Path, default=PROJECT_DIR / "training/test/values.jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT_DIR / ".training/value-adapter")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error("--steps and --threads must be positive")
    examples = load_examples(args.data)
    test_examples = load_examples(args.test_data)
    if {row["split_group"] for row in examples} & {row["split_group"] for row in test_examples}:
        parser.error("Train and test split groups overlap.")

    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "xpu": torch.xpu.is_available()}
    device = args.device
    if device == "auto":
        device = next(name for name in ("cuda", "xpu", "cpu") if available[name])
    if not available[device]:
        parser.error(f"{device} is unavailable in this PyTorch installation.")
    args.out.mkdir(parents=True, exist_ok=False)
    print(f"Loading base model on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    sequences = [encode(tokenizer, example) for example in examples]
    test_sequences = [encode(tokenizer, example) for example in test_examples]
    model = AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=True, dtype=torch.float32
    ).to(device)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))
    model.train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
    randomizer = random.Random(42)
    for step in range(args.steps):
        if step % len(sequences) == 0:
            randomizer.shuffle(sequences)
        prefix, target = sequences[step % len(sequences)]
        optimizer.zero_grad(set_to_none=True)
        loss = sequence_loss(model, prefix, target, device)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 8 == 0 or step + 1 == args.steps:
            print(f"Step {step + 1}/{args.steps} loss {loss.item():.4f}", flush=True)
    model.save_pretrained(args.out)
    model.eval()
    weighted_loss = 0.0
    target_tokens = 0
    with torch.no_grad():
        for prefix, target in test_sequences:
            weighted_loss += sequence_loss(model, prefix, target, device).item() * len(target)
            target_tokens += len(target)
    metrics = {
        "steps": args.steps, "train_examples": len(examples), "test_examples": len(test_examples),
        "test_token_loss": weighted_loss / target_tokens,
        "train_data": str(args.data.resolve()), "test_data": str(args.test_data.resolve()),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics), flush=True)
    print(f"Saved LoRA adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
