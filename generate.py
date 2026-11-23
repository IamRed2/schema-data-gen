import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry
from referencing.exceptions import Unresolvable


APP_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 4
INSTRUCTIONS = """Generate the requested number of varied fictional records.
Each record must satisfy the supplied JSON Schema.
Use the whole schema to keep related fields consistent. Read each field name
and its description together, along with its type and related field values.
Descriptions refine the field's meaning, role, topic or requested form.
Descriptions are optional: infer familiar fields such as name and age from
their names and types. A name should be a plausible fictional person's name;
an age should be a plausible numeric age within the schema constraints.
Field names and descriptions may be Korean or English.
Write generated natural-language values in the requested output language.
Preserve property names and literal values required by enum or const.
Return a JSON object with one key, "records", containing the array of records.
"""


def request_json(runner, model, generation_input, adapter=None, seed=None):
    count = generation_input["count"]
    response_schema = {
        "type": "object",
        "properties": {
            "records": {"type": "array", "items": {}, "minItems": count, "maxItems": count}
        },
        "required": ["records"],
        "additionalProperties": False,
    }
    prompt = json.dumps(generation_input, ensure_ascii=False) + "\n/no_think"
    command = [
        str(runner), "-m", str(model), "--conversation", "--single-turn",
        "--no-display-prompt", "--color", "off", "--no-warmup", "--no-escape",
        "--ctx-size", "4096", "--predict", "2048",
        "--system-prompt", INSTRUCTIONS, "--prompt", prompt,
        "--json-schema", json.dumps(response_schema),
    ]
    if adapter is not None:
        command.extend(["--lora", str(adapter)])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=runner.parent,
    )
    if result.returncode:
        detail = result.stderr.strip()
        raise RuntimeError(f"llama-completion exited with code {result.returncode}. {detail}")
    return result.stdout


def generate_records(validator, count, language, runner, model, adapter=None, seed=None, mode="auto"):
    if mode in ("auto", "field"):
        from value_generator import generate_values

        return generate_values(validator, count, language, runner, model, adapter, seed)
    generation_input = {
        "schema": validator.schema,
        "output_language": {"ko": "Korean", "en": "English"}[language],
    }
    records = []
    randomizer = random.Random(seed) if seed is not None else None
    while len(records) < count:
        batch_count = min(BATCH_SIZE, count - len(records))
        attempt_input = dict(generation_input, count=batch_count)
        for _ in range(3):
            attempt_seed = randomizer.randrange(2**32 - 1) if randomizer is not None else None
            response = request_json(runner, model, attempt_input, adapter, attempt_seed)
            try:
                envelope, _ = json.JSONDecoder().raw_decode(response.lstrip())
                batch = envelope.get("records") if isinstance(envelope, dict) else None
                if not isinstance(batch, list) or len(batch) != batch_count:
                    raise ValueError(f'Expected a "records" array with {batch_count} entries.')
                # Reject non-JSON numbers accepted by Python's JSON decoder.
                json.dumps(batch, allow_nan=False)
                for record in batch:
                    validator.validate(record)
            except (ValueError, ValidationError) as error:
                if isinstance(error, ValidationError):
                    failure = f"{error.json_path}: {error.message}"
                else:
                    failure = str(error)
                attempt_input["previous_validation_error"] = failure
                continue
            records.extend(batch)
            break
        else:
            raise RuntimeError(f"Batch starting at record {len(records) + 1} failed after 3 attempts: {failure}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate fictional data from a JSON Schema.")
    parser.add_argument("schema", type=Path)
    parser.add_argument("--out", type=Path, help="Write the JSON array to this file (default: stdout)")
    parser.add_argument("--count", type=int, default=5, help="Number of records (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for local model sampling (default: 42)")
    parser.add_argument("--mode", choices=("auto", "record", "field"), default="auto",
                        help="Python builds structures and parses model values; record is a legacy comparison mode")
    parser.add_argument("--language", choices=("ko", "en"), required=True)
    parser.add_argument("--model", type=Path, default=APP_DIR / "model.gguf", help="Local GGUF file")
    parser.add_argument("--adapter", type=Path, help="Optional local LoRA adapter in GGUF format")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "npu"), default="auto")
    parser.add_argument("--ov-model", type=Path, help="OpenVINO GenAI model directory for NPU/GPU generation")
    runner_name = "llama-completion.exe" if os.name == "nt" else "llama-completion"
    parser.add_argument("--runner", type=Path, default=APP_DIR / "runtime" / "llama.cpp" / runner_name,
                        help="llama.cpp executable")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be greater than zero")

    try:
        schema = json.loads(args.schema.read_text(encoding="utf-8-sig"))
        draft = Draft202012Validator.META_SCHEMA["$id"]
        if isinstance(schema, dict) and "$schema" in schema and schema["$schema"] not in (draft, draft + "#"):
            raise ValueError("Only JSON Schema Draft 2020-12 is supported.")
        Draft202012Validator.check_schema(schema)
        model = args.model.resolve()
        runner = args.runner.resolve()
        adapter = args.adapter.resolve() if args.adapter is not None else None
        if args.ov_model is not None:
            if adapter is not None or args.mode == "record":
                raise ValueError("OpenVINO requires value mode and a model directory with any adapter already merged.")
            if not (args.ov_model / "openvino_model.xml").is_file():
                raise ValueError(f"OpenVINO model files are missing: {args.ov_model}")
        elif args.device in ("gpu", "npu"):
            raise ValueError(f"--device {args.device} requires --ov-model PATH.")
        if adapter is not None and not adapter.is_file():
            raise ValueError(f"Local LoRA adapter is missing: {adapter}")
        if args.ov_model is None and not model.is_file():
            raise ValueError(
                f"Local GGUF model is missing: {model}. "
                "Place model.gguf next to generate.py or pass --model PATH."
            )
        if args.ov_model is None and not runner.is_file():
            raise ValueError(
                f"llama-completion is missing: {runner}. "
                f"Place {runner_name} and its runtime libraries in runtime/llama.cpp or pass --runner PATH."
            )
        validator = Draft202012Validator(
            schema,
            format_checker=Draft202012Validator.FORMAT_CHECKER,
            registry=Registry(),
        )
        if args.ov_model is None:
            records = generate_records(validator, args.count, args.language, runner, model, adapter, args.seed, args.mode)
        else:
            from openvino_backend import load_backend
            from value_generator import generate_values

            backend = load_backend(args.ov_model.resolve(), args.device)
            records = generate_values(validator, args.count, args.language, runner, model, seed=args.seed, backend=backend)
        output = (json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if args.out is None:
            sys.stdout.buffer.write(output)
        else:
            args.out.write_bytes(output)
    except (OSError, ValueError, SchemaError, ValidationError, RuntimeError, Unresolvable) as error:
        detail = error.message if isinstance(error, (SchemaError, ValidationError)) else str(error)
        parser.exit(1, f"Error: {detail}\n")


if __name__ == "__main__":
    main()
