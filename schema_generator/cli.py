import argparse
import json
import os
import random
import subprocess
import sys
from contextlib import ExitStack
from datetime import date
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry
from referencing.exceptions import Unresolvable

from schema_generator.value_generator import GenerationValidator


APP_DIR = Path(__file__).resolve().parents[1]
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


def generate_records(validator, count, language, runner, model, adapter=None, seed=None, mode="auto",
                     scenario=None):
    if mode in ("auto", "field"):
        from schema_generator.value_generator import generate_values

        return generate_values(validator, count, language, runner, model, adapter, seed, scenario=scenario)
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
    parser.add_argument("--seed", type=int, default=42, help="Seed for Python and local model sampling (default: 42)")
    parser.add_argument("--mode", choices=("auto", "record", "field"), default="auto",
                        help="Python builds structures and parses model values; record is a legacy comparison mode")
    parser.add_argument("--language", choices=("ko", "en"), required=True)
    parser.add_argument("--scenario", help="Shared natural-language situation for value generation")
    parser.add_argument("--reference-date", type=date.fromisoformat, help="Date sampling reference (YYYY-MM-DD; default: today)")
    parser.add_argument("--model", type=Path, default=APP_DIR / "models/model.gguf", help="Local GGUF file")
    parser.add_argument("--adapter", type=Path, help="Optional local LoRA adapter in GGUF format")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "npu"), default="auto")
    parser.add_argument("--ov-model", type=Path, help="OpenVINO GenAI model directory for NPU/GPU generation")
    parser.add_argument("--server-runner", type=Path, help="Local llama-server executable to reuse GGUF between fields")
    runner_name = "llama-completion.exe" if os.name == "nt" else "llama-completion"
    parser.add_argument("--runner", type=Path, default=APP_DIR / "runtime" / "llama.cpp" / runner_name,
                        help="llama.cpp executable")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be greater than zero")
    if args.mode == "record" and (args.scenario is not None or args.reference_date is not None):
        parser.error("--scenario and --reference-date require value generation mode.")

    try:
        schema = json.loads(args.schema.read_text(encoding="utf-8-sig"))
        draft = Draft202012Validator.META_SCHEMA["$id"]
        if isinstance(schema, dict) and "$schema" in schema and schema["$schema"] not in (draft, draft + "#"):
            raise ValueError("Only JSON Schema Draft 2020-12 is supported.")
        Draft202012Validator.check_schema(schema)
        model = args.model.resolve()
        runner = args.runner.resolve()
        adapter = args.adapter.resolve() if args.adapter is not None else None
        if args.server_runner is not None:
            if args.ov_model is not None or args.mode == "record" or args.device == "npu":
                raise ValueError("--server-runner requires GGUF value generation on CPU or GPU.")
        if args.ov_model is not None:
            if adapter is not None or args.mode == "record":
                raise ValueError("OpenVINO requires value mode and a model directory with any adapter already merged.")
        elif args.server_runner is None and args.device in ("gpu", "npu"):
            raise ValueError(f"--device {args.device} requires --ov-model; GPU also supports --server-runner.")
        def check_gguf_files():
            if adapter is not None and not adapter.is_file():
                raise RuntimeError(f"Local LoRA adapter is missing: {adapter}")
            if not model.is_file():
                raise RuntimeError(f"Local GGUF model is missing: {model}. "
                                   "Place model.gguf in models/ or pass --model PATH.")
            executable = args.server_runner or runner
            if not executable.is_file():
                raise RuntimeError(f"Local model runner is missing: {executable}")

        validator = GenerationValidator(
            schema,
            format_checker=Draft202012Validator.FORMAT_CHECKER,
            registry=Registry(),
        )
        if args.mode == "record":
            check_gguf_files()
            records = generate_records(
                validator, args.count, args.language, runner, model, adapter, args.seed, args.mode, args.scenario
            )
        else:
            from schema_generator.value_generator import generate_values, request_value

            with ExitStack() as stack:
                backend = None

                def generate_field(request, seed):
                    nonlocal backend
                    if backend is None:
                        if args.ov_model is not None:
                            from schema_generator.openvino_backend import load_backend

                            if not (args.ov_model / "openvino_model.xml").is_file():
                                raise RuntimeError(f"OpenVINO model files are missing: {args.ov_model}")
                            backend = load_backend(args.ov_model.resolve(), args.device)
                        else:
                            check_gguf_files()
                            if args.server_runner is not None:
                                from schema_generator.llama_server_backend import load_server_backend

                                backend = stack.enter_context(
                                    load_server_backend(args.server_runner.resolve(), model, adapter, args.device)
                                )
                            else:
                                def backend(request, seed):
                                    return request_value(runner, model, adapter, seed, request)
                    return backend(request, seed)

                records = generate_values(
                    validator, args.count, args.language, runner, model,
                    seed=args.seed, backend=generate_field, scenario=args.scenario,
                    reference_date=args.reference_date,
                )
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
