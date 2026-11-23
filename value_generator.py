import json
import math
import random
import subprocess
from fractions import Fraction

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


INSTRUCTIONS = """Generate one fictional field value as a JSON string.
Read the field name, description, type, constraints and related values together.
Descriptions are optional. Infer familiar fields such as name and age from names.
Use the requested language for natural language values.
For integer/number return numeric text, for boolean return "true" or "false",
and for null return "null", all encoded as a JSON string.
Return only the JSON string containing the actual value.
"""
TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
UNSUPPORTED = {
    "$ref", "$dynamicRef", "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
    "dependentSchemas", "dependentRequired", "patternProperties", "contains",
    "unevaluatedProperties", "unevaluatedItems",
}


def check_supported(schema, path="$"):
    if schema is False:
        raise ValueError(f"{path}: a false schema cannot generate a value.")
    if schema is True:
        return
    unsupported = UNSUPPORTED.intersection(schema)
    if unsupported:
        raise ValueError(f"{path}: generation does not yet support {', '.join(sorted(unsupported))}.")
    kinds = schema.get("type", [])
    if isinstance(kinds, str):
        kinds = [kinds]
    if set(kinds) - TYPES:
        raise ValueError(f"{path}: unsupported type.")
    for name, child in schema.get("properties", {}).items():
        check_supported(child, f"{path}.{name}")
    for index, child in enumerate(schema.get("prefixItems", [])):
        check_supported(child, f"{path}[{index}]")
    if "items" in schema and schema["items"] is not False:
        check_supported(schema["items"], f"{path}[]")
    if isinstance(schema.get("additionalProperties"), dict):
        check_supported(schema["additionalProperties"], path + ".*")


def parse_value(text, kind):
    if kind == "string":
        return text
    value = json.loads(text.strip())
    json.dumps(value, allow_nan=False)
    if kind == "integer":
        valid = type(value) is int
    elif kind == "number":
        valid = type(value) in (int, float)
    elif kind == "boolean":
        valid = type(value) is bool
    else:
        valid = value is None
    if not valid:
        raise ValueError(f"Expected text representing {kind}.")
    return value


def sample_bounded_number(validator, kind, sampler):
    schema = validator.schema
    lower = schema.get("minimum")
    upper = schema.get("maximum")
    exclusive_lower = schema.get("exclusiveMinimum")
    exclusive_upper = schema.get("exclusiveMaximum")
    if exclusive_lower is not None and (lower is None or exclusive_lower >= lower):
        lower = exclusive_lower
    else:
        exclusive_lower = None
    if exclusive_upper is not None and (upper is None or exclusive_upper <= upper):
        upper = exclusive_upper
    else:
        exclusive_upper = None
    if lower is None and upper is None:
        return None
    multiple = schema.get("multipleOf")
    step = Fraction(str(multiple)) if multiple is not None else Fraction(1)
    if kind == "integer":
        step = Fraction(step.numerator)
    if kind == "integer" or multiple is not None:
        first = None if lower is None else math.ceil(Fraction(str(lower)) / step)
        last = None if upper is None else math.floor(Fraction(str(upper)) / step)
        if first is not None and exclusive_lower is not None and first * step == Fraction(str(lower)):
            first += 1
        if last is not None and exclusive_upper is not None and last * step == Fraction(str(upper)):
            last -= 1
        first = last - 100 if first is None else first
        last = first + 100 if last is None else last
        if first <= last:
            for _ in range(16):
                number = sampler.randint(first, last) * step
                value = int(number) if number.denominator == 1 else float(number)
                if validator.is_valid(value):
                    return value
    else:
        first = math.nextafter(lower, math.inf) if exclusive_lower is not None else lower
        last = math.nextafter(upper, -math.inf) if exclusive_upper is not None else upper
        first = last - 100 if first is None else first
        last = first + 100 if last is None else last
        if first <= last:
            ratio = sampler.random()
            for value in (first * (1 - ratio) + last * ratio, first, last):
                if math.isfinite(value) and validator.is_valid(value):
                    return value
    return None


def value_output_schema(request):
    output_schema = {"type": "string"}
    patterns = {
        "integer": "^-?(0|[1-9][0-9]*)$",
        "number": "^-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$",
        "boolean": "^(true|false)$",
        "null": "^null$",
    }
    if request["target_type"] in patterns:
        output_schema["pattern"] = patterns[request["target_type"]]
    field_schema = request["field_schema"]
    if request["target_type"] == "string" and "pattern" not in field_schema:
        for keyword in ("minLength", "maxLength"):
            if keyword in field_schema:
                output_schema[keyword] = field_schema[keyword]
    if request["target_type"] == "integer" and "minimum" in field_schema and "maximum" in field_schema:
        lower, upper = math.ceil(field_schema["minimum"]), math.floor(field_schema["maximum"])
        if upper - lower <= 1000:
            validator = Draft202012Validator(field_schema)
            allowed = [str(value) for value in range(lower, upper + 1) if validator.is_valid(value)]
            if not allowed:
                raise ValueError("No integer satisfies the field constraints.")
            output_schema = {"type": "string", "enum": allowed}
    return output_schema


def request_value(runner, model, adapter, seed, request):
    output_schema = value_output_schema(request)
    command = [
        str(runner), "-m", str(model), "--conversation", "--single-turn",
        "--no-display-prompt", "--color", "off", "--no-warmup", "--no-escape",
        "--ctx-size", "4096", "--predict", "512",
        "--system-prompt", INSTRUCTIONS,
        "--prompt", json.dumps(request, ensure_ascii=False) + "\n/no_think",
        "--json-schema", json.dumps(output_schema),
        "--seed", str(seed),
    ]
    if adapter is not None:
        command.extend(["--lora", str(adapter)])
    result = subprocess.run(
        command, stdin=subprocess.DEVNULL, capture_output=True,
        encoding="utf-8", errors="replace", cwd=runner.parent,
    )
    if result.returncode:
        raise RuntimeError(f"llama-completion exited with code {result.returncode}: {result.stderr.strip()}")
    return decode_value(result.stdout)


def decode_value(output):
    value, _ = json.JSONDecoder().raw_decode(output.lstrip())
    if not isinstance(value, str):
        raise ValueError("The model must return a string containing the field value.")
    return value


def generate_values(validator, count, language, runner, model, adapter=None, seed=None, backend=None):
    check_supported(validator.schema)
    sampler = random.Random(seed)

    def build(current, name, path, context, depth=0, parent_description=None):
        if depth > 32:
            raise ValueError(f"{path}: maximum generation depth exceeded.")
        schema = current.schema
        if schema is True:
            schema = {}
        if "const" in schema:
            current.validate(schema["const"])
            return schema["const"]
        if "enum" in schema:
            choices = [value for value in schema["enum"] if current.is_valid(value)]
            if not choices:
                raise ValueError(f"{path}: no enum value satisfies the schema.")
            return sampler.choice(choices)
        kind = schema.get("type")
        if kind is None:
            kind = "object" if "properties" in schema else "array" if "items" in schema or "prefixItems" in schema else "string"
        if isinstance(kind, list):
            kind = sampler.choice(kind)
        if kind == "object":
            result = {}
            properties = schema.get("properties", {})
            extra_names = set(schema.get("required", [])) - properties.keys()
            for child_name in [*properties, *sorted(extra_names)]:
                child_schema = properties.get(child_name, schema.get("additionalProperties", {}))
                child = current.evolve(schema=child_schema)
                result[child_name] = build(
                    child, child_name, f"{path}.{child_name}", context | result,
                    depth + 1, schema.get("description"),
                )
            current.validate(result)
            return result
        if kind == "array":
            minimum = schema.get("minItems", 0)
            prefix = schema.get("prefixItems", [])
            default_maximum = max(minimum, len(prefix), 3)
            maximum = schema.get("maxItems", default_maximum)
            if schema.get("items") is False:
                maximum = min(maximum, len(prefix))
            if maximum < minimum:
                raise ValueError(f"{path}: array bounds cannot be satisfied.")
            length = sampler.randint(minimum, min(maximum, max(minimum, len(prefix), 3)))
            result = []
            for index in range(length):
                child_schema = prefix[index] if index < len(prefix) else schema.get("items", {})
                child = current.evolve(schema=child_schema)
                for attempt in range(3):
                    value = build(
                        child, name, f"{path}[{index}]", context, depth + 1, schema.get("description")
                    )
                    if not schema.get("uniqueItems") or all(
                        not current.evolve(schema={"const": item}).is_valid(value) for item in result
                    ):
                        break
                else:
                    raise ValueError(f"{path}: could not generate unique array items.")
                result.append(value)
            current.validate(result)
            return result

        request = {
            "field_name": name, "path": path, "field_schema": schema,
            "target_type": kind, "context": context,
            "output_language": {"ko": "Korean", "en": "English"}[language],
        }
        if parent_description:
            request["parent_description"] = parent_description
        for attempt in range(3):
            try:
                attempt_seed = sampler.randrange(2**32 - 1)
                text = (
                    request_value(runner, model, adapter, attempt_seed, request)
                    if backend is None else backend(request, attempt_seed)
                )
                value = parse_value(text, kind)
                current.validate(value)
                return value
            except (ValueError, ValidationError) as error:
                failure = error.message if isinstance(error, ValidationError) else str(error)
                request["previous_validation_error"] = failure
        if kind in ("integer", "number"):
            value = sample_bounded_number(current, kind, sampler)
            if value is not None:
                return value
        raise RuntimeError(f"{path}: value generation failed after 3 attempts: {failure}")

    records = []
    for _ in range(count):
        record = build(validator, "value", "$", {})
        validator.validate(record)
        records.append(record)
    return records
