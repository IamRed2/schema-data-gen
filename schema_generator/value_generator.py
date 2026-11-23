import json
import math
import random
import re
import subprocess
import unicodedata
from datetime import date, datetime, time, timedelta, timezone
from fractions import Fraction
from uuid import UUID

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from jsonschema.validators import extend


INSTRUCTIONS = """Generate one fictional field value as a JSON string.
Read the field name, description, type, constraints and related values together.
When a scenario is provided, read it in full and use the facts relevant to the field.
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


def validate_decimal_multiple(validator, divisor, value, schema):
    if validator.is_type(value, "number"):
        ratio = Fraction(str(value)) / Fraction(str(divisor))
        if ratio.denominator != 1:
            yield ValidationError(f"{value!r} is not a multiple of {divisor}")


# JSON decimal literals can give a near-integer quotient in binary floating point.
GenerationValidator = extend(Draft202012Validator, {"multipleOf": validate_decimal_multiple})


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
    multiple = schema.get("multipleOf")
    if lower is None and upper is None and multiple is None:
        return None
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
        if first is None and last is None:
            first, last = 0, 100
        if first is None:
            first = max(0, last - 100) if last > 0 else last - 100
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
        if first is None:
            first = max(0, last - 100) if last > 0 else last - 100
        last = first + 100 if last is None else last
        if first <= last:
            ratio = sampler.random()
            sampled = first * (1 - ratio) + last * ratio
            rounded = round(sampled, 2)
            for value in (rounded, sampled, first, last):
                if math.isfinite(value) and validator.is_valid(value):
                    return value
    return None


def sample_formatted_string(validator, name, context, reference_date, sampler):
    format_name = validator.schema.get("format")
    if format_name not in {"email", "idn-email", "uuid", "date", "date-time", "time"}:
        return None

    field = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
    if format_name in {"date", "date-time"}:
        first = max(date.min.toordinal(), reference_date.toordinal() - 365)
        last = reference_date.toordinal()
        if field in {"expires_at", "expiry_date", "expiration_date", "만료일", "만료일시"}:
            first, last = last, min(date.max.toordinal(), last + 365)
        if field in {"birth_date", "birthdate", "date_of_birth", "dob", "생년월일"}:
            age = next((context[key] for key in ("age", "나이") if key in context), None)
            if type(age) is int and 0 <= age < reference_date.year:
                year = reference_date.year - age
                first = date(max(1, year - 1), 1, 1).toordinal()
                last = min(date(year, 12, 31), reference_date).toordinal()
            else:
                age = None
                first = max(date.min.toordinal(), last - 80 * 366)
                last = max(first, last - 18 * 365)
        else:
            age = None

    if format_name in {"email", "idn-email"}:
        display_name = context.get("name", context.get("full_name", ""))
        ascii_name = unicodedata.normalize("NFKD", display_name if isinstance(display_name, str) else "")
        stem = ".".join(re.findall(r"[a-z0-9]+", ascii_name.lower()))[:40].strip(".") or "user"
        suffix = str(sampler.randint(1000, 999999))
        maximum = min(64, validator.schema.get("maxLength", 76) - len("@example.com"))
        minimum = max(1, validator.schema.get("minLength", 1) - len("@example.com"))
        if maximum < minimum:
            return None
        local = (stem[:max(0, maximum - len(suffix))] + suffix)[-maximum:]
        candidate = local.ljust(minimum, "0") + "@example.com"
        return candidate if validator.is_valid(candidate) else None

    for _ in range(24):
        if format_name == "uuid":
            candidate = str(UUID(int=sampler.getrandbits(128), version=4))
        else:
            if format_name != "time":
                day = date.fromordinal(sampler.randint(first, last))
                if age is not None:
                    elapsed = reference_date.year - day.year - (
                        (reference_date.month, reference_date.day) < (day.month, day.day)
                    )
                    if elapsed != age:
                        continue
            if format_name == "date":
                candidate = day.isoformat()
            else:
                clock = time(sampler.randrange(24), sampler.randrange(60), sampler.randrange(60), tzinfo=timezone.utc)
                stamp = clock if format_name == "time" else datetime.combine(day, clock)
                candidate = stamp.isoformat(timespec="seconds").replace("+00:00", "Z")
        if validator.is_valid(candidate):
            return candidate
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


def generate_values(validator, count, language, runner, model, adapter=None, seed=None, backend=None,
                    scenario=None, reference_date=None):
    check_supported(validator.schema)
    sampler = random.Random(seed)
    reference_date = reference_date if reference_date is not None else date.today()

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

        if kind == "string":
            candidate = sample_formatted_string(current, name, context, reference_date, sampler)
            if candidate is not None:
                return candidate
        elif kind in ("integer", "number"):
            candidate = sample_bounded_number(current, kind, sampler)
            if candidate is not None:
                return candidate
        elif kind in ("boolean", "null"):
            return bool(sampler.getrandbits(1)) if kind == "boolean" else None

        request = {
            "field_name": name, "path": path, "field_schema": schema,
            "target_type": kind, "context": context,
            "output_language": {"ko": "Korean", "en": "English"}[language],
        }
        if parent_description:
            request["parent_description"] = parent_description
        if scenario:
            request["scenario"] = scenario
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
        raise RuntimeError(f"{path}: value generation failed after 3 attempts: {failure}")

    records = []
    for _ in range(count):
        record = build(validator, "value", "$", {})
        validator.validate(record)
        records.append(record)
    return records
