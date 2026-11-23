import json
import sys

from schema_generator.value_generator import INSTRUCTIONS, decode_value, value_output_schema


def load_backend(model, device):
    try:
        import openvino as ov
        import openvino_genai as genai
    except ImportError as error:
        raise RuntimeError("Install openvino-genai in the inference Python environment.") from error
    available = {name.split(".")[0] for name in ov.Core().available_devices}
    candidates = ("NPU", "GPU", "CPU") if device == "auto" else (device.upper(),)
    failures = []
    for candidate in candidates:
        if candidate not in available:
            failures.append(f"{candidate}: unavailable")
            continue
        try:
            pipeline = genai.LLMPipeline(str(model), candidate)
        except RuntimeError as error:
            failures.append(f"{candidate}: {error}")
            continue
        print(f"OpenVINO generation device: {candidate}", file=sys.stderr)
        break
    else:
        raise RuntimeError("OpenVINO initialization failed. " + "\n".join(failures))

    def generate(request, seed):
        config = genai.GenerationConfig()
        config.max_new_tokens = 512
        config.do_sample = True
        config.temperature = 0.7
        config.rng_seed = seed
        grammar = genai.StructuredOutputConfig()
        grammar.json_schema = json.dumps(value_output_schema(request))
        config.structured_output_config = grammar
        history = [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False) + "\n/no_think"},
        ]
        prompt = pipeline.get_tokenizer().apply_chat_template(history, add_generation_prompt=True)
        config.apply_chat_template = False
        result = pipeline.generate(prompt, config)
        return decode_value(result if isinstance(result, str) else result.texts[0])

    return generate
