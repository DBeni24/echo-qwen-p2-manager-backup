from pathlib import Path
import json
import time
import torch

from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/workspace/echo/models/Qwen3.5-9B")
ADAPTER = Path("/workspace/echo/output/p0_false_state_lora_v01/adapter_final")

import sys
sys.path.insert(0, str(ROOT))
from scorers.scoring import score_case


def loadjl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("BF16:", torch.cuda.is_bf16_supported())

tok = AutoTokenizer.from_pretrained(
    BASE,
    local_files_only=True,
)

print("TOKENIZER_LOAD_PASS")

base = Qwen3_5ForConditionalGeneration.from_pretrained(
    BASE,
    dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True,
)

print(
    "BASE_LOAD_PASS",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GiB"
)

model = PeftModel.from_pretrained(
    base,
    ADAPTER,
    is_trainable=False,
)

model.eval()

print(
    "LORA_LOAD_PASS",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GiB"
)

cases = loadjl(ROOT / "cases/full470.jsonl")[:5]

for i, case in enumerate(cases, 1):
    messages = [
        {
            "role": "system",
            "content": case["system_prompt"],
        },
        {
            "role": "user",
            "content": "=== FELADAT ===\n" + case["prompt"],
        },
    ]

    rendered = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tok(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    t0 = time.time()

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=case["max_output_tokens"],
            do_sample=False,
            use_cache=True,
        )

    elapsed = time.time() - t0

    generated = out[0, inputs["input_ids"].shape[1]:]
    answer = tok.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    score = score_case(case, answer)

    print("=" * 90)
    print(f"[{i}/5] {case['case_id']}")
    print("STATUS:", score["status"])
    print("INPUT_TOKENS:", inputs["input_ids"].shape[1])
    print("OUTPUT_TOKENS:", generated.shape[0])
    print("SECONDS:", round(elapsed, 3))
    print("TOK/S:", round(generated.shape[0] / elapsed, 2) if elapsed else None)
    print("ANSWER:", answer)
    print("SCORE:", json.dumps(score, ensure_ascii=False))

print("=" * 90)
print("HF_LORA_FULL470_SMOKE_COMPLETE")
