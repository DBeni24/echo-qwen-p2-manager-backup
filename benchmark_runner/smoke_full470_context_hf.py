from pathlib import Path
import json
import sys
import time

import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/workspace/echo/models/Qwen3.5-9B")
ADAPTER = Path("/workspace/echo/output/p0_false_state_lora_v01/adapter_final")

sys.path.insert(0, str(ROOT))

from scorers.scoring import score_case
from runner.run_full470 import rag_text, make_long_context


def loadjl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


cases = {
    c["case_id"]: c
    for c in loadjl(ROOT / "cases/full470.jsonl")
}

packs = {
    p["pack_id"]: p
    for p in loadjl(ROOT / "rag_packs/all_rag_packs.jsonl")
}

wanted = [
    "RAG-001",
    "CTX-001",
    "CTX-011",
    "CTX-021",
    "CTX-031",
]

print("GPU:", torch.cuda.get_device_name(0))
print("BF16:", torch.cuda.is_bf16_supported())

tok = AutoTokenizer.from_pretrained(
    BASE,
    local_files_only=True,
)

base = Qwen3_5ForConditionalGeneration.from_pretrained(
    BASE,
    dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True,
)

model = PeftModel.from_pretrained(
    base,
    ADAPTER,
    is_trainable=False,
)

model.eval()

print(
    "MODEL_LOAD_PASS",
    "VRAM_GB:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
)

for idx, case_id in enumerate(wanted, 1):
    c = cases[case_id]

    parts = []

    if c.get("rag_pack_id"):
        parts.append(
            "=== RAG/KONTEXTUS ===\n"
            + rag_text(packs[c["rag_pack_id"]])
            + "\n=== RAG/KONTEXTUS VÉGE ==="
        )

    if c.get("context_spec"):
        parts.append(
            "=== HOSSZÚ KONTEXTUS ===\n"
            + make_long_context(c)
            + "\n=== HOSSZÚ KONTEXTUS VÉGE ==="
        )

    parts.append(
        "=== FELADAT ===\n" + c["prompt"]
    )

    messages = [
        {
            "role": "system",
            "content": c["system_prompt"],
        },
        {
            "role": "user",
            "content": "\n\n".join(parts),
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

    input_tokens = inputs["input_ids"].shape[1]

    print("=" * 100)
    print(
        f"[{idx}/{len(wanted)}]",
        case_id,
        "CTX_LIMIT:",
        c["context_length"],
        "INPUT_TOKENS:",
        input_tokens,
    )

    if input_tokens + c["max_output_tokens"] > c["context_length"]:
        print("RESOURCE_ERROR: CONTEXT_BUDGET_EXCEEDED")
        continue

    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=c["max_output_tokens"],
            do_sample=False,
            use_cache=True,
        )

    elapsed = time.time() - t0

    generated = out[
        0,
        inputs["input_ids"].shape[1]:
    ]

    answer = tok.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    score = score_case(c, answer)

    output_tokens = generated.shape[0]

    print("STATUS:", score["status"])
    print("OUTPUT_TOKENS:", output_tokens)
    print("SECONDS:", round(elapsed, 3))
    print(
        "TOK/S:",
        round(output_tokens / elapsed, 2)
        if elapsed else None,
    )
    print(
        "PEAK_VRAM_GB:",
        round(
            torch.cuda.max_memory_allocated() / 1024**3,
            2,
        ),
    )
    print("ANSWER:", answer)
    print(
        "SCORE:",
        json.dumps(
            score,
            ensure_ascii=False,
        ),
    )

    del inputs, out, generated
    torch.cuda.empty_cache()

print("=" * 100)
print("RAG_AND_CONTEXT_SMOKE_COMPLETE")
