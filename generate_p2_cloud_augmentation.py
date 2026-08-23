from pathlib import Path
import json
import hashlib
import torch

from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import PeftModel


ROOT = Path("/workspace/echo")

BASE = ROOT / "models/Qwen3.5-9B"
ADAPTER = ROOT / "output/p2_manager_policy_lora_v01/adapter_final"

INPUT = ROOT / "dataset/p2_manager_upgrade/p2_train_final.jsonl"
OUTPUT = ROOT / "dataset/p2_manager_upgrade/generated_cloud_v01.jsonl"


TARGET_KEYS = [
    "decision",
    "risk_level",
    "executor_class",
    "approval_required",
    "validation_required",
    "execution_mode",
]


VARIATIONS_PER_CASE = 10


def load_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def main():

    print("LOADING MODEL")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE,
        local_files_only=True,
    )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(
        model,
        ADAPTER,
        is_trainable=False,
    )

    model.eval()

    cases = load_jsonl(INPUT)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    generated = 0


    with OUTPUT.open(
        "w",
        encoding="utf-8"
    ) as out:

        for idx, case in enumerate(cases,1):

            messages = case["messages"]

            expected = case["messages"][-1]["content"]

            prompt = f"""
Te egy AION Manager training dataset generátor vagy.

A következő policy esetből készíts {VARIATIONS_PER_CASE} különböző
felhasználói helyzetet.

FONTOS:
A döntési target változatlan marad.

Target:
{expected}

Csak a helyzet szövege változhat.

Ne találj ki:
- új toolt
- új szerepkört
- új policy szabályt
- végrehajtott műveletet

Válaszolj JSONL formában.
"""


            rendered = tokenizer.apply_chat_template(
                [
                    {
                        "role":"user",
                        "content":prompt
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )


            inputs = tokenizer(
                rendered,
                return_tensors="pt"
            ).to("cuda")


            with torch.inference_mode():

                result = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                )


            text = tokenizer.decode(
                result[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )


            record = {
                "source_case": case.get("case_id"),
                "generated_text": text,
                "target": case.get("expected"),
            }


            out.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

            generated += 1


            print(
                f"{idx}/{len(cases)} generated={generated}"
            )


    print(
        json.dumps(
            {
                "status":"COMPLETE",
                "cases":len(cases),
                "generated":generated,
                "output":str(OUTPUT),
            },
            indent=2
        )
    )


if __name__ == "__main__":
    main()

