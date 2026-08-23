from __future__ import annotations

import json
import hashlib
import random
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


ROOT = Path("/workspace/echo")

INPUT = ROOT / "dataset/p2_manager_upgrade/p2_train_final.jsonl"
OUTPUT = ROOT / "dataset/p2_manager_upgrade/generated_manager_policy_v02.jsonl"
REPORT = ROOT / "dataset/p2_manager_upgrade/generated_manager_policy_v02_report.json"


MODEL_PATH = ROOT / "models/Qwen3.5-9B"

VARIATIONS_PER_CASE = 10


PROJECT_NAMES = [
    "Atlas",
    "Hermes",
    "AION",
    "Nova",
    "Orion",
    "Echo",
]


SYSTEM_PROMPT = """
Te egy AI rendszerfejlesztési adatgenerátor vagy.

Feladat:
A megadott policy döntést megtartva készíts új, természetes nyelvű feladatleírást.

Fontos:
- A target döntést nem módosíthatod.
- Nem találhatsz ki új executort.
- Nem találhatsz ki új toolt.
- Ne írj magyarázatot.
- Csak JSON objektumot adj vissza.

Formátum:

{
"user": "új feladatleírás"
}

A feladat legyen valós AI rendszerfejlesztési helyzet.
"""


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def valid_target(target):
    required = [
        "decision",
        "risk_level",
        "executor_class",
        "approval_required",
        "validation_required",
        "execution_mode",
    ]

    return all(
        x in target
        for x in required
    )


def extract_user(row):
    for m in row["messages"]:
        if m["role"] == "user":
            return m["content"]

    return ""


def generate(model, tokenizer, prompt):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        }
    ]


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to("cuda")


    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )


    decoded = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )


    return decoded.strip()


def parse_json(text):

    try:
        start = text.index("{")
        end = text.rindex("}") + 1

        return json.loads(
            text[start:end]
        )

    except Exception:
        return None


def main():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    rows = load_jsonl(INPUT)


    print(
        "INPUT CASES:",
        len(rows)
    )


    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )


    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    model.eval()


    generated = 0
    rejected = 0


    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as out:


        for index, row in enumerate(rows,1):

            target = row.get("target")


            if not target or not valid_target(target):
                rejected += 1
                continue


            original = extract_user(row)


            for variation in range(
                VARIATIONS_PER_CASE
            ):

                project = random.choice(
                    PROJECT_NAMES
                )


                prompt = f"""
Eredeti feladat:

{original}


Készíts egy új változatot.

Projekt neve:
{project}

A döntési cél:

{json.dumps(
    target,
    ensure_ascii=False
)}

A döntési cél maradjon változatlan.
"""


                answer = generate(
                    model,
                    tokenizer,
                    prompt,
                )


                parsed = parse_json(answer)


                if not parsed:
                    rejected += 1
                    continue


                if "user" not in parsed:
                    rejected += 1
                    continue


                record = {
                    "source_case":
                        row["case_id"],

                    "variation":
                        variation,

                    "messages": [
                        {
                            "role": "system",
                            "content":
                                row["messages"][0]["content"],
                        },
                        {
                            "role": "user",
                            "content":
                                parsed["user"],
                        }
                    ],

                    "target":
                        target,

                    "metadata": {
                        "generated_at":
                            datetime.utcnow().isoformat(),

                        "generator":
                            "Qwen3.5-9B",

                        "source":
                            "P2_MANAGER_POLICY_AUGMENTATION_V02",
                    }
                }


                out.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )


                generated += 1


            print(
                f"[{index}/{len(rows)}]",
                "generated=",
                generated,
                "rejected=",
                rejected,
            )


    report = {

        "status":
            "COMPLETE",

        "input_cases":
            len(rows),

        "generated":
            generated,

        "rejected":
            rejected,

        "variation_per_case":
            VARIATIONS_PER_CASE,

        "output":
            str(OUTPUT),

        "sha256":
            sha256(OUTPUT),

        "created":
            datetime.utcnow().isoformat(),
    }


    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
