from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import (
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,
)


ROOT = Path("/workspace/echo")

INPUT = ROOT / "dataset/p2_manager_upgrade/p2_train_final.jsonl"
OUTPUT = ROOT / "dataset/p2_manager_upgrade/generated_manager_policy_v03.jsonl"
REPORT = ROOT / "dataset/p2_manager_upgrade/generated_manager_policy_v03_report.json"

MODEL_PATH = ROOT / "models/Qwen3.5-9B"

VARIATIONS_PER_CASE = 10
MAX_RETRY = 2


PROJECT_NAMES = [
    "Atlas",
    "Hermes",
    "AION",
    "Echo",
    "Nova",
    "Orion",
]


SYSTEM_PROMPT = """
Te egy AI workflow policy adatgenerátor vagy.

Feladatod:
A megadott AI rendszeres feladatból készíts új természetes nyelvű változatot.

Szabályok:
- A döntési cél nem változhat.
- Nem találhatsz ki új modellt.
- Nem találhatsz ki új toolt.
- Nem adj választ a feladatra.
- Csak új feladatleírást készíts.

A válasz pontosan ebben a formában legyen:

TASK:
egy új feladatleírás magyarul
"""


def load_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def extract_user(row):

    for m in row["messages"]:
        if m["role"] == "user":
            return m["content"]

    return None


def parse_task(text):

    if "TASK:" in text:
        value = text.split(
            "TASK:",
            1
        )[1].strip()

        if len(value) > 20:
            return value


    try:
        start = text.index("{")
        end = text.rindex("}") + 1

        obj = json.loads(
            text[start:end]
        )

        for key in [
            "task",
            "user",
            "instruction",
        ]:
            if key in obj:
                if len(obj[key]) > 20:
                    return obj[key]

    except Exception:
        pass


    return None


def generate_once(
    model,
    tokenizer,
    original,
    target
):

    project = random.choice(
        PROJECT_NAMES
    )

    prompt = f"""
Eredeti feladat:

{original}

Projekt:
{project}

Döntési cél:

{json.dumps(
    target,
    ensure_ascii=False
)}

Készíts új változatot.
"""


    messages = [
        {
            "role":"system",
            "content":SYSTEM_PROMPT
        },
        {
            "role":"user",
            "content":prompt
        }
    ]


    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


    inputs = tokenizer(
        rendered,
        return_tensors="pt",
    ).to("cuda")


    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )


    return tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )


def main():

    rows = load_jsonl(INPUT)

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


        for idx,row in enumerate(rows,1):

            original = extract_user(row)
            target = row.get("target")


            if not original or not target:
                rejected += VARIATIONS_PER_CASE
                continue


            for variation in range(
                VARIATIONS_PER_CASE
            ):

                task = None


                for attempt in range(
                    MAX_RETRY
                ):

                    answer = generate_once(
                        model,
                        tokenizer,
                        original,
                        target,
                    )

                    task = parse_task(
                        answer
                    )

                    if task:
                        break


                if not task:
                    rejected += 1
                    continue


                record = {

                    "source_case":
                        row["case_id"],

                    "variation":
                        variation,

                    "messages":[
                        {
                            "role":"system",
                            "content":
                                row["messages"][0]["content"]
                        },
                        {
                            "role":"user",
                            "content":task
                        }
                    ],

                    "target":
                        target,

                    "metadata":{
                        "generator":
                            "Qwen3.5-9B",
                        "version":
                            "V03",
                        "created":
                            datetime.now(
                                timezone.utc
                            ).isoformat()
                    }
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
                f"[{idx}/{len(rows)}] "
                f"generated={generated} "
                f"rejected={rejected}"
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
            datetime.now(
                timezone.utc
            ).isoformat()
    }


    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
