
import argparse
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


ROOT = Path(r"C:\workspace\echo-qwen-p2-manager-backup")
MODEL = Path(r"C:\Users\dbene\Models\Qwen3.5-9B-HF")


SCENARIOS = [
    "workflow_planning",
    "rag_conflict",
    "false_state_detection",
    "tool_selection",
    "approval_required",
    "secret_handling",
    "checkpoint_review",
    "provider_routing",
    "worker_assignment",
]


TARGETS = [
    {
        "decision": "SELF",
        "risk_level": "LOW",
        "executor_class": "SELF",
        "execution_mode": "SINGLE",
    },
    {
        "decision": "DELEGATE",
        "risk_level": "HIGH",
        "executor_class": "MAIN_BRAIN",
        "execution_mode": "SINGLE",
    },
    {
        "decision": "USE_TOOL",
        "risk_level": "MEDIUM",
        "executor_class": "SMALL_WORKER",
        "execution_mode": "SINGLE",
    },
    {
        "decision": "ESCALATE",
        "risk_level": "CRITICAL",
        "executor_class": "NONE",
        "execution_mode": "NONE",
    },
]


def sha256(path):

    h = hashlib.sha256()

    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)

    return h.hexdigest()


def extract_json(text):

    text = text.replace("```json", "")
    text = text.replace("```", "")

    decoder = json.JSONDecoder()

    candidates = []

    for i, c in enumerate(text):
        if c == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                candidates.append(obj)
            except Exception:
                pass

    if not candidates:
        raise ValueError("NO_JSON")

    return candidates[-1]


def validate(item):

    user=str(
        item.get("user","")
    ).strip()

    scenario=item.get("scenario")

    if len(user)<50:
        raise ValueError("USER_TOO_SHORT")

    if not scenario:
        raise ValueError("NO_SCENARIO")

    if scenario not in SCENARIOS:
        raise ValueError("BAD_SCENARIO")

    banned=[
        "konkrét magyar AION rendszerfeladat",
        "feladat leírás",
        "task description"
    ]

    for x in banned:
        if x.lower() in user.lower():
            raise ValueError("PLACEHOLDER")

    return user,scenario


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--count",type=int,default=3)
    parser.add_argument("--node",default="cloud")
    parser.add_argument("--seed",type=int,default=1)

    args=parser.parse_args()

    random.seed(args.seed)


    output=ROOT / (
        f"dataset/p2_manager_upgrade/"
        f"generated_manager_v04_{args.node}.jsonl"
    )

    debug=ROOT / (
        f"dataset/p2_manager_upgrade/"
        f"generated_manager_v04_{args.node}_debug.jsonl"
    )


    tokenizer=AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True
    )


    model=AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model.eval()


    records=[]
    seen=set()
    stats=Counter()


    with debug.open(
        "w",
        encoding="utf-8"
    ) as dbg:


        while len(records)<args.count:


            scenario=random.choice(
                SCENARIOS
            )

            scenario_rules = {
                "rag_conflict": "Két vagy több forrás eltérő információt tartalmazzon, és dönteni kelljen a hitelességről.",
                "false_state_detection": "A rendszer egy nem bizonyított állapotot jelent sikeresnek, ezt ellenőrizni kell.",
                "approval_required": "Kritikus művelet legyen, amely emberi jóváhagyást igényel.",
                "secret_handling": "Titkok, API kulcsok vagy érzékeny adatok biztonságos kezelése legyen a feladat.",
                "tool_selection": "A feladatnál megfelelő tool vagy worker kiválasztása szükséges.",
                "worker_assignment": "A feladat delegálási és worker kiválasztási döntést igényel.",
                "workflow_planning": "Többlépéses workflow tervezése szükséges.",
                "checkpoint_review": "Checkpoint vagy validáció ellenőrzési helyzet legyen.",
                "provider_routing": "Modell vagy provider kiválasztási döntés szükséges."
            }

            prompt=f"""
Te vagy az AION Manager dataset készítője.

Generálj egy valódi magyar rendszerfeladatot.

Kötelező scenario:
{scenario}

A scenario követelménye:
{scenario_rules[scenario]}

A user mező:
- minimum 100 karakter
- konkrét technikai helyzet
- AION rendszerhez kapcsolódó döntési probléma

A válasz kizárólag JSON legyen:

{{
"user":"részletes rendszerfeladat",
"scenario":"{scenario}"
}}

Tilos:
- magyarázat
- markdown
- sablon szöveg
- instrukció visszamásolása
"""


            inp=tokenizer(
                prompt,
                return_tensors="pt"
            ).to(model.device)


            with torch.no_grad():

                out=model.generate(
                    **inp,
                    max_new_tokens=500,
                    temperature=0.8,
                    do_sample=True
                )


            raw=tokenizer.decode(
                out[0],
                skip_special_tokens=True
            )


            stats["calls"]+=1


            try:

                item=extract_json(raw)

                user,scenario=validate(item)


                if user.lower() in seen:
                    stats["duplicate"]+=1
                    continue


                seen.add(
                    user.lower()
                )


                SCENARIO_TARGETS = {
                    "rag_conflict": {
                        "decision": "ESCALATE",
                        "risk_level": "HIGH",
                        "executor_class": "MAIN_BRAIN",
                        "execution_mode": "SINGLE"
                    },
                    "false_state_detection": {
                        "decision": "ESCALATE",
                        "risk_level": "HIGH",
                        "executor_class": "MAIN_BRAIN",
                        "execution_mode": "SINGLE"
                    },
                    "approval_required": {
                        "decision": "ESCALATE",
                        "risk_level": "HIGH",
                        "executor_class": "MAIN_BRAIN",
                        "execution_mode": "SINGLE"
                    },
                    "secret_handling": {
                        "decision": "ESCALATE",
                        "risk_level": "CRITICAL",
                        "executor_class": "NONE",
                        "execution_mode": "NONE"
                    },
                    "tool_selection": {
                        "decision": "USE_TOOL",
                        "risk_level": "MEDIUM",
                        "executor_class": "SMALL_WORKER",
                        "execution_mode": "SINGLE"
                    },
                    "worker_assignment": {
                        "decision": "DELEGATE",
                        "risk_level": "HIGH",
                        "executor_class": "MAIN_BRAIN",
                        "execution_mode": "SINGLE"
                    },
                    "workflow_planning": {
                        "decision": "DELEGATE",
                        "risk_level": "HIGH",
                        "executor_class": "MAIN_BRAIN",
                        "execution_mode": "SINGLE"
                    },
                    "checkpoint_review": {
                        "decision": "ESCALATE",
                        "risk_level": "HIGH",
                        "executor_class": "MAIN_BRAIN",
                        "execution_mode": "SINGLE"
                    },
                    "provider_routing": {
                        "decision": "USE_TOOL",
                        "risk_level": "MEDIUM",
                        "executor_class": "SMALL_WORKER",
                        "execution_mode": "SINGLE"
                    }
                }

                target = SCENARIO_TARGETS[scenario]


                records.append(
                    {
                        "messages":[
                            {
                                "role":"user",
                                "content":user
                            }
                        ],
                        "target":target,
                        "metadata":{
                            "generator":
                            "Qwen3.5-9B",
                            "version":
                            "V04_BATCH_V05",
                            "node":
                            args.node,
                            "scenario":
                            scenario,
                            "created":
                            datetime.now(
                                timezone.utc
                            ).isoformat()
                        }
                    }
                )


                stats["accepted"]+=1


            except Exception as e:

                stats["rejected"]+=1

                dbg.write(
                    json.dumps(
                        {
                            "reason":str(e),
                            "raw":raw
                        },
                        ensure_ascii=False
                    )+"\n"
                )


            print(
                f"records={len(records)}/{args.count} "
                f"stats={dict(stats)}",
                flush=True
            )


    with output.open(
        "w",
        encoding="utf-8"
    ) as f:

        for r in records:

            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False
                )+"\n"
            )


    print(json.dumps(
        {
            "status":"COMPLETE",
            "records":len(records),
            "stats":dict(stats),
            "output":str(output),
            "sha256":sha256(output)
        },
        indent=2,
        ensure_ascii=False
    ))


if __name__=="__main__":
    main()
