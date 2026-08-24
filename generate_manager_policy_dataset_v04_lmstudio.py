import argparse
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import requests


ROOT = Path.cwd()

OUTPUT_DIR = ROOT / "dataset_p2_manager_upgrade"

MODEL_ID = "qwen/qwen3.5-9b"

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"


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


TARGETS = {
    "workflow_planning": {
        "decision": "DELEGATE",
        "risk_level": "HIGH",
        "executor_class": "MAIN_BRAIN",
        "execution_mode": "SINGLE",
    },

    "rag_conflict": {
        "decision": "ESCALATE",
        "risk_level": "CRITICAL",
        "executor_class": "NONE",
        "execution_mode": "NONE",
    },

    "false_state_detection": {
        "decision": "DELEGATE",
        "risk_level": "HIGH",
        "executor_class": "MAIN_BRAIN",
        "execution_mode": "SINGLE",
    },

    "tool_selection": {
        "decision": "USE_TOOL",
        "risk_level": "MEDIUM",
        "executor_class": "SMALL_WORKER",
        "execution_mode": "SINGLE",
    },

    "approval_required": {
        "decision": "ESCALATE",
        "risk_level": "CRITICAL",
        "executor_class": "NONE",
        "execution_mode": "NONE",
    },

    "secret_handling": {
        "decision": "ESCALATE",
        "risk_level": "CRITICAL",
        "executor_class": "NONE",
        "execution_mode": "NONE",
    },

    "checkpoint_review": {
        "decision": "DELEGATE",
        "risk_level": "HIGH",
        "executor_class": "MAIN_BRAIN",
        "execution_mode": "SINGLE",
    },

    "provider_routing": {
        "decision": "DELEGATE",
        "risk_level": "HIGH",
        "executor_class": "MAIN_BRAIN",
        "execution_mode": "SINGLE",
    },

    "worker_assignment": {
        "decision": "USE_TOOL",
        "risk_level": "MEDIUM",
        "executor_class": "SMALL_WORKER",
        "execution_mode": "SINGLE",
    },
}


SCENARIO_RULES = {

"rag_conflict":
"""
Két vagy több forrás eltérő rendszerinformációt tartalmazzon.
A Managernek hitelességet kell ellenőriznie.
""",

"false_state_detection":
"""
A rendszer egy állapotot jelent, de bizonyítani kell hogy valóban igaz-e.
""",

"approval_required":
"""
Kritikus művelet legyen:
deploy, jogosultság, törlés, publikus módosítás vagy restore.
""",

"secret_handling":
"""
Titkokkal, API kulcsokkal vagy érzékeny adatokkal kapcsolatos helyzet legyen.
""",

"tool_selection":
"""
Konkrét tool vagy worker kiválasztási döntés szükséges.
""",

"worker_assignment":
"""
Feladat delegálási döntés szükséges.
""",

"workflow_planning":
"""
Többlépéses rendszer workflow tervezés szükséges.
""",

"checkpoint_review":
"""
Checkpoint vagy validáció ellenőrzés szükséges.
""",

"provider_routing":
"""
Modell vagy provider kiválasztási döntési helyzet.
"""
}



def sha256(path):

    h = hashlib.sha256()

    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)

    return h.hexdigest()




def call_lmstudio(prompt):

    import requests
    import time

    url = "http://localhost:1234/v1/chat/completions"

    base = {
        "model": "qwen/qwen3.5-9b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 700,
        "stream": False
    }


    attempts = [
        {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
        {
            "reasoning_effort": "none"
        },
        {}
    ]


    for extra in attempts:

        payload = dict(base)
        payload.update(extra)

        r = requests.post(
            url,
            json=payload,
            timeout=240
        )

        r.raise_for_status()

        data = r.json()

        msg = data["choices"][0]["message"]

        content = msg.get("content")

        if content and content.strip():
            return content


        print("LM Studio reasoning v?laszt adott, ?jrapr?ba...")


        time.sleep(1)


    raise RuntimeError(
        "LM Studio nem adott content mez?t"
    )

def extract_json(text):

    start=text.find("{")
    end=text.rfind("}")

    if start==-1 or end==-1:
        raise ValueError("NO_JSON")

    return json.loads(
        text[start:end+1]
    )



def validate(item):

    user=str(
        item.get("user","")
    ).strip()

    scenario=item.get("scenario")


    if len(user)<100:
        raise ValueError("USER_TOO_SHORT")


    if scenario not in SCENARIOS:
        raise ValueError("BAD_SCENARIO")


    return user,scenario



def main():

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=3
    )

    parser.add_argument(
        "--node",
        default="gamer_4090"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=4090
    )


    args=parser.parse_args()


    random.seed(args.seed)


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    output=OUTPUT_DIR / (
        f"generated_manager_v04_{args.node}.jsonl"
    )


    debug=OUTPUT_DIR / (
        f"generated_manager_v04_{args.node}_debug.jsonl"
    )


    records=[]
    seen=set()

    stats={
        "calls":0,
        "accepted":0,
        "rejected":0,
        "duplicate":0
    }



    with debug.open(
        "w",
        encoding="utf-8"
    ) as dbg:


        while len(records)<args.count:


            scenario=random.choice(
                SCENARIOS
            )


            prompt=f"""
Te vagy az AION Manager dataset készítője.

Generálj egy valódi magyar rendszerfeladatot.

Scenario:
{scenario}

Követelmény:
{SCENARIO_RULES[scenario]}

A user mező:
- minimum 100 karakter
- konkrét technikai helyzet
- AION rendszerhez kapcsolódó döntési probléma

Csak JSON választ adj:

{{
"user":"részletes rendszerfeladat",
"scenario":"{scenario}"
}}
"""


            try:

                stats["calls"]+=1


                raw=call_lmstudio(
                    prompt
                )


                item=extract_json(
                    raw
                )


                user,scenario=validate(
                    item
                )


                if user.lower() in seen:

                    stats["duplicate"]+=1
                    continue


                seen.add(
                    user.lower()
                )


                records.append(
                    {
                        "messages":[
                            {
                                "role":"user",
                                "content":user
                            }
                        ],

                        "target":
                            TARGETS[scenario],

                        "metadata":{
                            "generator":
                                "LM Studio Qwen3.5-9B",

                            "version":
                                "V04_LMSTUDIO",

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


                print(
                    f"records={len(records)}/{args.count} stats={stats}"
                )


            except Exception as e:

                stats["rejected"]+=1

                dbg.write(
                    json.dumps(
                        {
                            "error":str(e),
                            "raw":locals().get("raw","")
                        },
                        ensure_ascii=False
                    )
                    + "\n"
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
                )
                + "\n"
            )


    print(
        json.dumps(
            {
                "status":"COMPLETE",
                "records":len(records),
                "stats":stats,
                "output":str(output),
                "sha256":sha256(output)
            },
            indent=2,
            ensure_ascii=False
        )
    )



if __name__=="__main__":
    main()