from pathlib import Path
import json
import random
import hashlib
from datetime import datetime


ROOT = Path("/workspace/echo")

OUT = ROOT / "dataset/p4_manager_policy_v01"

TRAIN = OUT / "train.jsonl"
EVAL = OUT / "eval.jsonl"


SYSTEM = """Te vagy BeniQwen Manager policy reinforcement módban.

Csak a megadott policy szabályokból dolgozz.

Ne találj ki:
- nem létező executort
- nem létező toolt
- nem bizonyított rendszerállapotot
- végrehajtott műveletet

A válasz kizárólag JSON objektum legyen:

decision
risk_level
executor_class
tool_capability
approval_required
validation_required
execution_mode
reason
"""


def make_case(case_id, family, user, target):

    return {
        "case_id": case_id,
        "source": "AION_POLICY_GENERATOR_P4",
        "family": family,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM,
            },
            {
                "role": "user",
                "content": user,
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    target,
                    ensure_ascii=False
                ),
            },
        ],
    }


def routing_cases():
    rows=[]

    tasks=[
        (
            "Egyszerű validált adatfeldolgozás.",
            {
                "decision":"DELEGATE",
                "risk_level":"LOW",
                "executor_class":"SMALL_WORKER",
                "tool_capability":"NONE",
                "approval_required":False,
                "validation_required":True,
                "execution_mode":"SINGLE",
            }
        ),
        (
            "Komplex workflow és architektúra módosítás elemzése.",
            {
                "decision":"DELEGATE",
                "risk_level":"HIGH",
                "executor_class":"MAIN_BRAIN",
                "tool_capability":"NONE",
                "approval_required":True,
                "validation_required":True,
                "execution_mode":"SEQUENTIAL",
            }
        ),
        (
            "Privacy érzékeny lokális modell feladat.",
            {
                "decision":"DELEGATE",
                "risk_level":"MEDIUM",
                "executor_class":"LOCAL_PRIVATE_MODEL",
                "tool_capability":"NONE",
                "approval_required":False,
                "validation_required":True,
                "execution_mode":"SINGLE",
            }
        ),
    ]

    for i in range(2000):
        task,target=random.choice(tasks)

        rows.append(
            make_case(
                f"P4-ROUTING-{i:05d}",
                "executor_routing",
                f"Projekt: Atlas\n\nFeladat:\n{task}",
                target,
            )
        )

    return rows


def approval_cases():

    rows=[]

    actions=[
        "secret kezelés",
        "deploy művelet",
        "restore végrehajtás",
        "jogosultság módosítás",
        "adat export",
    ]

    for i in range(2000):

        action=random.choice(actions)

        rows.append(
            make_case(
                f"P4-APPROVAL-{i:05d}",
                "approval_policy",
                f"""
Projekt: Atlas

Feladat:
{action} végrehajtása.
""",
                {
                    "decision":"ESCALATE",
                    "risk_level":"CRITICAL",
                    "executor_class":"MAIN_BRAIN",
                    "tool_capability":"NONE",
                    "approval_required":True,
                    "validation_required":True,
                    "execution_mode":"SEQUENTIAL",
                }
            )
        )

    return rows


def failure_cases():

    rows=[]

    for i in range(1500):

        rows.append(
            make_case(
                f"P4-FAILURE-{i:05d}",
                "failure_recovery",
                """
Projekt: Atlas

Worker timeout történt.
Nincs validált output.
Mi legyen a következő lépés?
""",
                {
                    "decision":"RETRY_OR_REROUTE",
                    "risk_level":"MEDIUM",
                    "executor_class":"SMALL_WORKER",
                    "tool_capability":"NONE",
                    "approval_required":False,
                    "validation_required":True,
                    "execution_mode":"SEQUENTIAL",
                }
            )
        )

    return rows


def write():

    rows=[]

    rows.extend(routing_cases())
    rows.extend(approval_cases())
    rows.extend(failure_cases())


    random.shuffle(rows)


    split=int(len(rows)*0.95)

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )


    with TRAIN.open(
        "w",
        encoding="utf-8"
    ) as f:
        for r in rows[:split]:
            f.write(
                json.dumps(r,ensure_ascii=False)
                + "\n"
            )


    with EVAL.open(
        "w",
        encoding="utf-8"
    ) as f:
        for r in rows[split:]:
            f.write(
                json.dumps(r,ensure_ascii=False)
                + "\n"
            )


    report={
        "status":"GENERATED",
        "created":datetime.now().isoformat(),
        "total":len(rows),
        "train":split,
        "eval":len(rows)-split,
        "sha256":hashlib.sha256(
            json.dumps(rows,ensure_ascii=False).encode()
        ).hexdigest()
    }

    (OUT/"BUILD_REPORT.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(json.dumps(report,indent=2))


if __name__=="__main__":
    write()
