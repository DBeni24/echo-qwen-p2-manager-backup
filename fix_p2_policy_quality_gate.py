from pathlib import Path
import json
import hashlib


INPUT = Path(
    "dataset_p2_manager_upgrade/generated_manager_v04_gamer_4090.jsonl"
)

OUTPUT = Path(
    "dataset_p2_manager_upgrade/generated_manager_v04_fixed.jsonl"
)


ESCALATE_SCENARIOS = [
    "secret_handling",
    "rag_conflict",
    "false_state_detection",
    "approval_required",
]


HIGH_PATTERNS = [
    "secret",
    "api kulcs",
    "jelszó",
    "jogosultság módosítás",
    "adatvesztés",
    "restore",
    "backup visszaállítás",
    "hamis állapot",
    "rag konfliktus",
    "jóváhagyás szükséges",
    "kritikus biztonsági",
]


def upgrade_policy(row):

    text = row["messages"][0]["content"].lower()
    scenario = row["metadata"].get("scenario")


    if scenario in ESCALATE_SCENARIOS:
        row["target"] = {
            "decision": "ESCALATE",
            "risk_level": "HIGH",
            "executor_class": "MAIN_BRAIN",
            "execution_mode": "SINGLE",
        }


    elif any(pattern in text for pattern in HIGH_PATTERNS):
        row["target"] = {
            "decision": "ESCALATE",
            "risk_level": "HIGH",
            "executor_class": "MAIN_BRAIN",
            "execution_mode": "SINGLE",
        }


    elif row["target"]["decision"] == "DELEGATE":
        row["target"] = {
            "decision": "DELEGATE",
            "risk_level": "MEDIUM",
            "executor_class": "SMALL_WORKER",
            "execution_mode": "SINGLE",
        }


    return row



def main():

    rows = []

    for line in INPUT.read_text(
        encoding="utf-8"
    ).splitlines():

        if line.strip():
            rows.append(
                upgrade_policy(
                    json.loads(line)
                )
            )


    with OUTPUT.open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


    print(
        {
            "status": "COMPLETE",
            "records": len(rows),
            "output": str(OUTPUT),
            "sha256": hashlib.sha256(
                OUTPUT.read_bytes()
            ).hexdigest(),
        }
    )


if __name__ == "__main__":
    main()
