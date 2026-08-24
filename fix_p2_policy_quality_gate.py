from pathlib import Path
import json
import hashlib


INPUT = Path(
    "dataset_p2_manager_upgrade/generated_manager_v04_gamer_4090.jsonl"
)

OUTPUT = Path(
    "dataset_p2_manager_upgrade/generated_manager_v04_gamer_4090_fixed.jsonl"
)


HIGH_PATTERNS = [
    "secret",
    "api kulcs",
    "jelszó",
    "jogosultság",
    "adatvesztés",
    "restore",
    "backup visszaállítás",
    "hamis állapot",
    "rag konfliktus",
    "approval",
    "jóváhagyás szükséges",
    "kritikus biztonsági"
]


ESCALATE_SCENARIOS = [
    "secret_handling",
    "rag_conflict",
    "false_state_detection",
    "approval_required"
]


def set_escalate(row):

    row["target"] = {
        "decision": "ESCALATE",
        "risk_level": "HIGH",
        "executor_class": "MAIN_BRAIN",
        "execution_mode": "SINGLE"
    }

    return row


def set_delegate(row):

    row["target"] = {
        "decision": "DELEGATE",
        "risk_level": "MEDIUM",
        "executor_class": "SMALL_WORKER",
        "execution_mode": "SINGLE"
    }

    return row


def set_tool(row):

    row["target"] = {
        "decision": "USE_TOOL",
        "risk_level": "MEDIUM",
        "executor_class": "SMALL_WORKER",
        "execution_mode": "SINGLE"
    }

    return row



def upgrade_policy(row):

    text = row["messages"][0]["content"].lower()
    scenario = row["metadata"].get("scenario")

    current_decision = row.get(
        "target",
        {}
    ).get(
        "decision"
    )


    # 1. Már meglévő DELEGATE döntés megtartása
    # Delegálás nem lehet Main Brain
    if current_decision == "DELEGATE":

        return set_delegate(row)


    # 2. Kritikus scenario mindig Main Brain
    if scenario in ESCALATE_SCENARIOS:

        return set_escalate(row)


    # 3. Biztonsági / integritási konfliktus
    if any(
        pattern in text
        for pattern in HIGH_PATTERNS
    ):

        return set_escalate(row)


    # 4. USE_TOOL marad worker oldalon
    if current_decision == "USE_TOOL":

        return set_tool(row)


    return row



def main():

    rows = []


    for line in INPUT.read_text(
        encoding="utf-8"
    ).splitlines():

        if line.strip():

            row = json.loads(line)

            rows.append(
                upgrade_policy(row)
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


    sha = hashlib.sha256(
        OUTPUT.read_bytes()
    ).hexdigest()


    print(
        {
            "status": "COMPLETE",
            "records": len(rows),
            "output": str(OUTPUT),
            "sha256": sha
        }
    )


if __name__ == "__main__":
    main()