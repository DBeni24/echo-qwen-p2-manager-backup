import json
import hashlib
import re
from pathlib import Path
from collections import Counter


ROOT = Path("/workspace/echo")

DATASET = ROOT / "dataset/p2_manager_upgrade"

TRAIN = DATASET / "p2_train_final.jsonl"


REQUIRED_TOP_KEYS = [
    "case_id",
    "family",
    "messages",
]


REQUIRED_TARGET_KEYS = [
    "decision",
    "risk_level",
    "approval_required",
    "validation_required",
]


SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{20,}",
    r"AIza[A-Za-z0-9_-]{20,}",
    r"-----BEGIN .* PRIVATE KEY-----",
]


def read_jsonl(path):
    with path.open() as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


rows = read_jsonl(TRAIN)


errors = []

case_ids = []
prompt_hashes = []

for idx, row in enumerate(rows):

    for key in REQUIRED_TOP_KEYS:
        if key not in row:
            errors.append(
                f"row_{idx}:missing_{key}"
            )

    case_ids.append(
        row.get("case_id")
    )


    messages = row.get("messages")

    if not isinstance(messages, list):
        errors.append(
            f"row_{idx}:invalid_messages"
        )
        continue


    prompt_text = "\n".join(
        x.get("content","")
        for x in messages
    )


    prompt_hashes.append(
        hashlib.sha256(
            prompt_text.encode()
        ).hexdigest()
    )


    target = row.get("target")

    if target:

        for key in REQUIRED_TARGET_KEYS:
            if key not in target:
                errors.append(
                    f"row_{idx}:missing_target_{key}"
                )


    for pattern in SECRET_PATTERNS:
        if re.search(
            pattern,
            prompt_text
        ):
            errors.append(
                f"row_{idx}:secret_pattern"
            )


duplicate_case_ids = [
    x for x,c in Counter(case_ids).items()
    if c > 1
]


duplicate_prompt_hashes = [
    x for x,c in Counter(prompt_hashes).items()
    if c > 1
]


report = {
    "dataset": "P2_MANAGER_FINAL",
    "records": len(rows),
    "schema_errors": errors,
    "duplicate_case_ids": duplicate_case_ids,
    "duplicate_prompt_hashes": duplicate_prompt_hashes,
    "family_distribution": Counter(
        r.get("family")
        for r in rows
    ),
    "status": (
        "PASS"
        if not errors
        and not duplicate_case_ids
        else "FAIL"
    )
}


(DATASET / "P2_QUALITY_GATE_REPORT.json").write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    )
)


print(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    )
)
