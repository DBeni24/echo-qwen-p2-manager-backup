from pathlib import Path
import json
import hashlib


ROOT = Path("/workspace/echo")

INPUT_DIR = ROOT / "runs/p2_manager_upgrade_01"
OUTPUT_DIR = INPUT_DIR / "sft"

TRAIN_IN = INPUT_DIR / "train.jsonl"
EVAL_IN = INPUT_DIR / "eval.jsonl"

TRAIN_OUT = OUTPUT_DIR / "train.jsonl"
EVAL_OUT = OUTPUT_DIR / "eval.jsonl"
GUARD_OUT = OUTPUT_DIR / "ambiguous_guard.jsonl"
REPORT_OUT = OUTPUT_DIR / "P2_SFT_PREP_REPORT.json"
MANIFEST_OUT = OUTPUT_DIR / "SHA256_MANIFEST.json"


REQUIRED_KEYS = [
    "decision",
    "risk_level",
    "executor_class",
    "tool_capability",
    "approval_required",
    "validation_required",
    "execution_mode",
]


SCALAR_KEYS = [
    "decision",
    "risk_level",
    "executor_class",
    "tool_capability",
    "approval_required",
    "validation_required",
    "execution_mode",
]


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )


def normalize_target(row):

    target = row.get("target")

    if target is None:
        target = row.get("expected")

    if not isinstance(target, dict):
        return None

    target = dict(target)

    case_id = row.get("case_id", "")

    # P1 ambiguity cases are intentional policy training cases.
    if case_id.startswith("P1-VAL-HALL-") or case_id.startswith("P1-VAL-CONFLICT-"):
        return {
            "decision": "RETRY_OR_REROUTE",
            "risk_level": "MEDIUM",
            "executor_class": "NONE",
            "tool_capability": "NONE",
            "approval_required": False,
            "validation_required": True,
            "execution_mode": "SEQUENTIAL",
            "reason": "Bizonytalan vagy konfliktusos worker eredmény miatt validáció és kontrollált újrarouting szükséges."
        }

    # Policy schema defaults
    target.setdefault(
        "executor_class",
        "NONE"
    )

    target.setdefault(
        "tool_capability",
        "NONE"
    )

    target.setdefault(
        "approval_required",
        False
    )

    target.setdefault(
        "reason",
        "Policy alapján meghatározott döntés."
    )


    # Known policy correction:
    # architecture/workflow redesign requires review boundary
    if (
        row.get("family") == "main_brain_routing"
        and target.get("executor_class") == "MAIN_BRAIN"
    ):
        target.setdefault(
            "approval_required",
            True
        )


    for key in REQUIRED_KEYS:
        if key not in target:
            return None


    for key in SCALAR_KEYS:
        if isinstance(target[key], (list, dict)):
            return None


    return target


def convert(rows, split):

    output = []
    rejected = []
    guard = []


    for row in rows:

        target = normalize_target(row)

        if target is None:

            rejected.append(
                {
                    "case_id": row.get("case_id"),
                    "reason": "INVALID_OR_AMBIGUOUS_TARGET"
                }
            )

            continue


        if row.get("case_id", "").startswith(
            "P1-VAL-HALL-"
        ):
            guard.append(
                {
                    "case_id": row.get("case_id"),
                    "reason": "AMBIGUOUS_POLICY_RESOLVED",
                    "normalized_target": target,
                }
            )


        output.append(
            {
                "id": row["case_id"],
                "messages": [
                    row["messages"][0],
                    row["messages"][1],
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            target,
                            ensure_ascii=False
                        ),
                    },
                ],
                "metadata": {
                    "source": "P2_SFT_PREPARATION",
                    "split": split,
                    "family": row.get("family"),
                },
            }
        )


    return output, rejected, guard


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


train_rows = read_jsonl(TRAIN_IN)
eval_rows = read_jsonl(EVAL_IN)


train, train_rejected, train_guard = convert(
    train_rows,
    "train"
)

eval_data, eval_rejected, eval_guard = convert(
    eval_rows,
    "eval"
)


write_jsonl(
    TRAIN_OUT,
    train
)

write_jsonl(
    EVAL_OUT,
    eval_data
)

write_jsonl(
    GUARD_OUT,
    train_guard + eval_guard
)


report = {
    "status": "PREPARED",
    "input_train": len(train_rows),
    "input_eval": len(eval_rows),
    "output_train": len(train),
    "output_eval": len(eval_data),
    "rejected": len(train_rejected) + len(eval_rejected),
    "guard_records": len(train_guard) + len(eval_guard),
    "rejected_cases": train_rejected + eval_rejected,
}


REPORT_OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_OUT.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


manifest = {}

for file in [
    TRAIN_OUT,
    EVAL_OUT,
    GUARD_OUT,
    REPORT_OUT,
]:
    manifest[file.name] = sha256(file)


MANIFEST_OUT.write_text(
    json.dumps(
        manifest,
        indent=2,
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
