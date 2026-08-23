import json
import hashlib
from pathlib import Path
from collections import Counter


ROOT = Path("/workspace/echo")

DATASET = ROOT / "dataset/p2_manager_upgrade"

INPUTS = [
    DATASET / "train_sanitized.jsonl",
    DATASET / "p2_policy_cases.jsonl",
]

OUT = DATASET / "p2_train_final.jsonl"


def read_jsonl(path):
    with path.open() as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


rows = []

for path in INPUTS:
    rows.extend(read_jsonl(path))


case_ids = [
    r["case_id"]
    for r in rows
]

duplicates = [
    x for x, c in Counter(case_ids).items()
    if c > 1
]


with OUT.open("w") as f:
    for row in rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False
            ) + "\n"
        )


sha = hashlib.sha256(
    OUT.read_bytes()
).hexdigest()


report = {
    "dataset": "P2_MANAGER_FINAL",
    "records": len(rows),
    "source_files": [
        str(x.name)
        for x in INPUTS
    ],
    "duplicate_case_ids": duplicates,
    "duplicate_count": len(duplicates),
    "families": Counter(
        r.get("family")
        for r in rows
    ),
    "sha256": sha,
}


(DATASET / "P2_FINAL_DATASET_REPORT.json").write_text(
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
