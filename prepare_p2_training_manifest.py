import json
import hashlib
from pathlib import Path
from collections import Counter


ROOT = Path("/workspace/echo")

DATASET = ROOT / "dataset/p2_manager_upgrade"
RUN = ROOT / "runs/p2_manager_upgrade_01"

SOURCE = DATASET / "p2_train_final.jsonl"

TRAIN_OUT = RUN / "train.jsonl"
EVAL_OUT = RUN / "eval.jsonl"
REGRESSION_OUT = RUN / "regression_reference.jsonl"
MANIFEST = RUN / "TRAIN_MANIFEST.json"
CONFIG = RUN / "TRAIN_CONFIG.json"


def read_jsonl(path):
    with path.open() as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def write_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )


def stable_bucket(case_id):
    digest = hashlib.sha256(
        case_id.encode()
    ).hexdigest()

    return int(digest[:8], 16) % 10


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


rows = read_jsonl(SOURCE)

train = []
eval_rows = []


for row in rows:

    bucket = stable_bucket(
        row["case_id"]
    )

    if bucket == 0:
        eval_rows.append(row)
    else:
        train.append(row)


write_jsonl(
    TRAIN_OUT,
    train
)

write_jsonl(
    EVAL_OUT,
    eval_rows
)


# P1 regression reference külön
regression_source = DATASET / "regression_guard_sanitized.jsonl"

if regression_source.exists():

    regression = read_jsonl(
        regression_source
    )

    write_jsonl(
        REGRESSION_OUT,
        regression
    )

else:
    regression = []


config = {
    "training_type": "LORA_POLICY_REINFORCEMENT",
    "base_dataset": "P2_MANAGER_FINAL",
    "source_records": len(rows),
    "train_records": len(train),
    "eval_records": len(eval_rows),
    "regression_records": len(regression),
    "training_started": False
}


CONFIG.write_text(
    json.dumps(
        config,
        indent=2,
        ensure_ascii=False
    )
)


manifest = {
    "run": "P2_MANAGER_UPGRADE_01",
    "status": "PREPARED_ONLY",
    "files": {
        "train.jsonl": sha256(TRAIN_OUT),
        "eval.jsonl": sha256(EVAL_OUT),
        "regression_reference.jsonl": sha256(REGRESSION_OUT)
    },
    "counts": {
        "source": len(rows),
        "train": len(train),
        "eval": len(eval_rows),
        "regression": len(regression)
    },
    "families": Counter(
        r["family"]
        for r in rows
    )
}


MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False
    )
)


print(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False
    )
)
