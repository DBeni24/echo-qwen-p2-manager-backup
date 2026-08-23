import json
import hashlib
from pathlib import Path
from collections import Counter


ROOT = Path("/workspace/echo")
DATASET = ROOT / "dataset/p2_manager_upgrade"

TRAIN_IN = DATASET / "train.jsonl"
GUARD_IN = DATASET / "regression_guard.jsonl"

TRAIN_OUT = DATASET / "train_sanitized.jsonl"
GUARD_OUT = DATASET / "regression_guard_sanitized.jsonl"
REFERENCE_OUT = DATASET / "manager_capability_reference.jsonl"


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


train_rows = read_jsonl(TRAIN_IN)
guard_rows = read_jsonl(GUARD_IN)


clean_train = []
moved_reference = []

for row in train_rows:
    if isinstance(row.get("target"), dict):
        clean_train.append(row)
    else:
        moved_reference.append(row)


clean_guard = guard_rows + moved_reference


write_jsonl(TRAIN_OUT, clean_train)
write_jsonl(GUARD_OUT, clean_guard)


manager_reference = [
    x for x in moved_reference
    if x.get("family") in [
        "manager_communication",
        "main_brain_architecture",
        "software_project_planning",
        "result_review_and_next_step",
    ]
]

write_jsonl(
    REFERENCE_OUT,
    manager_reference
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


report = {
    "input_train_records": len(train_rows),
    "output_train_records": len(clean_train),
    "moved_to_guard": len(moved_reference),
    "guard_total": len(clean_guard),
    "manager_reference_records": len(manager_reference),
    "families_moved": Counter(
        x.get("family")
        for x in moved_reference
    )
}


(DATASET / "P2_DATASET_SANITIZE_REPORT.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False)
)


manifest = {}

for p in [
    TRAIN_OUT,
    GUARD_OUT,
    REFERENCE_OUT,
    DATASET / "P2_DATASET_SANITIZE_REPORT.json"
]:
    manifest[p.name] = sha256(p)


(DATASET / "SHA256_MANIFEST_SANITIZED.json").write_text(
    json.dumps(manifest, indent=2)
)


print(json.dumps(report, indent=2, ensure_ascii=False))
