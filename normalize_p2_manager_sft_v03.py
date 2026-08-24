import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/workspace/echo")

SRC = ROOT / "dataset/p2_manager_upgrade/p2_manager_sft_v02.jsonl"

OUT = ROOT / "dataset/p2_manager_upgrade/p2_manager_sft_v03_normalized.jsonl"
MANIFEST = ROOT / "dataset/p2_manager_upgrade/p2_manager_sft_v03_normalized_manifest.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_value(value, field, metadata):
    if isinstance(value, list):
        if len(value) == 0:
            return None

        metadata.setdefault("fallback_values", {})

        if len(value) > 1:
            metadata["fallback_values"][field] = value[1:]

        return value[0]

    return value


rows = []
stats = {
    "total": 0,
    "target_existing": 0,
    "expected_migrated": 0,
    "list_normalized": 0,
}


with SRC.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        r = json.loads(line)

        stats["total"] += 1

        metadata = dict(r.get("metadata", {}))
        normalization = {}

        if "target" in r:
            target = dict(r["target"])
            stats["target_existing"] += 1

        elif "expected" in r:
            target = dict(r["expected"])
            stats["expected_migrated"] += 1
            normalization["source"] = "expected_to_target"

        else:
            continue

        for field in [
            "decision",
            "risk_level",
            "executor_class",
            "execution_mode",
        ]:
            if field in target and isinstance(target[field], list):
                stats["list_normalized"] += 1
                target[field] = normalize_value(
                    target[field],
                    field,
                    normalization
                )

        metadata["normalization"] = normalization

        out = {
            "messages": r["messages"],
            "target": target,
            "metadata": metadata,
        }

        if "case_id" in r:
            out["case_id"] = r["case_id"]

        if "family" in r:
            out["family"] = r["family"]

        rows.append(out)


with OUT.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False
            ) + "\n"
        )


manifest = {
    "status": "CREATED",
    "created": datetime.now(timezone.utc).isoformat(),
    "input": str(SRC),
    "output": str(OUT),
    "records": len(rows),
    "stats": stats,
    "sha256": sha256(OUT),
}


MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print(json.dumps(manifest, indent=2, ensure_ascii=False))
