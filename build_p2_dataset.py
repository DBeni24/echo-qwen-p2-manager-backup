import json
from pathlib import Path
from collections import Counter

ROOT = Path("/workspace/echo")

RUN = ROOT / "runs/p1_behavior_discovery_01"
OUT = ROOT / "dataset/p2_manager_upgrade"

OUT.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(x) for x in f if x.strip()]


results = read_jsonl(RUN / "results.jsonl")
cases = read_jsonl(RUN / "case_set.jsonl")
transitions = read_jsonl(RUN / "BASE_P0_TRANSITIONS.jsonl")


case_map = {
    x["case_id"]: x
    for x in cases
}


result_map = {}

for r in results:
    result_map[(r["mode"], r["case_id"])] = r


train = []
guard = []
excluded = []


priority_families = {
    "conflicting_worker_outputs",
    "software_project_planning",
    "vague_idea_brainstorm",
    "main_brain_architecture",
}
for t in transitions:
    cid = t["case_id"]

    case = case_map.get(cid)

    if not case:
        excluded.append({
            "case_id": cid,
            "reason": "missing_case"
        })
        continue

    p0 = result_map.get(("p0", cid))
    base = result_map.get(("base", cid))

    if not p0 or not base:
        excluded.append({
            "case_id": cid,
            "reason": "missing_result"
        })
        continue

    record = {
        "case_id": cid,
        "family": t["family"],
        "messages": case["messages"],
        "target": case.get("expected"),
        "metadata": {
            "source": "P1_BEHAVIOR_DISCOVERY_01",
            "p0_score": p0["heuristic_score"],
            "base_score": base["heuristic_score"],
            "relation": t["relation"],
        }
    }

    if t["relation"] == "P0_BETTER":
        train.append(record)

    elif (
        t["family"] in priority_families
        and t["relation"] in ["TIE", "BASE_BETTER"]
    ):
        train.append(record)

    else:
        guard.append(record)


for name, data in [
    ("train.jsonl", train),
    ("regression_guard.jsonl", guard),
    ("excluded_cases.jsonl", excluded),
]:
    with (OUT / name).open("w") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


report = {
    "train_records": len(train),
    "guard_records": len(guard),
    "excluded_records": len(excluded),
    "families": Counter(
        x["family"] for x in train
    )
}


with (OUT / "BUILD_REPORT.json").open("w") as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


print(json.dumps(report, indent=2, ensure_ascii=False))

