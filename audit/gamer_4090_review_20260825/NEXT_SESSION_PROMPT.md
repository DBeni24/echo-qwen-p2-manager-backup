# GAMER RTX 4090 V04 RECOVERY HANDOFF

## Historical run
- Generator: generate_manager_policy_dataset_v04_lmstudio.py
- Requested count: 5000
- Node: gamer_4090
- Seed: 4090
- Historical Python PID: 46356
- Working directory: H:\AION\echo-qwen-p2-manager-backup
- The V04 process was intentionally stopped before reaching 5000.

## Salvage
- Final private Python RAM dump exists locally.
- Final dump SHA256:
FFA1588B5AB9903A8B4A3F0A9B16FE0B430CE61D75879780660B68A48D4C4DB1
- Final dump size: 97361408 bytes
- NEVER upload the dump to public GitHub.

## Debug snapshot
- Public redacted debug snapshot: 616 JSONL records
- JSON validation: 616 good / 0 bad
- SHA256:
148919E98E247FCB5649B617FE17980BBA00C6A72A7094B2213080B8AE1384F6

## Important interpretation
- The 616 rows are debug/rejected generations, NOT the accepted training dataset.
- V04 may have held accepted records only in the Python records list in RAM.
- Recover accepted records from a COPY of the private final dump.
- Never modify the original dump.

## Known historical V04 issues
- old partial ManagerDecision target schema
- user-only messages plus separate target
- hard scenario-to-target bindings
- rejected generations contain reasoning/prompt leakage
- generated AION/system facts may be fabricated
- dataset is NOT directly train-ready

## Recovery workflow
1. Verify private dump SHA256.
2. Work from a copy of the dump.
3. Attempt recovery of Python in-memory records.
4. Preserve recovered raw records unchanged.
5. Create a separate audited derivative.
6. Classify TRAIN_READY / REPAIRABLE / SEMANTIC_REJECT / REVIEW.
7. Canonicalize to the full ManagerDecision schema.
8. Deduplicate and make leakage-safe family/scenario splits.

## Security
A local LM Studio API credential was visible during operational inspection.
Do not copy process command lines or credentials into public artifacts.
Rotate/recreate the local LM Studio credential before future LM Studio use.
