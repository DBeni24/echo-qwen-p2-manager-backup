# AI SERVER 3090 - RECOVERY PROMPT

Use this file as the handoff context for the next ChatGPT session.

## Shutdown state

- Machine: AI-SERVER / RTX 3090 24GB
- Generator: generate_manager_policy_dataset_v04_final.py
- Requested count: 3000
- Node: ai_server_3090
- Seed: 3090
- Active worker PID during salvage: 21432
- The machine was intentionally shut down before reaching 3000 records.

## Critical V04 behavior

The V04 generator stores accepted records in the Python RAM list named records.
The accepted output JSONL is normally written only after the requested count is reached.
Therefore the incomplete accepted dataset was not safely available as a normal JSONL at shutdown time.

## Private RAM salvage

A full process-memory dump was successfully created BEFORE shutdown:

D:\echo_backups\ai_server_3090_20260825\shutdown_salvage\python_21432_DBGHELP_FULL.dmp

MiniDumpWriteDump returned True.
Dump size at completion: 24768274506 bytes (~23.067 GiB).

IMPORTANT: THE DMP IS PRIVATE. NEVER UPLOAD IT TO PUBLIC GITHUB.
It may contain arbitrary process memory and sensitive material.

The dump SHA256 may have been calculated separately after this note was written.
Check the local salvage directory/manifest before doing recovery work.

## Final debug snapshot

Private final debug snapshot:
D:\echo_backups\ai_server_3090_20260825\shutdown_salvage\generated_manager_v04_ai_server_3090_debug_SHUTDOWN_FINAL.jsonl

Lines: 264
Bytes: 603162
SHA256: A02EA03FE8D0ED92D322252D9A54862E99D77B266033136930B8E5DDD63DB814

The debug JSONL contains rejected/error generations. It is NOT the accepted dataset.

## Existing public GitHub evidence

Directory:
audit/ai_server_3090_review_20260825/

Already contains:
- generated_manager_v04_ai_server_3090_snapshot_REDACTED.jsonl
- generate_manager_policy_dataset_v04_final_RUNNING_COPY.py
- PROVENANCE.txt

The earlier public redacted snapshot contains 261 valid debug/reject JSONL rows.

## NEXT SESSION PRIORITY

1. Preserve the original .dmp byte-for-byte.
2. Compute/verify SHA256 of the .dmp.
3. Work from a COPY of the dump when attempting recovery.
4. Attempt to recover the in-memory Python records list / accepted V04 records.
5. Validate every recovered record before using it.
6. Do NOT treat recovered V04 records as train-ready.
7. Audit/canonicalize them into the current ManagerDecision schema.
8. Label records TRAIN_READY / REPAIRABLE / SEMANTIC_REJECT / REVIEW.
9. Deduplicate and use leakage-safe scenario/family splits before training.

## Known V04 problems

- only user message + separate partial target
- target lacks the full canonical ManagerDecision fields
- hard-coded scenario-to-target mapping
- prompt echo / think leakage in rejected outputs
- fabricated system/AION details can appear
- prompt asks for 100 chars but validator only enforces 50
- generator decodes prompt + completion together

Do not overwrite the historical raw evidence. Create repaired derivatives with provenance.
