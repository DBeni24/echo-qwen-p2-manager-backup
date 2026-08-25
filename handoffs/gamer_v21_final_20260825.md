# Gamer RTX 4090 — V21 final run handoff

Date: 2026-08-25

Status: DOKUMENTÁLVA

## Run identity

- Runtime: `V05_SAFE_POLICY_V21_V20_QUALITY_RUNTIME_RC1`
- Run ID: `night_v21_gamer_4090_20260825_seed409023`
- Node: `gamer_4090`
- Stop reason: `MAX_RUNTIME`
- Generator process after completion: stopped
- Error scan: PASS
- Temporary checkpoint files: 0

## Final durable counts

- Accepted: **1637**
- Rejected: **1151**
- Candidate calls: **2788**
- Initial inference calls: **2788**
- Total inference calls: **2863**
- Repair calls: **75**
- Repair successes: **37**
- Direct accepts: **1600**
- Semantic rejects: **795**
- Repair failures: **38**

## Decision distribution

- SELF: 188
- USE_TOOL: 739
- ESCALATE: 599
- DELEGATE: 111

## Risk distribution

- LOW: 380
- MEDIUM: 543
- HIGH: 643
- CRITICAL: 71

## Scenario distribution

- workflow_planning: 168
- provider_routing: 205
- rag_conflict: 219
- tool_selection: 226
- approval_required: 147
- worker_assignment: 200
- false_state_detection: 182
- secret_handling: 123
- checkpoint_review: 167

## Final integrity

Final live dataset SHA256:

`243be382fb9d30e24a034224a33949b7cd0ea1196c4870f66f7d65c691df8259`

Final checkpoint:

`checkpoint_20260825T185421Z_n0001637.jsonl`

The final checkpoint matched the live dataset during the final local integrity audit.

Durable line counts:

- `dataset.live.jsonl`: 1637
- `rejects.private.jsonl`: 1151

Runtime ended cleanly with:

`MAX_RUNTIME_REACHED`

## Private backup

The full private archive is intentionally **not uploaded to this public repository**.

Archive:

`gamer_v21_final_20260825.tar.gz`

Archive size:

`21,256,044 bytes`

Archive SHA256:

`9ac85fd120f82194b1f96fbb788a640a0cc479a3944e962bfde5cc46303cf222`

Verification:

`GAMER_FINAL_BACKUP=PASS`

## Notes

- Windows execution used the frozen V21 runtime with a separate Windows checkpoint compatibility shim.
- The frozen V21 policy/runtime source was not modified for the Windows checkpoint workaround.
- No `.tmp` checkpoint artifacts remained at completion.
- Error check passed.
- These 1637 accepted records are generated V21 corpus records, **not yet declared TRAIN_READY**.
- Before LoRA training the corpus still requires merge, canonicalization, deduplication, policy audit, balancing, and family-based train/eval splitting.
- Private reject payloads and private archives must remain outside this public repository.

## Combined V21 night-run bookkeeping

Cloud V21 accepted: **5295**

Gamer RTX 4090 V21 accepted: **1637**

Combined accepted before merge/dedup/audit:

**6932**
