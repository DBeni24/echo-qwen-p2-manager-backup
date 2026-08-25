import argparse
import hashlib
import json
import os
import random
import signal
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import generate_manager_policy_dataset_v05_safe_v01 as base
import generate_manager_policy_dataset_v05_safe_v12_frozen as v12


VERSION = "V05_SAFE_POLICY_V13_RESUME_HARDENED"

# ----------------------------------------------------------------------
# Frozen semantic layers.
# ----------------------------------------------------------------------

base.VERSION = VERSION
base.sample_policy_facts = v12.v10.sample_policy_facts
base.derive_target = v12.derive_target
base.build_generation_prompt = v12.v10.v9.build_generation_prompt
base.validate_user_text = v12.v10.v9.validate_user_text


# ======================================================================
# DURABLE RESUME RECONSTRUCTION
#
# Completed candidate identity is reconstructed from BOTH:
#
#   accepted -> dataset.live.jsonl
#   rejected -> rejects.private.jsonl
#
# The logs are the durable truth. state.json may legitimately lag behind
# if the process dies between durable append and state update.
# ======================================================================

def _jsonl_rows(path: Path):
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"CORRUPT_JSONL:{path.name}:line={line_no}:"
                    f"{type(exc).__name__}"
                ) from exc

            if not isinstance(obj, dict):
                raise RuntimeError(
                    f"INVALID_JSONL_OBJECT:{path.name}:line={line_no}"
                )

            yield line_no, obj


def load_existing_v13(
    live_path: Path,
    reject_path: Path,
):
    seen = set()

    accepted = 0
    rejected = 0
    last_call = 0

    decisions = Counter()
    risks = Counter()
    scenarios = Counter()

    # call_no -> outcome
    completed_calls = {}

    # ----------------------------------------------------------
    # Accepted records
    # ----------------------------------------------------------

    for line_no, obj in _jsonl_rows(live_path) or []:
        try:
            user = obj["messages"][0]["content"]
            target = obj["target"]
            metadata = obj["metadata"]

            call_no = int(
                metadata["candidate_call"]
            )

            scenario = metadata["scenario"]

        except Exception as exc:
            raise RuntimeError(
                f"INVALID_LIVE_SCHEMA:line={line_no}"
            ) from exc

        if call_no < 1:
            raise RuntimeError(
                f"INVALID_CANDIDATE_CALL:{call_no}"
            )

        if call_no in completed_calls:
            raise RuntimeError(
                f"DUPLICATE_CANDIDATE_CALL:{call_no}:"
                f"{completed_calls[call_no]}:accepted"
            )

        completed_calls[call_no] = "accepted"

        user_hash = hashlib.sha256(
            base.normalize_text(
                user
            ).encode("utf-8")
        ).hexdigest()

        seen.add(user_hash)

        accepted += 1
        last_call = max(
            last_call,
            call_no,
        )

        decisions[
            target["decision"]
        ] += 1

        risks[
            target["risk_level"]
        ] += 1

        scenarios[
            scenario
        ] += 1

    # ----------------------------------------------------------
    # Rejected candidates
    # ----------------------------------------------------------

    for line_no, obj in _jsonl_rows(reject_path) or []:
        try:
            call_no = int(
                obj["call"]
            )
        except Exception as exc:
            raise RuntimeError(
                f"INVALID_REJECT_SCHEMA:line={line_no}"
            ) from exc

        if call_no < 1:
            raise RuntimeError(
                f"INVALID_REJECT_CALL:{call_no}"
            )

        if call_no in completed_calls:
            raise RuntimeError(
                f"DUPLICATE_CANDIDATE_CALL:{call_no}:"
                f"{completed_calls[call_no]}:rejected"
            )

        completed_calls[call_no] = "rejected"

        rejected += 1

        last_call = max(
            last_call,
            call_no,
        )

    # ----------------------------------------------------------
    # A completed call sequence must be contiguous.
    #
    # If the process died during a model inference, that unfinished
    # call is NOT present in either durable log. It will therefore be
    # retried with the same deterministic seed after resume.
    # ----------------------------------------------------------

    if completed_calls:
        expected = set(
            range(
                1,
                last_call + 1,
            )
        )

        actual = set(
            completed_calls
        )

        missing = sorted(
            expected - actual
        )

        if missing:
            preview = ",".join(
                str(x)
                for x in missing[:20]
            )

            raise RuntimeError(
                "CANDIDATE_CALL_SEQUENCE_GAP:"
                + preview
            )

    return (
        seen,
        accepted,
        rejected,
        last_call,
        decisions,
        risks,
        scenarios,
    )


def load_original_started_at(
    state_path: Path,
):
    if not state_path.exists():
        return None

    try:
        obj = json.loads(
            state_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    value = obj.get(
        "started_at"
    )

    if (
        isinstance(value, str)
        and value.strip()
    ):
        return value

    return None


def validate_state_not_ahead(
    state_path: Path,
    accepted: int,
    last_call: int,
):
    if not state_path.exists():
        return

    obj = json.loads(
        state_path.read_text(
            encoding="utf-8"
        )
    )

    state_accepted = int(
        obj.get(
            "accepted",
            0,
        )
    )

    state_calls = int(
        obj.get(
            "stats",
            {},
        ).get(
            "calls",
            0,
        )
    )

    # The state may lag behind durable append logs.
    # It must never be ahead of them.
    if state_accepted > accepted:
        raise RuntimeError(
            "STATE_AHEAD_OF_LIVE:"
            f"{state_accepted}>{accepted}"
        )

    if state_calls > last_call:
        raise RuntimeError(
            "STATE_AHEAD_OF_DURABLE_CALLS:"
            f"{state_calls}>{last_call}"
        )


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--node",
        default="cloud_rtx6000",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=6000,
    )

    parser.add_argument(
        "--run-id",
        required=True,
    )

    parser.add_argument(
        "--checkpoint-minutes",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=0.0,
        help="0 = nincs időlimit",
    )

    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit(
            "--count must be >= 1"
        )

    if args.checkpoint_minutes <= 0:
        raise SystemExit(
            "--checkpoint-minutes must be > 0"
        )

    run_dir = (
        base.DEFAULT_RUN_ROOT
        / args.run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    live_path = (
        run_dir
        / "dataset.live.jsonl"
    )

    reject_path = (
        run_dir
        / "rejects.private.jsonl"
    )

    state_path = (
        run_dir
        / "state.json"
    )

    config_path = (
        run_dir
        / "config.json"
    )

    config = {
        "version": VERSION,
        "model": str(
            base.MODEL
        ),
        "count": args.count,
        "node": args.node,
        "seed": args.seed,
        "run_id": args.run_id,
        "checkpoint_minutes":
            args.checkpoint_minutes,
        "max_runtime_hours":
            args.max_runtime_hours,
    }

    if config_path.exists():
        old = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )

        for key in [
            "version",
            "model",
            "node",
            "seed",
            "run_id",
        ]:
            if (
                old.get(key)
                != config.get(key)
            ):
                raise SystemExit(
                    f"RESUME_CONFIG_MISMATCH:{key}"
                )

    # count/runtime/checkpoint interval may change on resume.
    base.atomic_json_write(
        config_path,
        config,
    )

    (
        seen,
        accepted,
        rejected,
        last_call,
        decisions,
        risks,
        scenarios,
    ) = load_existing_v13(
        live_path,
        reject_path,
    )

    validate_state_not_ahead(
        state_path,
        accepted,
        last_call,
    )

    stats = Counter()

    stats["accepted"] = (
        accepted
    )

    stats["rejected"] = (
        rejected
    )

    # Backward-compatible name.
    stats["calls"] = (
        last_call
    )

    # Explicit V13 name.
    stats["candidate_calls"] = (
        last_call
    )

    original_started_at = (
        load_original_started_at(
            state_path
        )
    )

    started_at = (
        original_started_at
        or base.utc_now()
    )

    session_start_monotonic = (
        time.monotonic()
    )

    last_checkpoint = (
        session_start_monotonic
    )

    base.STOP_REQUESTED = False

    print(
        json.dumps(
            {
                "status":
                    "STARTING",
                "version":
                    VERSION,
                "run_id":
                    args.run_id,
                "resume_records":
                    accepted,
                "resume_rejected":
                    rejected,
                "resume_last_call":
                    last_call,
                "target_count":
                    args.count,
                "run_dir":
                    str(run_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            base.MODEL,
            trust_remote_code=True,
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            base.MODEL,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    )

    model.eval()

    signal.signal(
        signal.SIGINT,
        base.signal_handler,
    )

    signal.signal(
        signal.SIGTERM,
        base.signal_handler,
    )

    with live_path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as live, reject_path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as rejects:

        while (
            accepted < args.count
            and not base.STOP_REQUESTED
        ):
            elapsed_hours = (
                time.monotonic()
                - session_start_monotonic
            ) / 3600.0

            if (
                args.max_runtime_hours > 0
                and elapsed_hours
                >= args.max_runtime_hours
            ):
                print(
                    "MAX_RUNTIME_REACHED",
                    flush=True,
                )
                break

            call_no = (
                stats["candidate_calls"]
                + 1
            )

            stats[
                "candidate_calls"
            ] = call_no

            stats[
                "calls"
            ] = call_no

            # --------------------------------------------------
            # Deterministic policy candidate identity.
            # --------------------------------------------------

            call_seed = (
                args.seed
                + call_no * 1000003
            )

            rng = random.Random(
                call_seed
            )

            torch.manual_seed(
                call_seed
            )

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    call_seed
                )

            scenario = random.Random(
                call_seed
            ).choice(
                base.SCENARIOS
            )

            # Use the SAME rng state convention as the old main:
            # scenario choice must consume rng before facts sampling.
            rng = random.Random(
                call_seed
            )

            scenario = rng.choice(
                base.SCENARIOS
            )

            facts = (
                base.sample_policy_facts(
                    rng,
                    scenario,
                )
            )

            target = (
                base.derive_target(
                    scenario,
                    facts,
                )
            )

            base.validate_target(
                target
            )

            prompt = (
                base.build_generation_prompt(
                    scenario,
                    facts,
                )
            )

            try:
                inp = tokenizer(
                    prompt,
                    return_tensors="pt",
                ).to(
                    model.device
                )

                with torch.no_grad():
                    out = model.generate(
                        **inp,
                        max_new_tokens=520,
                        temperature=0.75,
                        top_p=0.90,
                        do_sample=True,
                        repetition_penalty=1.08,
                    )

                generated_tokens = (
                    out[0][
                        inp.input_ids.shape[1]:
                    ]
                )

                raw = tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                ).strip()

                item = (
                    base.extract_json(
                        raw
                    )
                )

                user = (
                    base.validate_user_text(
                        item.get(
                            "user",
                            "",
                        )
                    )
                )

                user_hash = (
                    hashlib.sha256(
                        base.normalize_text(
                            user
                        ).encode(
                            "utf-8"
                        )
                    ).hexdigest()
                )

                if user_hash in seen:
                    raise ValueError(
                        "EXACT_NORMALIZED_DUPLICATE"
                    )

                record = (
                    base.make_record(
                        user=user,
                        scenario=scenario,
                        facts=facts,
                        target=target,
                        args=args,
                        call_no=call_no,
                    )
                )

                serialized = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":",
                        ),
                    )
                )

                # ----------------------------------------------
                # Durable accepted outcome.
                # ----------------------------------------------

                live.write(
                    serialized
                    + "\n"
                )

                live.flush()
                os.fsync(
                    live.fileno()
                )

                seen.add(
                    user_hash
                )

                accepted += 1

                stats[
                    "accepted"
                ] = accepted

                decisions[
                    target["decision"]
                ] += 1

                risks[
                    target["risk_level"]
                ] += 1

                scenarios[
                    scenario
                ] += 1

                # State AFTER durable outcome.
                base.write_state(
                    state_path=
                        state_path,
                    args=args,
                    accepted=
                        accepted,
                    stats=stats,
                    decisions=
                        decisions,
                    risks=risks,
                    scenarios=
                        scenarios,
                    live_path=
                        live_path,
                    status="RUNNING",
                    started_at=
                        started_at,
                )

            except Exception as exc:
                stats[
                    "rejected"
                ] += 1

                reject_record = {
                    "created":
                        base.utc_now(),
                    "version":
                        VERSION,
                    "call":
                        call_no,
                    "scenario":
                        scenario,
                    "policy_facts":
                        facts,
                    "reason":
                        str(exc),
                }

                # ----------------------------------------------
                # Durable rejected outcome.
                # Raw generation is deliberately NOT logged.
                # ----------------------------------------------

                rejects.write(
                    json.dumps(
                        reject_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                rejects.flush()
                os.fsync(
                    rejects.fileno()
                )

                # V13 difference:
                # state is also updated after a durable reject.
                base.write_state(
                    state_path=
                        state_path,
                    args=args,
                    accepted=
                        accepted,
                    stats=stats,
                    decisions=
                        decisions,
                    risks=risks,
                    scenarios=
                        scenarios,
                    live_path=
                        live_path,
                    status="RUNNING",
                    started_at=
                        started_at,
                )

            now = time.monotonic()

            if (
                now - last_checkpoint
                >= args.checkpoint_minutes
                * 60.0
            ):
                cp = base.checkpoint(
                    run_dir=
                        run_dir,
                    live_file_handle=
                        live,
                    live_path=
                        live_path,
                    accepted=
                        accepted,
                    args=args,
                    stats=stats,
                    decisions=
                        decisions,
                    risks=risks,
                    scenarios=
                        scenarios,
                    started_at=
                        started_at,
                    reason="PERIODIC",
                )

                print(
                    "CHECKPOINT "
                    f"records={accepted} "
                    f"path={cp}",
                    flush=True,
                )

                last_checkpoint = now

            print(
                f"records={accepted}/{args.count} "
                f"candidate_calls="
                f"{stats['candidate_calls']} "
                f"rejected="
                f"{stats['rejected']} "
                f"decisions="
                f"{dict(decisions)}",
                flush=True,
            )

        final_reason = (
            "SIGNAL"
            if base.STOP_REQUESTED
            else (
                "TARGET_REACHED"
                if accepted
                >= args.count
                else "MAX_RUNTIME"
            )
        )

        final_cp = base.checkpoint(
            run_dir=run_dir,
            live_file_handle=live,
            live_path=live_path,
            accepted=accepted,
            args=args,
            stats=stats,
            decisions=decisions,
            risks=risks,
            scenarios=scenarios,
            started_at=started_at,
            reason=final_reason,
        )

        base.write_state(
            state_path=state_path,
            args=args,
            accepted=accepted,
            stats=stats,
            decisions=decisions,
            risks=risks,
            scenarios=scenarios,
            live_path=live_path,
            status=final_reason,
            started_at=started_at,
        )

    print(
        json.dumps(
            {
                "status":
                    final_reason,
                "run_id":
                    args.run_id,
                "records":
                    accepted,
                "stats":
                    dict(stats),
                "decisions":
                    dict(decisions),
                "risks":
                    dict(risks),
                "scenarios":
                    dict(scenarios),
                "live_file":
                    str(live_path),
                "live_sha256":
                    (
                        base.sha256_file(
                            live_path
                        )
                        if live_path.exists()
                        else None
                    ),
                "final_checkpoint":
                    (
                        str(final_cp)
                        if final_cp is not None
                        else None
                    ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
