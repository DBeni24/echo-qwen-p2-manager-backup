import argparse
import hashlib
import json
import os
import random
import signal
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import generate_manager_policy_dataset_v05_safe_v01 as base
import generate_manager_policy_dataset_v05_safe_v12_frozen as v12
import generate_manager_policy_dataset_v05_safe_v13_frozen as v13
import generate_manager_policy_dataset_v05_safe_v20_quality_frozen as v20
import generate_manager_policy_dataset_v05_safe_v21_repair_core_frozen as repair


VERSION = "V05_SAFE_POLICY_V21_V20_QUALITY_RUNTIME_RC1"


# ======================================================================
# Frozen semantic stack
# ======================================================================

base.VERSION = VERSION
base.sample_policy_facts = v12.v10.sample_policy_facts
base.derive_target = v12.derive_target
base.build_generation_prompt = v12.v10.v9.build_generation_prompt


# ======================================================================
# MODEL INFERENCE
# ======================================================================

def generate_user(
    model,
    tokenizer,
    prompt,
    seed,
):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    inp = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=520,
            temperature=0.75,
            top_p=0.90,
            do_sample=True,
            repetition_penalty=1.08,
        )

    generated_tokens = out[
        0
    ][
        inp.input_ids.shape[1]:
    ]

    raw = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    item = base.extract_json(raw)

    user = item.get(
        "user",
        "",
    )

    if not isinstance(user, str):
        raise ValueError(
            "USER_NOT_STRING"
        )

    return user.strip()


# ======================================================================
# DURABLE V15 RESUME RECONSTRUCTION
# ======================================================================

def load_existing_v15(
    live_path,
    reject_path,
):
    (
        seen,
        accepted,
        rejected,
        last_call,
        decisions,
        risks,
        scenarios,
    ) = v13.load_existing_v13(
        live_path,
        reject_path,
    )

    extra = Counter()

    def add_runtime_fields(
        obj,
        accepted_outcome,
    ):
        if accepted_outcome:
            src = obj["metadata"]
        else:
            src = obj

        inference_count = int(
            src.get(
                "inference_calls_for_candidate",
                1,
            )
        )

        repair_attempted = bool(
            src.get(
                "repair_attempted",
                False,
            )
        )

        initial_attempted = int(
            src.get(
                "initial_inference_calls_for_candidate",
                1,
            )
        )

        if initial_attempted != 1:
            raise RuntimeError(
                "V15_INVALID_INITIAL_INFERENCE_COUNT:"
                f"{initial_attempted}"
            )

        expected_inference_count = (
            2
            if repair_attempted
            else 1
        )

        if (
            inference_count
            != expected_inference_count
        ):
            raise RuntimeError(
                "V15_INFERENCE_COUNT_MISMATCH:"
                f"{inference_count}:"
                f"{repair_attempted}"
            )

        extra[
            "initial_inference_calls"
        ] += 1

        extra[
            "inference_calls"
        ] += inference_count

        if repair_attempted:
            extra[
                "repair_calls"
            ] += 1

            if accepted_outcome:
                extra[
                    "repair_successes"
                ] += 1
            else:
                extra[
                    "repair_failures"
                ] += 1

        elif accepted_outcome:
            extra[
                "direct_accepts"
            ] += 1

        if (
            not accepted_outcome
            and str(
                src.get(
                    "reason",
                    "",
                )
            ).startswith(
                "QUALITY_SEMANTIC_REJECT:"
            )
        ):
            extra[
                "semantic_rejects"
            ] += 1

    for _, obj in (
        v13._jsonl_rows(
            live_path
        )
        or []
    ):
        add_runtime_fields(
            obj,
            True,
        )

    for _, obj in (
        v13._jsonl_rows(
            reject_path
        )
        or []
    ):
        add_runtime_fields(
            obj,
            False,
        )

    stats = Counter()

    stats["accepted"] = accepted
    stats["rejected"] = rejected
    stats["calls"] = last_call
    stats["candidate_calls"] = last_call

    for key, value in extra.items():
        stats[key] = value

    return (
        seen,
        accepted,
        last_call,
        decisions,
        risks,
        scenarios,
        stats,
    )


# ======================================================================
# RUNTIME AUDIT METADATA
# ======================================================================

def runtime_metadata(
    initial_result,
    repair_result,
    inference_calls_for_candidate,
    repair_attempted,
):
    return {
        "initial_quality_classification":
            (
                initial_result[
                    "classification"
                ]
                if initial_result
                is not None
                else None
            ),

        "initial_quality_code":
            (
                initial_result[
                    "code"
                ]
                if initial_result
                is not None
                else None
            ),

        "repair_attempted":
            repair_attempted,

        "repair_final_classification":
            (
                repair_result[
                    "classification"
                ]
                if repair_result
                is not None
                else None
            ),

        "repair_final_code":
            (
                repair_result[
                    "code"
                ]
                if repair_result
                is not None
                else None
            ),

        "initial_inference_calls_for_candidate":
            1,

        "inference_calls_for_candidate":
            inference_calls_for_candidate,
    }


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
        "version":
            VERSION,

        "model":
            str(base.MODEL),

        "policy_version":
            "V05_SAFE_POLICY_V12_REASON_QUALITY",

        "quality_version":
            v20.VERSION,

        "repair_core_version":
            repair.VERSION,

        "count":
            args.count,

        "node":
            args.node,

        "seed":
            args.seed,

        "run_id":
            args.run_id,

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
            "policy_version",
            "quality_version",
            "repair_core_version",
            "node",
            "seed",
            "run_id",
        ]:
            if (
                old.get(key)
                != config.get(key)
            ):
                raise SystemExit(
                    "RESUME_CONFIG_MISMATCH:"
                    + key
                )

    base.atomic_json_write(
        config_path,
        config,
    )

    (
        seen,
        accepted,
        last_call,
        decisions,
        risks,
        scenarios,
        stats,
    ) = load_existing_v15(
        live_path,
        reject_path,
    )

    v13.validate_state_not_ahead(
        state_path,
        accepted,
        last_call,
    )

    original_started_at = (
        v13.load_original_started_at(
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
                    stats["rejected"],

                "resume_last_call":
                    last_call,

                "resume_inference_calls":
                    stats[
                        "inference_calls"
                    ],

                "resume_repair_calls":
                    stats[
                        "repair_calls"
                    ],

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
        AutoTokenizer.from_pretrained(
            base.MODEL,
            trust_remote_code=True,
        )
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
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
                stats[
                    "candidate_calls"
                ]
                + 1
            )

            call_seed = (
                args.seed
                + call_no * 1000003
            )

            repair_seed = (
                call_seed
                + 700000001
            )

            rng = random.Random(
                call_seed
            )

            scenario = rng.choice(
                base.SCENARIOS
            )

            # Policy derivation failures are INTERNAL failures,
            # not dataset rejects. Let them abort the process.
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

            outcome = None
            final_user = None
            reject_reason = None

            initial_result = None
            repair_result = None

            repair_attempted = False
            inference_calls_for_candidate = 0

            # ==================================================
            # GENERATION / CLASSIFICATION
            #
            # No durable write happens inside this try block.
            # ==================================================

            try:
                inference_calls_for_candidate += 1

                initial_user = generate_user(
                    model,
                    tokenizer,
                    prompt,
                    call_seed,
                )

                initial_result = (
                    repair.route_user(
                        initial_user,
                        scenario,
                        facts,
                    )
                )

                if (
                    initial_result["route"]
                    == repair.DIRECT_ACCEPT
                ):
                    final_user = initial_user
                    outcome = "ACCEPT"

                elif (
                    initial_result["route"]
                    == repair.REJECT
                ):
                    reject_reason = (
                        "QUALITY_SEMANTIC_REJECT:"
                        + initial_result[
                            "code"
                        ]
                    )

                    outcome = "REJECT"

                elif (
                    initial_result["route"]
                    == repair.REPAIR
                ):
                    repair_attempted = True

                    repair_prompt = (
                        repair.build_repair_prompt(
                            initial_user,
                            scenario,
                            facts,
                            initial_result,
                        )
                    )

                    inference_calls_for_candidate += 1

                    repaired_user = generate_user(
                        model,
                        tokenizer,
                        repair_prompt,
                        repair_seed,
                    )

                    repair_result = (
                        repair.route_user(
                            repaired_user,
                            scenario,
                            facts,
                        )
                    )

                    # No second repair.
                    if (
                        repair_result["route"]
                        != repair.DIRECT_ACCEPT
                    ):
                        raise ValueError(
                            "REPAIR_NOT_PASS:"
                            f"{repair_result['classification']}:"
                            f"{repair_result['code']}"
                        )

                    final_user = repaired_user
                    outcome = "ACCEPT"

                else:
                    raise ValueError(
                        "UNKNOWN_REPAIR_ROUTE:"
                        + str(
                            initial_result[
                                "route"
                            ]
                        )
                    )

                if outcome == "ACCEPT":
                    user_hash = (
                        hashlib.sha256(
                            base.normalize_text(
                                final_user
                            ).encode(
                                "utf-8"
                            )
                        ).hexdigest()
                    )

                    if user_hash in seen:
                        raise ValueError(
                            "EXACT_NORMALIZED_DUPLICATE"
                        )

            except Exception as exc:
                outcome = "REJECT"
                reject_reason = str(exc)

            audit = runtime_metadata(
                initial_result=
                    initial_result,

                repair_result=
                    repair_result,

                inference_calls_for_candidate=
                    inference_calls_for_candidate,

                repair_attempted=
                    repair_attempted,
            )

            # ==================================================
            # DURABLE COMMIT
            #
            # Critical difference from V13:
            #
            # State/checkpoint failures are NOT interpreted as a
            # generation reject after a durable outcome exists.
            # ==================================================

            if outcome == "ACCEPT":
                record = (
                    base.make_record(
                        user=final_user,
                        scenario=scenario,
                        facts=facts,
                        target=target,
                        args=args,
                        call_no=call_no,
                    )
                )

                record[
                    "metadata"
                ].update(
                    audit
                )

                record[
                    "metadata"
                ][
                    "repair_seed_used"
                ] = (
                    repair_seed
                    if repair_attempted
                    else None
                )

                serialized = json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )

                live.write(
                    serialized
                    + "\n"
                )

                live.flush()
                os.fsync(
                    live.fileno()
                )

                # Only after durable append:
                seen.add(
                    record[
                        "metadata"
                    ][
                        "user_sha256"
                    ]
                )

                accepted += 1

                stats[
                    "accepted"
                ] = accepted

                decisions[
                    target[
                        "decision"
                    ]
                ] += 1

                risks[
                    target[
                        "risk_level"
                    ]
                ] += 1

                scenarios[
                    scenario
                ] += 1

            elif outcome == "REJECT":
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
                        reject_reason,

                    **audit,
                }

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

                stats[
                    "rejected"
                ] += 1

            else:
                raise RuntimeError(
                    "V15_NO_OUTCOME"
                )

            # ==================================================
            # Durable outcome counters
            # ==================================================

            stats[
                "calls"
            ] = call_no

            stats[
                "candidate_calls"
            ] = call_no

            stats[
                "initial_inference_calls"
            ] += 1

            stats[
                "inference_calls"
            ] += (
                inference_calls_for_candidate
            )

            if repair_attempted:
                stats[
                    "repair_calls"
                ] += 1

                if outcome == "ACCEPT":
                    stats[
                        "repair_successes"
                    ] += 1
                else:
                    stats[
                        "repair_failures"
                    ] += 1

            elif outcome == "ACCEPT":
                stats[
                    "direct_accepts"
                ] += 1

            if (
                outcome == "REJECT"
                and str(
                    reject_reason
                ).startswith(
                    "QUALITY_SEMANTIC_REJECT:"
                )
            ):
                stats[
                    "semantic_rejects"
                ] += 1

            # ==================================================
            # STATE AFTER durable outcome.
            #
            # Any failure here aborts; it must NOT create a second
            # outcome for the same candidate.
            # ==================================================

            base.write_state(
                state_path=
                    state_path,

                args=
                    args,

                accepted=
                    accepted,

                stats=
                    stats,

                decisions=
                    decisions,

                risks=
                    risks,

                scenarios=
                    scenarios,

                live_path=
                    live_path,

                status=
                    "RUNNING",

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

                    args=
                        args,

                    stats=
                        stats,

                    decisions=
                        decisions,

                    risks=
                        risks,

                    scenarios=
                        scenarios,

                    started_at=
                        started_at,

                    reason=
                        "PERIODIC",
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
                f"inferences="
                f"{stats['inference_calls']} "
                f"repairs="
                f"{stats['repair_calls']} "
                f"repair_ok="
                f"{stats['repair_successes']} "
                f"rejected="
                f"{stats['rejected']}",
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
            run_dir=
                run_dir,

            live_file_handle=
                live,

            live_path=
                live_path,

            accepted=
                accepted,

            args=
                args,

            stats=
                stats,

            decisions=
                decisions,

            risks=
                risks,

            scenarios=
                scenarios,

            started_at=
                started_at,

            reason=
                final_reason,
        )

        base.write_state(
            state_path=
                state_path,

            args=
                args,

            accepted=
                accepted,

            stats=
                stats,

            decisions=
                decisions,

            risks=
                risks,

            scenarios=
                scenarios,

            live_path=
                live_path,

            status=
                final_reason,

            started_at=
                started_at,
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
                        if final_cp
                        is not None
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
