from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/workspace/echo/models/Qwen3.5-9B")
ADAPTER = Path("/workspace/echo/output/p0_false_state_lora_v01/adapter_final")

sys.path.insert(0, str(ROOT))

from scorers.scoring import score_case
from runner.run_full470 import (
    loadj,
    loadjl,
    rag_text,
    make_long_context,
    reports,
)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_user_text(case, packs):
    parts = []

    if case.get("rag_pack_id"):
        parts.append(
            "=== RAG/KONTEXTUS ===\n"
            + rag_text(packs[case["rag_pack_id"]])
            + "\n=== RAG/KONTEXTUS VÉGE ==="
        )

    if case.get("context_spec"):
        parts.append(
            "=== HOSSZÚ KONTEXTUS ===\n"
            + make_long_context(case)
            + "\n=== HOSSZÚ KONTEXTUS VÉGE ==="
        )

    parts.append(
        "=== FELADAT ===\n" + case["prompt"]
    )

    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["base", "lora"],
        default="lora",
    )
    ap.add_argument(
        "--new-run",
        action="store_true",
    )
    ap.add_argument(
        "--only-category",
    )
    ap.add_argument(
        "--limit",
        type=int,
    )
    args = ap.parse_args()

    cases = loadjl(
        ROOT / "cases/full470.jsonl"
    )

    packs = {
        x["pack_id"]: x
        for x in loadjl(
            ROOT / "rag_packs/all_rag_packs.jsonl"
        )
    }

    if args.only_category:
        cases = [
            c for c in cases
            if c["category"] == args.only_category
        ]

    if args.limit:
        cases = cases[:args.limit]

    statep = (
        ROOT
        / "runs"
        / f"current_run_hf_{args.mode}.json"
    )

    statep.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if statep.exists() and not args.new_run:
        state = loadj(statep)
        run_dir = Path(state["run_dir"])

    else:
        run_dir = (
            ROOT
            / "runs"
            / (
                f"full470_hf_{args.mode}_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
            )
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        state = {
            "run_dir": str(run_dir),
            "started_at": datetime.now().isoformat(),
            "complete": False,
            "backend": "transformers_hf",
            "mode": args.mode,
        }

        statep.write_text(
            json.dumps(
                state,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    respath = run_dir / "results.jsonl"

    done = {}

    if respath.exists():
        for r in loadjl(respath):
            done[r["case_id"]] = r

    print("RUN:", run_dir)
    print("MODE:", args.mode)
    print(
        "REMAINING:",
        sum(
            c["case_id"] not in done
            for c in cases
        ),
    )

    # Exact benchmark snapshots.
    if not (
        run_dir / "cases_snapshot.jsonl"
    ).exists():
        shutil.copy2(
            ROOT / "cases/full470.jsonl",
            run_dir / "cases_snapshot.jsonl",
        )

        shutil.copy2(
            ROOT
            / "rag_packs/all_rag_packs.jsonl",
            run_dir / "rag_packs_snapshot.jsonl",
        )

        shutil.copy2(
            ROOT / "cases/suite_manifest.json",
            run_dir / "suite_manifest.json",
        )

        shutil.copy2(
            ROOT / "holdout/holdout_manifest.json",
            run_dir / "holdout_manifest.json",
        )

    print("CUDA:", torch.cuda.is_available())
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print(
        "BF16:",
        torch.cuda.is_bf16_supported(),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE,
        local_files_only=True,
    )

    base_model = (
        Qwen3_5ForConditionalGeneration
        .from_pretrained(
            BASE,
            dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=True,
        )
    )

    if args.mode == "lora":
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER,
            is_trainable=False,
        )
    else:
        model = base_model

    model.eval()

    runtime = {
        "backend": "transformers_hf",
        "mode": args.mode,
        "base_path": str(BASE),
        "adapter_path": (
            str(ADAPTER)
            if args.mode == "lora"
            else None
        ),
        "dtype": "bfloat16",
        "device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tokenizer_source": "BASE_MODEL",
        "enable_thinking": False,
        "do_sample": False,
        "use_cache": True,
        "runner_concurrency": 1,
        "checkpoint_state": (
            "CHECKPOINT_CANDIDATE"
            if args.mode == "lora"
            else "BASE_CONTROL"
        ),
    }

    # Artifact fingerprints where available.
    for name, path in {
        "base_config_sha256":
            BASE / "config.json",
        "tokenizer_config_sha256":
            BASE / "tokenizer_config.json",
        "adapter_config_sha256":
            ADAPTER / "adapter_config.json",
        "adapter_model_sha256":
            ADAPTER / "adapter_model.safetensors",
    }.items():
        if path.exists():
            runtime[name] = sha256_file(path)

    (
        run_dir / "runtime_config_hf.json"
    ).write_text(
        json.dumps(
            runtime,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "MODEL_LOAD_PASS VRAM_GB:",
        round(
            torch.cuda.memory_allocated()
            / 1024**3,
            2,
        ),
    )

    events = []

    with respath.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as out:

        for idx, case in enumerate(
            cases,
            1,
        ):
            case_id = case["case_id"]

            if case_id in done:
                continue

            target_ctx = int(
                case["context_length"]
            )

            print(
                f"[{idx:03d}/{len(cases):03d}] "
                f"{case_id} "
                f"{case['category']} "
                f"ctx={target_ctx} ... ",
                end="",
                flush=True,
            )

            r = {
                "case_id": case_id,
                "category": case["category"],
                "subcategory":
                    case["subcategory"],
                "visibility":
                    case["visibility"],
                "severity":
                    case["severity"],
                "backend":
                    "transformers_hf",
                "mode":
                    args.mode,
                "request_meta": {
                    "context_length":
                        target_ctx,
                    "max_output_tokens":
                        case[
                            "max_output_tokens"
                        ],
                    "reasoning":
                        case[
                            "reasoning_mode"
                        ],
                },
            }

            try:
                user_text = build_user_text(
                    case,
                    packs,
                )

                messages = [
                    {
                        "role": "system",
                        "content":
                            case[
                                "system_prompt"
                            ],
                    },
                    {
                        "role": "user",
                        "content":
                            user_text,
                    },
                ]

                rendered = (
                    tokenizer
                    .apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )

                inputs = tokenizer(
                    rendered,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to("cuda")

                input_tokens = int(
                    inputs[
                        "input_ids"
                    ].shape[1]
                )

                max_new = int(
                    case[
                        "max_output_tokens"
                    ]
                )

                if (
                    input_tokens + max_new
                    > target_ctx
                ):
                    r["status"] = (
                        "RESOURCE_ERROR"
                    )
                    r["error"] = (
                        "context_budget_exceeded"
                    )
                    r["stats"] = {
                        "input_tokens":
                            input_tokens,
                    }

                else:
                    torch.cuda.reset_peak_memory_stats()

                    t0 = time.time()

                    with torch.inference_mode():
                        generated_all = (
                            model.generate(
                                **inputs,
                                max_new_tokens=max_new,
                                do_sample=False,
                                use_cache=True,
                            )
                        )

                    elapsed = (
                        time.time() - t0
                    )

                    generated = generated_all[
                        0,
                        input_tokens:
                    ]

                    output_tokens = int(
                        generated.shape[0]
                    )

                    final = tokenizer.decode(
                        generated,
                        skip_special_tokens=True,
                    ).strip()

                    r[
                        "elapsed_wall_seconds"
                    ] = elapsed

                    r[
                        "final_answer"
                    ] = final

                    r[
                        "reasoning_text"
                    ] = ""

                    r[
                        "model_instance_id"
                    ] = (
                        "Qwen3.5-9B+P0-LoRA"
                        if args.mode == "lora"
                        else "Qwen3.5-9B-BASE"
                    )

                    r["stats"] = {
                        "input_tokens":
                            input_tokens,
                        "total_output_tokens":
                            output_tokens,
                        "tokens_per_second":
                            (
                                output_tokens
                                / elapsed
                                if elapsed
                                else None
                            ),
                        "time_to_first_token_seconds":
                            None,
                        "peak_vram_gb":
                            round(
                                torch.cuda
                                .max_memory_allocated()
                                / 1024**3,
                                3,
                            ),
                    }

                    if not final:
                        r["status"] = (
                            "MODEL_FAILURE"
                        )
                        r["score"] = {
                            "status":
                                "MODEL_FAILURE",
                            "reasons": [
                                "empty_final_answer"
                            ],
                            "details": {},
                        }

                    elif (
                        output_tokens
                        >= max_new
                    ):
                        r["status"] = (
                            "OUTPUT_BUDGET_HIT"
                        )
                        r["score"] = {
                            "status":
                                "OUTPUT_BUDGET_HIT",
                            "reasons": [
                                "max_output_tokens_reached"
                            ],
                            "details": {
                                "word_count":
                                    len(
                                        final.split()
                                    )
                            },
                        }

                    else:
                        r["score"] = score_case(
                            case,
                            final,
                        )
                        r["status"] = (
                            r["score"]["status"]
                        )

                    del (
                        inputs,
                        generated_all,
                        generated,
                    )

            except torch.cuda.OutOfMemoryError as e:
                r["status"] = "RESOURCE_ERROR"
                r["error"] = (
                    "CUDA_OOM: "
                    + str(e)
                )
                torch.cuda.empty_cache()

            except Exception as e:
                r["status"] = "RUNTIME_ERROR"
                r["error"] = (
                    type(e).__name__
                    + ": "
                    + str(e)
                )

            print(r["status"])

            out.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

            done[case_id] = r

    results = [
        done[c["case_id"]]
        for c in cases
        if c["case_id"] in done
    ]

    (
        run_dir / "runtime_events.json"
    ).write_text(
        json.dumps(
            events,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Original Full470 reporting logic.
    reports(
        run_dir,
        results,
        cases,
    )

    if len(results) == len(cases):
        state["complete"] = True
        state[
            "completed_at"
        ] = datetime.now().isoformat()

        statep.write_text(
            json.dumps(
                state,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        archive_base = (
            ROOT
            / f"UPLOAD_ME_ECHO_FULL470_HF_{args.mode.upper()}"
        )

        zp = shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=run_dir,
        )

        print("=" * 100)
        print(
            "FULL470_HF_COMPLETE"
        )
        print("MODE:", args.mode)
        print("RESULTS:", len(results))
        print("UPLOAD ZIP:", zp)

    else:
        print(
            "INCOMPLETE - rerun the same command to resume."
        )


if __name__ == "__main__":
    main()
