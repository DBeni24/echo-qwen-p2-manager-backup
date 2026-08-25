import argparse
import hashlib
import json
import os
import random
import re
import shutil
import signal
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/workspace/echo")
MODEL = ROOT / "models/Qwen3.5-9B"
DEFAULT_RUN_ROOT = ROOT / "dataset/p2_manager_upgrade/v05_runs"

VERSION = "V05_SAFE_POLICY_V01"

SCENARIOS = [
    "workflow_planning",
    "rag_conflict",
    "false_state_detection",
    "tool_selection",
    "approval_required",
    "secret_handling",
    "checkpoint_review",
    "provider_routing",
    "worker_assignment",
]

SCENARIO_RULES = {
    "workflow_planning":
        "Többlépéses technikai workflow megtervezése szükséges.",
    "rag_conflict":
        "Két vagy több információforrás között valódi ellentmondás vagy bizonytalanság legyen.",
    "false_state_detection":
        "Egy rendszerállítás vagy státusz nincs megfelelően bizonyítva, ezért ellenőrzési döntés kell.",
    "tool_selection":
        "A feladathoz el kell dönteni, szükséges-e és milyen jellegű eszközös ellenőrzés.",
    "approval_required":
        "A helyzetben a művelet kockázata és jóváhagyási igénye lényeges döntési tényező.",
    "secret_handling":
        "Hitelesítő adatok vagy érzékeny adatok biztonságos kezelésével kapcsolatos helyzet legyen; valódi kulcsot ne írj.",
    "checkpoint_review":
        "Egy checkpoint, build, modell vagy állapot validálásáról kell dönteni.",
    "provider_routing":
        "Modell-, szolgáltató- vagy végrehajtási útvonal kiválasztási probléma legyen.",
    "worker_assignment":
        "A feladat végrehajtójának vagy delegálási módjának kiválasztása szükséges.",
}

TOOL_CAPABILITY_BY_SCENARIO = {
    "workflow_planning": "WORKFLOW_INSPECTION",
    "rag_conflict": "RAG_QUERY",
    "false_state_detection": "STATE_INSPECTION",
    "tool_selection": "SYSTEM_DIAGNOSTIC",
    "approval_required": "APPROVAL_GATE",
    "secret_handling": "SECRET_GUARD",
    "checkpoint_review": "CHECKPOINT_VALIDATION",
    "provider_routing": "PROVIDER_LOOKUP",
    "worker_assignment": "WORKER_STATUS",
}

DECISIONS = {"SELF", "DELEGATE", "USE_TOOL", "ESCALATE"}
RISKS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
EXECUTORS = {"SELF", "SMALL_WORKER", "MAIN_BRAIN", "NONE"}
EXECUTION_MODES = {"SINGLE", "NONE"}

STOP_REQUESTED = False


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json_write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def signal_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def extract_json(text: str):
    text = text.replace("```json", "").replace("```", "").strip()

    decoder = json.JSONDecoder()
    candidates = []

    for i, c in enumerate(text):
        if c == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    candidates.append(obj)
            except Exception:
                pass

    if not candidates:
        raise ValueError("NO_JSON")

    return candidates[-1]


SECRET_PATTERNS = [
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b", re.I),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
]

CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"
)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def has_secret_like_literal(text: str) -> bool:
    return any(p.search(text) for p in SECRET_PATTERNS)


def validate_user_text(user: str):
    user = str(user).strip()

    if len(user) < 120:
        raise ValueError("USER_TOO_SHORT")

    if len(user) > 2200:
        raise ValueError("USER_TOO_LONG")

    banned = [
        "konkrét magyar aion rendszerfeladat",
        "feladat leírás",
        "task description",
        "kötelező scenario",
        "a scenario követelménye",
        "a válasz kizárólag json",
    ]

    low = user.lower()

    for item in banned:
        if item in low:
            raise ValueError("PROMPT_LEAKAGE")

    if CJK_RE.search(user):
        raise ValueError("UNEXPECTED_SCRIPT")

    if has_secret_like_literal(user):
        raise ValueError("SECRET_LIKE_LITERAL")

    # Ne tanítsunk fiktív AION verziókat/protokollokat tényként.
    if re.search(
        r"\bAION\s+(?:v\d|verzió|version|protokoll|protocol)\b",
        user,
        re.I,
    ):
        raise ValueError("FABRICATED_AION_PRODUCT_DETAIL")

    # Nagyon egyszerű ismétlődési védelem.
    chunks = [
        x.strip().lower()
        for x in re.split(r"[.!?]\s+", user)
        if len(x.strip()) >= 25
    ]
    if chunks:
        counts = Counter(chunks)
        if max(counts.values()) >= 3:
            raise ValueError("EXCESSIVE_SENTENCE_REPETITION")

    return user


def sample_policy_facts(rng: random.Random, scenario: str):
    """
    A scenario NEM határozza meg közvetlenül a targetet.

    Először egy kockázati/végrehajtási profilt választunk,
    abból konkrét policy facteket hozunk létre,
    és a targetet kizárólag derive_target() számolja.
    """

    profile = rng.choices(
        [
            "safe_self",
            "routine_delegate",
            "tool_read",
            "tool_validation",
            "evidence_conflict",
            "approval_escalation",
            "critical_operation",
            "secret_block",
        ],
        weights=[14, 18, 20, 13, 11, 10, 9, 5],
        k=1,
    )[0]

    # Secret-block profil főleg secret_handlingre kerüljön,
    # de maga a scenario továbbra sem egyenlő egy fix targettel.
    if profile == "secret_block" and scenario != "secret_handling":
        if rng.random() < 0.75:
            profile = "approval_escalation"

    facts = {
        "profile": profile,
        "impact": "LOW",
        "evidence_quality": "VERIFIED",
        "reversibility": "EASY",
        "production_scope": "LOCAL",
        "external_side_effect": False,
        "destructive": False,
        "requires_live_data": False,
        "requires_tool": False,
        "delegation_requested": False,
        "task_complexity": "BOUNDED",
        "secret_exposure_requested": False,
        "explicit_human_approval_gate": False,
    }

    if profile == "safe_self":
        facts.update(
            impact=rng.choice(["LOW", "MEDIUM"]),
            evidence_quality="VERIFIED",
            reversibility="EASY",
            production_scope=rng.choice(["LOCAL", "TEAM"]),
            task_complexity="BOUNDED",
        )

    elif profile == "routine_delegate":
        facts.update(
            impact=rng.choice(["LOW", "MEDIUM"]),
            evidence_quality=rng.choice(["VERIFIED", "PARTIAL"]),
            reversibility="EASY",
            production_scope=rng.choice(["LOCAL", "TEAM"]),
            delegation_requested=True,
            task_complexity=rng.choice(["ROUTINE", "BOUNDED"]),
        )

    elif profile == "tool_read":
        facts.update(
            impact=rng.choice(["LOW", "MEDIUM"]),
            evidence_quality=rng.choice(["PARTIAL", "UNVERIFIED"]),
            reversibility="EASY",
            production_scope=rng.choice(["LOCAL", "TEAM"]),
            requires_live_data=True,
            requires_tool=True,
            task_complexity="BOUNDED",
        )

    elif profile == "tool_validation":
        facts.update(
            impact=rng.choice(["MEDIUM", "HIGH"]),
            evidence_quality="PARTIAL",
            reversibility=rng.choice(["EASY", "REVERSIBLE_WITH_BACKUP"]),
            production_scope=rng.choice(["TEAM", "PRODUCTION"]),
            requires_live_data=True,
            requires_tool=True,
            task_complexity="COMPLEX" if rng.random() < 0.35 else "BOUNDED",
        )

    elif profile == "evidence_conflict":
        facts.update(
            impact=rng.choice(["MEDIUM", "HIGH"]),
            evidence_quality=rng.choice(["CONFLICTING", "UNVERIFIED"]),
            reversibility=rng.choice(["EASY", "REVERSIBLE_WITH_BACKUP"]),
            production_scope=rng.choice(["TEAM", "PRODUCTION"]),
            requires_live_data=True,
            requires_tool=True,
            task_complexity="COMPLEX",
        )

    elif profile == "approval_escalation":
        facts.update(
            impact=rng.choice(["HIGH", "CRITICAL"]),
            evidence_quality=rng.choice(["VERIFIED", "PARTIAL"]),
            reversibility=rng.choice(
                ["REVERSIBLE_WITH_BACKUP", "DIFFICULT"]
            ),
            production_scope="PRODUCTION",
            external_side_effect=True,
            explicit_human_approval_gate=True,
            task_complexity="COMPLEX",
        )

    elif profile == "critical_operation":
        facts.update(
            impact="CRITICAL",
            evidence_quality=rng.choice(
                ["PARTIAL", "UNVERIFIED", "CONFLICTING"]
            ),
            reversibility=rng.choice(["DIFFICULT", "IRREVERSIBLE"]),
            production_scope="PRODUCTION",
            external_side_effect=True,
            destructive=True,
            requires_live_data=True,
            requires_tool=True,
            task_complexity="COMPLEX",
        )

    elif profile == "secret_block":
        facts.update(
            impact="CRITICAL",
            evidence_quality="VERIFIED",
            reversibility="IRREVERSIBLE",
            production_scope=rng.choice(["TEAM", "PRODUCTION"]),
            external_side_effect=True,
            secret_exposure_requested=True,
            task_complexity="COMPLEX",
        )

    return facts


def derive_target(scenario: str, facts: dict):
    capability = TOOL_CAPABILITY_BY_SCENARIO[scenario]

    # 1. Titok felfedése / veszélyes credential kezelés:
    # approval sem teszi biztonságossá.
    if facts["secret_exposure_requested"]:
        return {
            "decision": "ESCALATE",
            "risk_level": "CRITICAL",
            "executor_class": "NONE",
            "tool_capability": "SECRET_GUARD",
            "approval_required": False,
            "validation_required": True,
            "execution_mode": "NONE",
            "reason":
                "A kérés hitelesítő vagy érzékeny adat veszélyes "
                "felfedését eredményezné; ezt nem szabad automatikusan "
                "végrehajtani.",
        }

    # 2. Kritikus/destruktív/irreverzibilis production művelet.
    critical = (
        facts["impact"] == "CRITICAL"
        or facts["destructive"]
        or facts["reversibility"] == "IRREVERSIBLE"
    )

    if critical:
        return {
            "decision": "ESCALATE",
            "risk_level": "CRITICAL",
            "executor_class": "MAIN_BRAIN",
            "tool_capability": (
                capability if facts["requires_tool"] else "NONE"
            ),
            "approval_required": True,
            "validation_required": True,
            "execution_mode": "SINGLE",
            "reason":
                "A művelet kritikus vagy destruktív hatású, ezért "
                "emberi jóváhagyás és független validáció szükséges.",
        }

    # 3. Explicit approval gate.
    if facts["explicit_human_approval_gate"]:
        return {
            "decision": "ESCALATE",
            "risk_level": "HIGH",
            "executor_class": "MAIN_BRAIN",
            "tool_capability": (
                capability if facts["requires_tool"] else "NONE"
            ),
            "approval_required": True,
            "validation_required": True,
            "execution_mode": "SINGLE",
            "reason":
                "A production hatású művelethez explicit emberi "
                "jóváhagyási kapu tartozik.",
        }

    # 4. Ellentmondó/nem igazolt állapot magasabb hatás mellett.
    if (
        facts["evidence_quality"] in {"CONFLICTING", "UNVERIFIED"}
        and facts["impact"] in {"MEDIUM", "HIGH"}
        and facts["production_scope"] in {"TEAM", "PRODUCTION"}
    ):
        return {
            "decision": "ESCALATE",
            "risk_level": "HIGH",
            "executor_class": "MAIN_BRAIN",
            "tool_capability": capability,
            "approval_required": False,
            "validation_required": True,
            "execution_mode": "SINGLE",
            "reason":
                "A rendelkezésre álló bizonyíték nem elég megbízható "
                "a hatásos döntéshez; előbb feloldott és validált "
                "állapot szükséges.",
        }

    # 5. Eszközös/live ellenőrzés.
    if facts["requires_tool"] or facts["requires_live_data"]:
        risk = "HIGH" if facts["impact"] == "HIGH" else "MEDIUM"
        executor = (
            "MAIN_BRAIN"
            if facts["impact"] == "HIGH"
            or facts["task_complexity"] == "COMPLEX"
            else "SMALL_WORKER"
        )

        return {
            "decision": "USE_TOOL",
            "risk_level": risk,
            "executor_class": executor,
            "tool_capability": capability,
            "approval_required": False,
            "validation_required": True,
            "execution_mode": "SINGLE",
            "reason":
                "A döntéshez friss vagy ellenőrizhető rendszeradat "
                "szükséges, ezért eszközös lekérdezés és validáció kell.",
        }

    # 6. Delegálható, korlátozott feladat.
    if facts["delegation_requested"]:
        risk = "MEDIUM" if facts["impact"] == "MEDIUM" else "LOW"
        executor = (
            "MAIN_BRAIN"
            if facts["task_complexity"] == "COMPLEX"
            else "SMALL_WORKER"
        )

        return {
            "decision": "DELEGATE",
            "risk_level": risk,
            "executor_class": executor,
            "tool_capability": "NONE",
            "approval_required": False,
            "validation_required": facts["impact"] != "LOW",
            "execution_mode": "SINGLE",
            "reason":
                "A feladat korlátozott és delegálható; a végrehajtó "
                "a kockázat és komplexitás alapján választható.",
        }

    # 7. Alacsony kockázatú belső Manager-döntés.
    risk = "MEDIUM" if facts["impact"] == "MEDIUM" else "LOW"

    return {
        "decision": "SELF",
        "risk_level": risk,
        "executor_class": "SELF",
        "tool_capability": "NONE",
        "approval_required": False,
        "validation_required": False,
        "execution_mode": "SINGLE",
        "reason":
            "A feladat alacsony kockázatú, visszafordítható és "
            "elegendő bizonyítékkal rendelkezik, ezért közvetlenül "
            "kezelhető.",
    }


def validate_target(target: dict):
    required = {
        "decision",
        "risk_level",
        "executor_class",
        "tool_capability",
        "approval_required",
        "validation_required",
        "execution_mode",
        "reason",
    }

    if set(target.keys()) != required:
        raise ValueError("TARGET_SCHEMA_MISMATCH")

    if target["decision"] not in DECISIONS:
        raise ValueError("BAD_DECISION")

    if target["risk_level"] not in RISKS:
        raise ValueError("BAD_RISK")

    if target["executor_class"] not in EXECUTORS:
        raise ValueError("BAD_EXECUTOR")

    if target["execution_mode"] not in EXECUTION_MODES:
        raise ValueError("BAD_EXECUTION_MODE")

    if not isinstance(target["approval_required"], bool):
        raise ValueError("BAD_APPROVAL_BOOL")

    if not isinstance(target["validation_required"], bool):
        raise ValueError("BAD_VALIDATION_BOOL")

    if not isinstance(target["tool_capability"], str):
        raise ValueError("BAD_TOOL_CAPABILITY")

    if not isinstance(target["reason"], str) or len(target["reason"]) < 20:
        raise ValueError("BAD_REASON")


def facts_for_prompt(facts: dict):
    # A belső profile ne kerüljön a Qwen promptjába.
    return {
        k: v
        for k, v in facts.items()
        if k != "profile"
    }


def build_generation_prompt(scenario: str, facts: dict):
    return f"""
Te egy magyar technikai dataset példát készítesz egy Manager döntési modellhez.

Írj EGY természetes magyar felhasználói rendszerfeladatot.

Scenario-család:
{scenario}

Scenario jelentése:
{SCENARIO_RULES[scenario]}

A feladatnak természetesen és egyértelműen tükröznie kell ezeket a tényeket:
{json.dumps(facts_for_prompt(facts), ensure_ascii=False, indent=2)}

Fontos szabályok:
- A helyzet legyen konkrét, technikai és hihető.
- A policy-tények következzenek magából a feladat szövegéből.
- Ne írd le a policy mezőneveket vagy a belső targetet.
- Ne adj meg valódi vagy valódinak látszó API kulcsot, jelszót vagy tokent.
- Ha credential kell a példához, csak ilyen szemantikus placeholdert használj: <API_KEY> vagy <SECRET>.
- Ne találj ki AION verziót, protokollt, termékfunkciót vagy modulnevet tényként.
- Az AION legfeljebb általános Manager/worker rendszerként szerepelhet.
- Ne állíts sikeresnek vagy létezőnek nem ellenőrzött rendszerállapotot.
- Kerüld a személyneveket, telefonszámokat, valódi címeket és egyéb szükségtelen személyes adatokat.
- Legyen természetes magyar, ne ismételj mondatokat.
- 120 és 1600 karakter közötti legyen.
- Ne tartalmazzon választ vagy ManagerDecision JSON-t.

Kizárólag ezt a JSON objektumot add vissza:
{{
  "user": "a magyar feladat"
}}
""".strip()


def make_record(user, scenario, facts, target, args, call_no):
    assistant = json.dumps(
        target,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    user_hash = hashlib.sha256(
        normalize_text(user).encode("utf-8")
    ).hexdigest()

    return {
        "messages": [
            {
                "role": "user",
                "content": user,
            },
            {
                "role": "assistant",
                "content": assistant,
            },
        ],
        "target": target,
        "metadata": {
            "generator": "Qwen3.5-9B",
            "version": VERSION,
            "node": args.node,
            "run_id": args.run_id,
            "scenario": scenario,
            "policy_facts": facts,
            "candidate_call": call_no,
            "user_sha256": user_hash,
            "created": utc_now(),
        },
    }


def load_existing(live_path: Path):
    seen = set()
    accepted = 0
    last_call = 0
    decisions = Counter()
    risks = Counter()
    scenarios = Counter()

    if not live_path.exists():
        return seen, accepted, last_call, decisions, risks, scenarios

    with live_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            obj = json.loads(line)

            user = obj["messages"][0]["content"]
            seen.add(
                hashlib.sha256(
                    normalize_text(user).encode("utf-8")
                ).hexdigest()
            )

            accepted += 1
            last_call = max(
                last_call,
                int(obj.get("metadata", {}).get("candidate_call", 0)),
            )
            decisions[obj["target"]["decision"]] += 1
            risks[obj["target"]["risk_level"]] += 1
            scenarios[obj["metadata"]["scenario"]] += 1

    return seen, accepted, last_call, decisions, risks, scenarios


def write_state(
    state_path,
    args,
    accepted,
    stats,
    decisions,
    risks,
    scenarios,
    live_path,
    status,
    started_at,
):
    obj = {
        "status": status,
        "version": VERSION,
        "run_id": args.run_id,
        "node": args.node,
        "target_count": args.count,
        "accepted": accepted,
        "stats": dict(stats),
        "decisions": dict(decisions),
        "risks": dict(risks),
        "scenarios": dict(scenarios),
        "started_at": started_at,
        "updated_at": utc_now(),
        "live_file": str(live_path),
        "live_size_bytes": (
            live_path.stat().st_size if live_path.exists() else 0
        ),
    }

    atomic_json_write(state_path, obj)


def checkpoint(
    run_dir,
    live_file_handle,
    live_path,
    accepted,
    args,
    stats,
    decisions,
    risks,
    scenarios,
    started_at,
    reason,
):
    if accepted == 0 or not live_path.exists():
        return None

    live_file_handle.flush()
    os.fsync(live_file_handle.fileno())

    cp_dir = run_dir / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cp_path = cp_dir / f"checkpoint_{stamp}_n{accepted:07d}.jsonl"
    tmp_path = cp_path.with_suffix(".jsonl.tmp")

    shutil.copyfile(live_path, tmp_path)

    # A snapshotot is lemezre kényszerítjük.
    with tmp_path.open("rb") as f:
        os.fsync(f.fileno())

    os.replace(tmp_path, cp_path)

    digest = sha256_file(cp_path)

    meta = {
        "status": "CHECKPOINT",
        "reason": reason,
        "version": VERSION,
        "run_id": args.run_id,
        "node": args.node,
        "records": accepted,
        "sha256": digest,
        "source_live_file": str(live_path),
        "checkpoint_file": str(cp_path),
        "stats": dict(stats),
        "decisions": dict(decisions),
        "risks": dict(risks),
        "scenarios": dict(scenarios),
        "started_at": started_at,
        "created": utc_now(),
    }

    atomic_json_write(
        cp_path.with_suffix(cp_path.suffix + ".meta.json"),
        meta,
    )

    return cp_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--node", default="cloud_rtx6000")
    parser.add_argument("--seed", type=int, default=6000)
    parser.add_argument("--run-id", required=True)

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
        raise SystemExit("--count must be >= 1")

    if args.checkpoint_minutes <= 0:
        raise SystemExit("--checkpoint-minutes must be > 0")

    run_dir = DEFAULT_RUN_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    live_path = run_dir / "dataset.live.jsonl"
    reject_path = run_dir / "rejects.private.jsonl"
    state_path = run_dir / "state.json"
    config_path = run_dir / "config.json"

    config = {
        "version": VERSION,
        "model": str(MODEL),
        "count": args.count,
        "node": args.node,
        "seed": args.seed,
        "run_id": args.run_id,
        "checkpoint_minutes": args.checkpoint_minutes,
        "max_runtime_hours": args.max_runtime_hours,
    }

    if not config_path.exists():
        atomic_json_write(config_path, config)
    else:
        old = json.loads(config_path.read_text(encoding="utf-8"))

        # Resume során a count és runtime növelhető,
        # de a generációs identitás ne változzon.
        for key in ["version", "model", "node", "seed", "run_id"]:
            if old.get(key) != config.get(key):
                raise SystemExit(
                    f"RESUME_CONFIG_MISMATCH: {key}"
                )

    (
        seen,
        accepted,
        last_call,
        decisions,
        risks,
        scenarios,
    ) = load_existing(live_path)

    stats = Counter()
    stats["accepted"] = accepted
    stats["calls"] = last_call

    started_at = utc_now()
    start_monotonic = time.monotonic()
    last_checkpoint = start_monotonic

    print(
        json.dumps(
            {
                "status": "STARTING",
                "version": VERSION,
                "run_id": args.run_id,
                "resume_records": accepted,
                "resume_last_call": last_call,
                "target_count": args.count,
                "run_dir": str(run_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.eval()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with live_path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as live, reject_path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as rejects:

        while accepted < args.count and not STOP_REQUESTED:
            elapsed_hours = (
                time.monotonic() - start_monotonic
            ) / 3600.0

            if (
                args.max_runtime_hours > 0
                and elapsed_hours >= args.max_runtime_hours
            ):
                print("MAX_RUNTIME_REACHED", flush=True)
                break

            call_no = stats["calls"] + 1
            stats["calls"] = call_no

            # Reprodukálható, resume-barát per-call RNG.
            call_seed = args.seed + call_no * 1000003
            rng = random.Random(call_seed)
            torch.manual_seed(call_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(call_seed)

            scenario = rng.choice(SCENARIOS)
            facts = sample_policy_facts(rng, scenario)
            target = derive_target(scenario, facts)
            validate_target(target)

            prompt = build_generation_prompt(
                scenario,
                facts,
            )

            try:
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

                generated_tokens = out[0][inp.input_ids.shape[1]:]

                raw = tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                ).strip()

                item = extract_json(raw)
                user = validate_user_text(item.get("user", ""))

                user_hash = hashlib.sha256(
                    normalize_text(user).encode("utf-8")
                ).hexdigest()

                if user_hash in seen:
                    raise ValueError("EXACT_NORMALIZED_DUPLICATE")

                record = make_record(
                    user=user,
                    scenario=scenario,
                    facts=facts,
                    target=target,
                    args=args,
                    call_no=call_no,
                )

                serialized = json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                # CRASH-SAFE:
                # minden accepted rekord azonnal durable lemezre.
                live.write(serialized + "\n")
                live.flush()
                os.fsync(live.fileno())

                seen.add(user_hash)
                accepted += 1

                stats["accepted"] = accepted
                decisions[target["decision"]] += 1
                risks[target["risk_level"]] += 1
                scenarios[scenario] += 1

                write_state(
                    state_path=state_path,
                    args=args,
                    accepted=accepted,
                    stats=stats,
                    decisions=decisions,
                    risks=risks,
                    scenarios=scenarios,
                    live_path=live_path,
                    status="RUNNING",
                    started_at=started_at,
                )

            except Exception as exc:
                stats["rejected"] += 1

                # A reject log privát artifact.
                # Nem kerülhet automatikusan public GitHubra.
                reject_record = {
                    "created": utc_now(),
                    "call": call_no,
                    "scenario": scenario,
                    "policy_facts": facts,
                    "reason": str(exc),
                }

                rejects.write(
                    json.dumps(
                        reject_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rejects.flush()
                os.fsync(rejects.fileno())

            now = time.monotonic()

            if (
                now - last_checkpoint
                >= args.checkpoint_minutes * 60.0
            ):
                cp = checkpoint(
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
                    reason="PERIODIC",
                )

                print(
                    f"CHECKPOINT records={accepted} path={cp}",
                    flush=True,
                )

                last_checkpoint = now

            print(
                f"records={accepted}/{args.count} "
                f"calls={stats['calls']} "
                f"rejected={stats['rejected']} "
                f"decisions={dict(decisions)}",
                flush=True,
            )

        final_reason = (
            "SIGNAL"
            if STOP_REQUESTED
            else (
                "TARGET_REACHED"
                if accepted >= args.count
                else "MAX_RUNTIME"
            )
        )

        final_cp = checkpoint(
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

        write_state(
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
                "status": final_reason,
                "run_id": args.run_id,
                "records": accepted,
                "stats": dict(stats),
                "decisions": dict(decisions),
                "risks": dict(risks),
                "scenarios": dict(scenarios),
                "live_file": str(live_path),
                "live_sha256": (
                    sha256_file(live_path)
                    if live_path.exists()
                    else None
                ),
                "final_checkpoint": (
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
