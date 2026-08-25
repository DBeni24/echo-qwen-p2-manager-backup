import json
import random
import re
from collections import Counter
from pathlib import Path

import generate_manager_policy_dataset_v05_safe_v01 as base


base.VERSION = "V05_SAFE_POLICY_V03"

# Preserve ORIGINAL V01 functions before monkey-patching base.
_BASE_LOAD_EXISTING = base.load_existing
_BASE_VALIDATE_USER_TEXT = base.validate_user_text

CURRENT_CONTEXT = None


SCENARIO_PROFILES = {
    "workflow_planning": [
        ("safe_self", 20),
        ("tool_read", 55),
        ("evidence_conflict", 25),
    ],
    "rag_conflict": [
        ("tool_validation", 55),
        ("evidence_conflict", 45),
    ],
    "false_state_detection": [
        ("tool_validation", 55),
        ("evidence_conflict", 45),
    ],
    "tool_selection": [
        ("safe_self", 20),
        ("tool_read", 60),
        ("evidence_conflict", 20),
    ],
    "approval_required": [
        ("approval_escalation", 75),
        ("critical_operation", 25),
    ],
    "secret_handling": [
        ("safe_self", 25),
        ("tool_validation", 25),
        ("secret_block", 50),
    ],
    "checkpoint_review": [
        ("tool_validation", 55),
        ("evidence_conflict", 45),
    ],
    "provider_routing": [
        ("safe_self", 20),
        ("tool_read", 55),
        ("evidence_conflict", 25),
    ],
    "worker_assignment": [
        ("routine_delegate", 55),
        ("tool_read", 30),
        ("evidence_conflict", 15),
    ],
}


def choose_profile(rng, scenario):
    pairs = SCENARIO_PROFILES[scenario]
    names = [x[0] for x in pairs]
    weights = [x[1] for x in pairs]
    return rng.choices(names, weights=weights, k=1)[0]


def sample_policy_facts(rng: random.Random, scenario: str):
    global CURRENT_CONTEXT

    profile = choose_profile(rng, scenario)

    facts = {
        "profile": profile,
        "operation_type": "PLAN_ONLY",
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
            operation_type="PLAN_ONLY",
            impact=rng.choice(["LOW", "MEDIUM"]),
            evidence_quality="VERIFIED",
            reversibility="EASY",
            production_scope=rng.choice(["LOCAL", "TEAM"]),
            task_complexity="BOUNDED",
        )

    elif profile == "routine_delegate":
        facts.update(
            operation_type="DELEGATE",
            impact=rng.choice(["LOW", "MEDIUM"]),
            evidence_quality=rng.choice(["VERIFIED", "PARTIAL"]),
            reversibility="EASY",
            production_scope=rng.choice(["LOCAL", "TEAM"]),
            delegation_requested=True,
            task_complexity=rng.choice(["ROUTINE", "BOUNDED"]),
        )

    elif profile == "tool_read":
        facts.update(
            operation_type="READ_ONLY",
            impact=rng.choice(["LOW", "MEDIUM"]),
            evidence_quality=rng.choice(["PARTIAL", "UNVERIFIED"]),
            reversibility="EASY",
            production_scope=rng.choice(["LOCAL", "TEAM"]),
            requires_live_data=True,
            requires_tool=True,
            task_complexity="BOUNDED",
        )

    elif profile == "tool_validation":
        scope = (
            "TEAM"
            if scenario == "secret_handling"
            else rng.choice(["TEAM", "PRODUCTION"])
        )

        facts.update(
            operation_type="VALIDATE",
            impact=rng.choice(["MEDIUM", "HIGH"]),
            evidence_quality="PARTIAL",
            reversibility="EASY",
            production_scope=scope,
            requires_live_data=True,
            requires_tool=True,
            task_complexity=rng.choice(["BOUNDED", "COMPLEX"]),
        )

    elif profile == "evidence_conflict":
        facts.update(
            operation_type="VALIDATE",
            impact=rng.choice(["MEDIUM", "HIGH"]),
            evidence_quality=rng.choice(["CONFLICTING", "UNVERIFIED"]),
            reversibility="EASY",
            production_scope=rng.choice(["TEAM", "PRODUCTION"]),
            requires_live_data=True,
            requires_tool=True,
            task_complexity="COMPLEX",
        )

    elif profile == "approval_escalation":
        facts.update(
            operation_type="EXECUTE_REVERSIBLE",
            impact="HIGH",
            evidence_quality=rng.choice(["VERIFIED", "PARTIAL"]),
            reversibility="REVERSIBLE_WITH_BACKUP",
            production_scope="PRODUCTION",
            external_side_effect=True,
            destructive=False,
            requires_live_data=True,
            requires_tool=True,
            task_complexity="COMPLEX",
            explicit_human_approval_gate=True,
        )

    elif profile == "critical_operation":
        facts.update(
            operation_type="EXECUTE_DESTRUCTIVE",
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
            explicit_human_approval_gate=True,
        )

    elif profile == "secret_block":
        facts.update(
            operation_type="SECRET_DISCLOSURE",
            impact="CRITICAL",
            evidence_quality="VERIFIED",
            reversibility="IRREVERSIBLE",
            production_scope=rng.choice(["TEAM", "PRODUCTION"]),
            external_side_effect=True,
            destructive=False,
            requires_live_data=False,
            requires_tool=False,
            task_complexity="COMPLEX",
            secret_exposure_requested=True,
            explicit_human_approval_gate=False,
        )

    CURRENT_CONTEXT = {
        "scenario": scenario,
        "facts": facts,
    }

    return facts


def derive_target(scenario: str, facts: dict):
    capability = base.TOOL_CAPABILITY_BY_SCENARIO[scenario]
    operation = facts["operation_type"]

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
                "A kérés hitelesítő vagy érzékeny adat felfedését "
                "eredményezné; ezt nem szabad automatikusan végrehajtani.",
        }

    if operation == "EXECUTE_DESTRUCTIVE":
        return {
            "decision": "ESCALATE",
            "risk_level": "CRITICAL",
            "executor_class": "MAIN_BRAIN",
            "tool_capability": capability,
            "approval_required": True,
            "validation_required": True,
            "execution_mode": "SINGLE",
            "reason":
                "A kért production művelet destruktív vagy nehezen "
                "visszafordítható; jóváhagyás és validáció szükséges.",
        }

    if (
        operation == "EXECUTE_REVERSIBLE"
        or facts["explicit_human_approval_gate"]
    ):
        return {
            "decision": "ESCALATE",
            "risk_level": "HIGH",
            "executor_class": "MAIN_BRAIN",
            "tool_capability": capability,
            "approval_required": True,
            "validation_required": True,
            "execution_mode": "SINGLE",
            "reason":
                "A production állapotot módosító művelethez explicit "
                "emberi jóváhagyás és végrehajtás előtti validáció kell.",
        }

    if (
        facts["evidence_quality"] in {"CONFLICTING", "UNVERIFIED"}
        and (
            facts["impact"] == "HIGH"
            or facts["production_scope"] == "PRODUCTION"
        )
    ):
        return {
            "decision": "ESCALATE",
            "risk_level": "HIGH",
            "executor_class": "MAIN_BRAIN",
            "tool_capability": (
                capability if facts["requires_tool"] else "NONE"
            ),
            "approval_required": False,
            "validation_required": True,
            "execution_mode": "SINGLE",
            "reason":
                "A nagy hatású döntéshez rendelkezésre álló állapot "
                "ellentmondó vagy nem igazolt; előbb validálni kell.",
        }

    if facts["delegation_requested"]:
        risk = {
            "LOW": "LOW",
            "MEDIUM": "MEDIUM",
            "HIGH": "HIGH",
            "CRITICAL": "HIGH",
        }[facts["impact"]]

        executor = (
            "MAIN_BRAIN"
            if facts["impact"] == "HIGH"
            or facts["task_complexity"] == "COMPLEX"
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
                "A feladat delegálható, a végrehajtó pedig a hatás és "
                "komplexitás alapján választható ki.",
        }

    if facts["requires_tool"] or facts["requires_live_data"]:
        risk = {
            "LOW": "LOW",
            "MEDIUM": "MEDIUM",
            "HIGH": "HIGH",
            "CRITICAL": "HIGH",
        }[facts["impact"]]

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
            "A feladat tervezési vagy döntési jellegű, nem igényel "
            "élő rendszeradatot vagy állapotmódosítást.",
    }


def operation_instruction(operation):
    return {
        "PLAN_ONLY":
            "Csak tervet, javaslatot vagy döntést kérjen. "
            "Semmit ne indítson el és ne módosítson.",
        "READ_ONLY":
            "Csak olvasási, lekérdezési vagy elemzési műveletet kérjen. "
            "Semmilyen állapotmódosítást vagy delegálást ne kérjen.",
        "VALIDATE":
            "Csak ellenőrzést, validációt vagy összehasonlítást kérjen. "
            "Ne kérjen rollbacket, deployt vagy más végrehajtást.",
        "DELEGATE":
            "Kifejezetten egy korlátozott feladat delegálását kérje, "
            "de ne kérjen production állapotmódosítást.",
        "EXECUTE_REVERSIBLE":
            "Konkrét production állapotmódosítást kérjen, amely "
            "visszaállítható, és mondja ki, hogy emberi jóváhagyás kell.",
        "EXECUTE_DESTRUCTIVE":
            "Konkrét magas kockázatú production műveletet kérjen, "
            "amely destruktív vagy nehezen visszafordítható, és "
            "mondja ki az emberi jóváhagyási követelményt.",
        "SECRET_DISCLOSURE":
            "Egy placeholderrel jelölt credential teljes értékének "
            "felfedését kérje, például <API_KEY> vagy <SECRET>. "
            "Valódi vagy valódinak látszó credentialt ne generáljon.",
    }[operation]


def build_generation_prompt(scenario: str, facts: dict):
    return f"""
Te egy magyar technikai dataset példát készítesz egy Manager döntési modellhez.

Írj EGY természetes, konkrét magyar felhasználói rendszerfeladatot.

Scenario-család:
{scenario}

Scenario jelentése:
{base.SCENARIO_RULES[scenario]}

Kötelező helyzettények:
- művelet jellege: {facts["operation_type"]}
- hatás: {facts["impact"]}
- bizonyíték minősége: {facts["evidence_quality"]}
- visszafordíthatóság: {facts["reversibility"]}
- környezet: {facts["production_scope"]}
- külső állapotváltozás: {facts["external_side_effect"]}
- destruktív: {facts["destructive"]}
- friss rendszeradat kell: {facts["requires_live_data"]}
- tool kell: {facts["requires_tool"]}
- delegálást kér: {facts["delegation_requested"]}
- komplexitás: {facts["task_complexity"]}
- titok felfedését kér: {facts["secret_exposure_requested"]}
- emberi approval gate: {facts["explicit_human_approval_gate"]}

Művelettípus-specifikus szabály:
{operation_instruction(facts["operation_type"])}

További kötelező szabályok:
- A szövegből egyértelműen következzen minden fenti helyzettény.
- Ne írd le a policy mezőneveket vagy a targetet.
- Ne adj meg valódi vagy valódinak látszó API kulcsot, tokent vagy jelszót.
- Credentialhez csak <API_KEY> vagy <SECRET> placeholder használható.
- Ne találj ki AION verziót, protokollt vagy nem bizonyított modult.
- Ne állíts sikeresnek nem ellenőrzött rendszerállapotot.
- Ne használj személynevet, telefonszámot vagy szükségtelen személyes adatot.
- Természetes magyar nyelvet használj.
- Ne ismételj mondatokat.
- 120 és 1600 karakter közötti legyen.
- Ne tartalmazzon választ vagy ManagerDecision JSON-t.

Kizárólag ezt add vissza:
{{
  "user": "a magyar feladat"
}}
""".strip()


def contains_any(text, parts):
    low = text.lower()
    return any(x in low for x in parts)


def validate_user_text(user: str):
    user = _BASE_VALIDATE_USER_TEXT(user)

    if CURRENT_CONTEXT is None:
        raise ValueError("NO_POLICY_CONTEXT")

    scenario = CURRENT_CONTEXT["scenario"]
    facts = CURRENT_CONTEXT["facts"]
    op = facts["operation_type"]
    low = user.lower()

    policy_leak = [
        "operation_type",
        "evidence_quality",
        "requires_tool",
        "requires_live_data",
        "delegation_requested",
        "external_side_effect",
        "production_scope",
        "secret_exposure_requested",
    ]

    if contains_any(low, policy_leak):
        raise ValueError("POLICY_FIELD_LEAKAGE")

    delegation_terms = [
        "delegáld",
        "delegeld",
        "delegálását",
        "bízd rá",
        "oszd ki",
        "rendeld hozzá",
    ]

    execution_terms = [
        "indítsd el",
        "indíts el",
        "hajtsd végre",
        "futtasd le",
        "telepítsd",
        "deployold",
        "állítsd le",
        "töröld",
        "módosítsd",
        "alkalmazd a változtatást",
        "kapcsold ki",
        "kapcsold be",
        "rotáld",
    ]

    validation_terms = [
        "ellenőriz",
        "validál",
        "vizsgál",
        "audit",
        "hasonlíts",
        "vesd össze",
        "elemezd",
    ]

    live_terms = [
        "aktuális",
        "jelenlegi",
        "legutóbbi",
        "live",
        "valós idej",
        "státusz",
        "log",
        "metrika",
        "adatbázis",
        "api",
        "checkpoint",
        "monitoring",
    ]

    approval_terms = [
        "jóváhagy",
        "engedély",
        "approval",
    ]

    production_terms = [
        "production",
        "éles rendszer",
        "éles környezet",
        "prod-",
        "prod ",
    ]

    secret_terms = [
        "api kulcs",
        "api-key",
        "token",
        "jelszó",
        "secret",
        "<api_key>",
        "<secret>",
    ]

    disclosure_terms = [
        "add meg",
        "írd ki",
        "mutasd meg",
        "fedd fel",
        "teljes érték",
    ]

    if facts["delegation_requested"]:
        if not contains_any(low, delegation_terms + ["worker", "munkatárs"]):
            raise ValueError("DELEGATION_NOT_EXPRESSED")
    else:
        if contains_any(low, delegation_terms):
            raise ValueError("UNEXPECTED_DELEGATION")

    if op in {"PLAN_ONLY", "READ_ONLY", "VALIDATE"}:
        if contains_any(low, execution_terms):
            raise ValueError("UNEXPECTED_EXECUTION_REQUEST")

        if re.search(
            r"(indít|futtat|hajt).{0,50}\brollback\b",
            low,
        ):
            raise ValueError("UNEXPECTED_ROLLBACK_EXECUTION")

    if op == "PLAN_ONLY":
        if not contains_any(
            low,
            ["terv", "tervez", "javas", "lépés", "workflow", "válassz", "dönts"],
        ):
            raise ValueError("PLAN_NOT_EXPRESSED")

    if op == "READ_ONLY":
        if not contains_any(
            low,
            validation_terms + ["lekér", "nézd meg", "gyűjts", "listáz"],
        ):
            raise ValueError("READ_ONLY_NOT_EXPRESSED")

    if op == "VALIDATE":
        if not contains_any(low, validation_terms):
            raise ValueError("VALIDATION_NOT_EXPRESSED")

    if op in {"EXECUTE_REVERSIBLE", "EXECUTE_DESTRUCTIVE"}:
        if not contains_any(low, execution_terms):
            raise ValueError("EXECUTION_NOT_EXPRESSED")

    if facts["explicit_human_approval_gate"]:
        if not contains_any(low, approval_terms):
            raise ValueError("APPROVAL_NOT_EXPRESSED")

    if facts["production_scope"] == "PRODUCTION":
        if not contains_any(low, production_terms):
            raise ValueError("PRODUCTION_SCOPE_NOT_EXPRESSED")
    else:
        if contains_any(low, production_terms):
            raise ValueError("UNEXPECTED_PRODUCTION_SCOPE")

    if facts["requires_live_data"]:
        if not contains_any(low, live_terms):
            raise ValueError("LIVE_DATA_NOT_EXPRESSED")

    if facts["secret_exposure_requested"]:
        if not (
            contains_any(low, secret_terms)
            and contains_any(low, disclosure_terms)
        ):
            raise ValueError("SECRET_DISCLOSURE_NOT_EXPRESSED")

        if "<api_key>" not in low and "<secret>" not in low:
            raise ValueError("SECRET_PLACEHOLDER_MISSING")

    if not facts["secret_exposure_requested"]:
        if (
            contains_any(low, secret_terms)
            and contains_any(low, disclosure_terms)
        ):
            raise ValueError("UNEXPECTED_SECRET_DISCLOSURE")

    return user


def load_existing(live_path: Path):
    (
        seen,
        accepted,
        last_call,
        decisions,
        risks,
        scenarios,
    ) = _BASE_LOAD_EXISTING(live_path)

    reject_path = live_path.parent / "rejects.private.jsonl"
    state_path = live_path.parent / "state.json"

    if reject_path.exists():
        for line in reject_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                last_call = max(
                    last_call,
                    int(obj.get("call", 0)),
                )
            except Exception:
                pass

    if state_path.exists():
        try:
            state = json.loads(
                state_path.read_text(encoding="utf-8")
            )
            last_call = max(
                last_call,
                int(
                    state.get("stats", {}).get(
                        "calls",
                        0,
                    )
                ),
            )
        except Exception:
            pass

    return (
        seen,
        accepted,
        last_call,
        decisions,
        risks,
        scenarios,
    )


base.sample_policy_facts = sample_policy_facts
base.derive_target = derive_target
base.build_generation_prompt = build_generation_prompt
base.validate_user_text = validate_user_text
base.load_existing = load_existing


if __name__ == "__main__":
    base.main()
