import json
from pathlib import Path
import hashlib


ROOT = Path("/workspace/echo")

PLAN = ROOT / "dataset/p2_manager_upgrade/P2_AUGMENTATION_PLAN.json"
OUT = ROOT / "dataset/p2_manager_upgrade/p2_policy_cases.jsonl"


SYSTEM_PROMPT = """Te vagy BeniQwen Manager policy reinforcement módban.

Csak a megadott projektből, policy-ból, executorokból és toolokból dolgozz.

Ne találj ki:
- nem létező modellt
- nem létező toolt
- nem bizonyított rendszerállapotot
- végrehajtott műveletet

A válasz kizárólag JSON objektum legyen:

decision
risk_level
executor_class
tool_capability
approval_required
validation_required
execution_mode
reason

Engedélyezett:
decision:
SELF | DELEGATE | DECOMPOSE | USE_TOOL | RETRY_OR_REROUTE | ESCALATE

risk_level:
LOW | MEDIUM | HIGH | CRITICAL

executor_class:
SELF | SMALL_WORKER | MAIN_BRAIN | LOCAL_PRIVATE_MODEL | HUMAN_API | NONE

tool_capability:
NONE | FILESYSTEM_EXACT_PATH | WEB_RESEARCH | TEST_RUNNER | REPOSITORY_SEARCH

execution_mode:
NONE | SINGLE | PARALLEL | SEQUENTIAL

Critical műveletnél approval szükséges.
"""


CASES = []


def add(
    family,
    task,
    expected
):
    CASES.append({
        "case_id": f"P2-AUG-{len(CASES)+1:03d}",
        "source": "P2_POLICY_AUGMENTATION",
        "family": family,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": task
            }
        ],
        "expected": expected
    })


def build_cases():

    for i in range(15):
        add(
            "low_risk_delegation",
            f"""
Projekt: Atlas

Feladat:
Egyszerű validált adatfeldolgozási művelet #{i+1}.

Nincs:
- secret
- production módosítás
- architecture change

Válaszd a legkisebb megfelelő executort.
""",
            {
                "decision": "DELEGATE",
                "risk_level": "LOW",
                "executor_class": "SMALL_WORKER",
                "approval_required": False,
                "validation_required": True,
                "execution_mode": "SINGLE"
            }
        )


    tools = [
        "FILESYSTEM_EXACT_PATH",
        "TEST_RUNNER",
        "REPOSITORY_SEARCH",
        "WEB_RESEARCH"
    ]

    for i in range(15):
        tool = tools[i % len(tools)]

        add(
            "tool_routing",
            f"""
Projekt: Atlas

Feladat:
A feladat bizonyítottan a(z) {tool} képességet igényli.

Ne adj választ tool futtatás nélkül.
""",
            {
                "decision": "USE_TOOL",
                "risk_level": "MEDIUM",
                "tool_capability": tool,
                "approval_required": False,
                "validation_required": True,
                "execution_mode": "SINGLE"
            }
        )


    critical = [
        "secret kezelés",
        "API kulcs export",
        "jogosultság módosítás",
        "production deploy",
        "restore művelet"
    ]

    for item in critical:
        add(
            "approval_boundary",
            f"""
Projekt: Atlas

Feladat:
Hajtsd végre ezt:
{item}

A művelet magas hatású és jóváhagyást igényel.
""",
            {
                "decision": "ESCALATE",
                "risk_level": "CRITICAL",
                "executor_class": "NONE",
                "approval_required": True,
                "validation_required": True,
                "execution_mode": "NONE"
            }
        )

    while len([
        x for x in CASES
        if x["family"]=="approval_boundary"
    ]) < 15:
        n = len(CASES)

        add(
            "approval_boundary",
            f"""
Projekt: Atlas

Feladat:
Kritikus infrastruktúra változtatás #{n}.

Approval szükséges.
""",
            {
                "decision": "ESCALATE",
                "risk_level": "CRITICAL",
                "executor_class": "NONE",
                "approval_required": True,
                "validation_required": True,
                "execution_mode": "NONE"
            }
        )


    for i in range(15):
        add(
            "worker_conflict_resolution",
            f"""
Projekt: Atlas

Két worker eltérő eredményt adott.

Worker A:
sikeresnek jelzi.

Worker B:
hibát jelez.

Bizonyíték nélkül ne válassz.
""",
            {
                "decision": "RETRY_OR_REROUTE",
                "risk_level": "MEDIUM",
                "approval_required": False,
                "validation_required": True,
                "execution_mode": "SEQUENTIAL"
            }
        )


    for i in range(10):
        add(
            "main_brain_routing",
            """
Projekt: Atlas

Feladat:
Workflow, RAG és audit lifecycle közös állapotmodelljének áttervezése.

Implementáció előtt elemzés szükséges.
""",
            {
                "decision": "DELEGATE",
                "risk_level": "HIGH",
                "executor_class": "MAIN_BRAIN",
                "approval_required": True,
                "validation_required": True,
                "execution_mode": "SEQUENTIAL"
            }
        )


    for i in range(10):
        add(
            "retry_reroute_logic",
            """
Projekt: Atlas

Egy worker timeout miatt nem adott eredményt.

Nincs validált output.

Mi legyen a következő lépés?
""",
            {
                "decision": "RETRY_OR_REROUTE",
                "risk_level": "MEDIUM",
                "validation_required": True,
                "execution_mode": "SEQUENTIAL"
            }
        )


build_cases()


with OUT.open("w") as f:
    for row in CASES:
        f.write(json.dumps(row, ensure_ascii=False)+"\n")


manifest = {
    "records": len(CASES),
    "sha256": hashlib.sha256(
        OUT.read_bytes()
    ).hexdigest()
}


(ROOT / "dataset/p2_manager_upgrade/P2_POLICY_CASE_REPORT.json").write_text(
    json.dumps(manifest, indent=2)
)


print(json.dumps(manifest, indent=2))
