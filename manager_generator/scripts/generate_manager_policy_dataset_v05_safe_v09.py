import re

import generate_manager_policy_dataset_v05_safe_v08 as v8
import generate_manager_policy_dataset_v05_safe_v06 as v6
import generate_manager_policy_dataset_v05_safe_v05 as v5
import generate_manager_policy_dataset_v05_safe_v04 as v4
import generate_manager_policy_dataset_v05_safe_v03 as v3
import generate_manager_policy_dataset_v05_safe_v01 as base


base.VERSION = "V05_SAFE_POLICY_V09_QUALITY"

_V03_DERIVE_TARGET = v3.derive_target
_V08_VALIDATE = v8.validate_user_text


# ======================================================================
# TARGET REASON FIX
# ======================================================================

def derive_target(scenario, facts):
    target = dict(
        _V03_DERIVE_TARGET(
            scenario,
            facts,
        )
    )

    conflict_escalation = (
        facts["evidence_quality"] in {
            "CONFLICTING",
            "UNVERIFIED",
        }
        and (
            facts["impact"] == "HIGH"
            or facts["production_scope"] == "PRODUCTION"
        )
        and target["decision"] == "ESCALATE"
        and target["risk_level"] == "HIGH"
        and not facts["secret_exposure_requested"]
        and facts["operation_type"]
        not in {
            "EXECUTE_REVERSIBLE",
            "EXECUTE_DESTRUCTIVE",
        }
    )

    if conflict_escalation:
        if (
            facts["impact"] == "HIGH"
            and facts["production_scope"] == "PRODUCTION"
        ):
            target["reason"] = (
                "A döntés nagy hatású production környezetet érint, "
                "miközben az állapotinformáció ellentmondó vagy nem "
                "igazolt; előbb validáció szükséges."
            )

        elif facts["production_scope"] == "PRODUCTION":
            target["reason"] = (
                "Production környezetben az állapotinformáció "
                "ellentmondó vagy nem igazolt; a biztonságos döntéshez "
                "előbb validáció és magasabb szintű felülvizsgálat kell."
            )

        else:
            target["reason"] = (
                "A döntés nagy hatású, miközben az állapotinformáció "
                "ellentmondó vagy nem igazolt; előbb validáció és "
                "magasabb szintű felülvizsgálat szükséges."
            )

    return target


# ======================================================================
# V09 PROMPT
# ======================================================================

def mandatory_anchors(scenario, facts):
    anchors = list(
        v8.mandatory_anchors(
            scenario,
            facts,
        )
    )

    # A V06 direct-disclosure regex explicit disclosure igét is vár.
    if facts["operation_type"] == "SECRET_DISCLOSURE":
        if not any(
            x.lower() == "írd ki"
            for x in anchors
        ):
            anchors.append("írd ki")

    return anchors


def build_generation_prompt(scenario, facts):
    draft = v8.compose_draft(
        scenario,
        facts,
    )

    anchors = mandatory_anchors(
        scenario,
        facts,
    )

    anchor_text = ", ".join(
        f'"{x}"'
        for x in anchors
    )

    return f"""
Fogalmazd át az alábbi technikai kérést természetes magyar
felhasználói kérdéssé.

{draft}

A végső kérés:
- legalább 3 teljes mondatból álljon;
- 220 és 750 karakter közötti legyen;
- tartsa meg az összes lényeges tényt;
- tartsa meg pontosan a kért művelet jellegét;
- ne adjon hozzá új műveletet;
- ne változtasson ellenőrzést végrehajtássá;
- ne változtasson olvasási feladatot tervezéssé;
- ne változtasson végrehajtást puszta tervvé;
- ne találjon ki új rendszerállapotot;
- ne találjon ki terméknevet, verziót vagy AION-funkciót;
- ne utaljon arra, hogy szöveget szerkeszt, szabályokat követ,
  vagy egy belső generálási utasításból dolgozik;
- egyszerű, természetes szakmai magyar nyelvet használjon.

Ezek a kifejezések maradjanak meg a végső kérésben:
{anchor_text}

Kizárólag egyetlen JSON objektumot adj vissza ebben a formában:
{{"user":"a kész magyar felhasználói kérés"}}
""".strip()


# ======================================================================
# QUALITY + META LEAKAGE GATES
# ======================================================================

META_LEAK_PATTERNS = [
    r"\btartalmi\s+vázlat\b",
    r"\birányelv(?:ek)?\s+szerint\b",
    r"\bpolicy\s+szerint\b",
    r"\bkövetelmény(?:ek)?\s+szerint\b",
    r"\bforrásszöveg\b",
    r"\bszerkesztési\s+utasítás\b",
    r"\bgenerálási\s+utasítás\b",
    r"\ba\s+jelentést\s+(?:ne|nem)\s+változt",
    r"\ba\s+kért\s+műveletet\s+(?:ne|nem)\s+változt",
]

KNOWN_LANGUAGE_BAD_PATTERNS = [
    r"\bidőtünk\b",
    r"\belemd\b",
    r"\baz\s+<api_key>\s+kapcsolódó\b",
    r"\baz\s+<secret>\s+kapcsolódó\b",
]

WORKFLOW_READONLY_DRIFT_PATTERNS = [
    r"\bhatározd\s+meg\s+a\s+folyamat\s+sorrendjét\b",
    r"\bhatározzuk\s+meg\s+a\s+folyamat\s+sorrendjét\b",
    r"\btervezd\s+meg\b",
    r"\bkészíts\s+(?:egy\s+)?tervet\b",
    r"\bállítsd\s+össze\s+(?:a\s+)?workflow\b",
]


def validate_user_text(user):
    user = _V08_VALIDATE(user)

    if v3.CURRENT_CONTEXT is None:
        raise ValueError(
            "NO_POLICY_CONTEXT_V09"
        )

    scenario = v3.CURRENT_CONTEXT["scenario"]
    facts = v3.CURRENT_CONTEXT["facts"]
    low = user.lower()

    for pattern in META_LEAK_PATTERNS:
        if re.search(
            pattern,
            low,
            re.I,
        ):
            raise ValueError(
                "PROMPT_META_LEAKAGE"
            )

    for pattern in KNOWN_LANGUAGE_BAD_PATTERNS:
        if re.search(
            pattern,
            low,
            re.I,
        ):
            raise ValueError(
                "LANGUAGE_QUALITY_KNOWN_BAD"
            )

    if (
        scenario == "workflow_planning"
        and facts["operation_type"] == "READ_ONLY"
    ):
        for pattern in WORKFLOW_READONLY_DRIFT_PATTERNS:
            if re.search(
                pattern,
                low,
                re.I,
            ):
                raise ValueError(
                    "WORKFLOW_READONLY_DRIFT"
                )

    return user


# ======================================================================
# PATCH BASE MAIN
# ======================================================================

base.derive_target = derive_target
v3.derive_target = derive_target

base.build_generation_prompt = build_generation_prompt
base.validate_user_text = validate_user_text


if __name__ == "__main__":
    base.main()
