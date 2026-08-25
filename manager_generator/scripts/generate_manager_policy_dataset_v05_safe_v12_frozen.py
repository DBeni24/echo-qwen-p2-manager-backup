import hashlib
import json

import generate_manager_policy_dataset_v05_safe_v10_frozen as v10
import generate_manager_policy_dataset_v05_safe_v01 as base


base.VERSION = "V05_SAFE_POLICY_V12_REASON_QUALITY"

_V10_DERIVE_TARGET = v10.v9.derive_target


# ======================================================================
# DETERMINISTIC VARIANT SELECTION
# ======================================================================

def stable_variant(scenario, facts, variants):
    payload = {
        "scenario": scenario,
        "facts": facts,
    }

    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256(raw).digest()

    idx = int.from_bytes(
        digest[:8],
        "big",
    ) % len(variants)

    return variants[idx]


# ======================================================================
# LANGUAGE HELPERS
# ======================================================================

def scope_phrase(facts):
    return {
        "LOCAL": "lokális környezetben",
        "TEAM": "belső csapatkörnyezetben",
        "PRODUCTION": "production környezetben",
    }[facts["production_scope"]]


def evidence_phrase(facts):
    return {
        "VERIFIED":
            "a szükséges információ ellenőrzött",
        "PARTIAL":
            "a rendelkezésre álló bizonyíték csak részleges",
        "UNVERIFIED":
            "az aktuális állapot még nincs megbízhatóan igazolva",
        "CONFLICTING":
            "a rendelkezésre álló források ellentmondanak egymásnak",
    }[facts["evidence_quality"]]


def subject_forms(scenario):
    """
    nominative:
        mondat alanyaként használható.

    about:
        "... kapcsán" / "... ellenőrzéséhez" szerkezetben
        természetesen használható.
    """

    return {
        "workflow_planning": {
            "nominative": "a workflow és annak lépései",
            "about": "a workflow megtervezéséhez",
        },
        "rag_conflict": {
            "nominative": "az információforrások megbízhatósága",
            "about": "az információforrások összevetéséhez",
        },
        "false_state_detection": {
            "nominative": "a jelentett rendszerállapot",
            "about": "a jelentett rendszerállapot ellenőrzéséhez",
        },
        "tool_selection": {
            "nominative": "a diagnosztikai eszközök alkalmassága",
            "about": "a diagnosztikai eszközök értékeléséhez",
        },
        "approval_required": {
            "nominative": "a production művelet",
            "about": "a production művelet végrehajtásához",
        },
        "secret_handling": {
            "nominative": "a hitelesítő adat kezelési helyzete",
            "about": "a hitelesítő adat biztonságos kezeléséhez",
        },
        "checkpoint_review": {
            "nominative": "a checkpoint állapota",
            "about": "a checkpoint ellenőrzéséhez",
        },
        "provider_routing": {
            "nominative": "a szolgáltatói vagy modellútvonal",
            "about": "a szolgáltatói vagy modellútvonal kiválasztásához",
        },
        "worker_assignment": {
            "nominative": "a worker alkalmassága",
            "about": "a worker alkalmasságának megítéléséhez",
        },
    }[scenario]


# ======================================================================
# REASON
# ======================================================================

def derive_reason(scenario, facts, target):
    operation = facts["operation_type"]
    decision = target["decision"]

    scope = scope_phrase(facts)
    evidence = evidence_phrase(facts)

    forms = subject_forms(scenario)
    subject = forms["nominative"]
    about = forms["about"]

    # ----------------------------------------------------------
    # SECRET DISCLOSURE
    # ----------------------------------------------------------

    if facts["secret_exposure_requested"]:
        return stable_variant(
            scenario,
            facts,
            [
                (
                    "A kérés hitelesítő adat közvetlen felfedését "
                    "igényli, ezért automatikus végrehajtás helyett "
                    "biztonsági blokkolás és felülvizsgálat szükséges."
                ),
                (
                    "A feladat egy érzékeny hitelesítő adat teljes "
                    "értékének kiadására irányul; ezt a rendszer nem "
                    "hajthatja végre automatikusan."
                ),
                (
                    "A kért művelet érzékeny hitelesítő adat "
                    "felfedésével járna, ezért a kérést blokkolni és "
                    "magasabb szinten felülvizsgálni szükséges."
                ),
            ],
        )

    # ----------------------------------------------------------
    # DESTRUCTIVE
    # ----------------------------------------------------------

    if operation == "EXECUTE_DESTRUCTIVE":
        return stable_variant(
            scenario,
            facts,
            [
                (
                    "A production művelet destruktív vagy nehezen "
                    "visszafordítható, ezért végrehajtás előtt emberi "
                    "jóváhagyás és validáció szükséges."
                ),
                (
                    "A kért production változtatás magas kockázatú és "
                    "súlyos következménnyel járhat; csak validáció és "
                    "explicit emberi jóváhagyás után hajtható végre."
                ),
                (
                    "A művelet visszaállíthatósága korlátozott, ezért "
                    "a production állapot módosítása előtt kötelező a "
                    "felülvizsgálat és az emberi engedély."
                ),
            ],
        )

    # ----------------------------------------------------------
    # REVERSIBLE EXECUTION
    # ----------------------------------------------------------

    if (
        operation == "EXECUTE_REVERSIBLE"
        or facts["explicit_human_approval_gate"]
    ):
        return stable_variant(
            scenario,
            facts,
            [
                (
                    "A production állapotot módosító művelet "
                    "visszaállítható, de végrehajtás előtt explicit "
                    "emberi jóváhagyás és validáció szükséges."
                ),
                (
                    "A kért production változtatás tényleges "
                    "állapotmódosítással jár, ezért csak ellenőrzés "
                    "és emberi jóváhagyás után hajtható végre."
                ),
                (
                    "A művelet végrehajtható, de production hatása "
                    "miatt előzetes validációra és jóváhagyási kapura "
                    "van szükség."
                ),
            ],
        )

    # ----------------------------------------------------------
    # CONFLICT / UNVERIFIED ESCALATION
    # ----------------------------------------------------------

    conflict_escalation = (
        facts["evidence_quality"] in {
            "CONFLICTING",
            "UNVERIFIED",
        }
        and (
            facts["impact"] == "HIGH"
            or facts["production_scope"] == "PRODUCTION"
        )
        and decision == "ESCALATE"
        and target["risk_level"] == "HIGH"
    )

    if conflict_escalation:
        return stable_variant(
            scenario,
            facts,
            [
                (
                    f"{subject.capitalize()} kapcsán {evidence}; "
                    f"{scope} ezért magasabb szintű validáció "
                    "szükséges a döntés előtt."
                ),
                (
                    f"{scope.capitalize()} {subject} nem igazolható "
                    "kellő bizonyossággal, ezért a döntést validációra "
                    "és magasabb szintű felülvizsgálatra kell emelni."
                ),
                (
                    f"{about.capitalize()} nem áll rendelkezésre "
                    "kellően megbízható bizonyíték; a kockázati szint "
                    "miatt validáció és magasabb szintű felülvizsgálat "
                    "szükséges."
                ),
            ],
        )

    # ----------------------------------------------------------
    # DELEGATION
    # ----------------------------------------------------------

    if decision == "DELEGATE":
        complexity = {
            "ROUTINE": "rutinszerű",
            "BOUNDED": "jól körülhatárolt",
            "COMPLEX": "összetett",
        }[facts["task_complexity"]]

        return stable_variant(
            scenario,
            facts,
            [
                (
                    f"A feladat {complexity} és delegálható; "
                    "a végrehajtó a hatás, az alkalmasság és a "
                    "feladat összetettsége alapján választható ki."
                ),
                (
                    f"{subject.capitalize()} alapján a feladat "
                    "átadható egy megfelelő workernek, mert a "
                    "kockázati és komplexitási feltételek ezt "
                    "lehetővé teszik."
                ),
                (
                    "A feladat végrehajtása delegálható; a megfelelő "
                    "worker kiválasztását a hatás és a komplexitás "
                    "határozza meg."
                ),
            ],
        )

    # ----------------------------------------------------------
    # TOOL
    # ----------------------------------------------------------

    if decision == "USE_TOOL":
        return stable_variant(
            scenario,
            facts,
            [
                (
                    f"{about.capitalize()} friss rendszeradat "
                    f"szükséges, miközben {evidence}; ezért eszközös "
                    "lekérdezés és validáció indokolt."
                ),
                (
                    f"{subject.capitalize()} csak aktuális "
                    "rendszeradatból ellenőrizhető megfelelően, ezért "
                    "a döntés előtt eszközös lekérdezés szükséges."
                ),
                (
                    f"Mivel {about} aktuális és ellenőrizhető adat "
                    "szükséges, a feladatot eszközös validációval "
                    "kell folytatni."
                ),
                (
                    f"{scope.capitalize()} {about} friss bizonyíték "
                    "szükséges; az állapotot ezért "
                    "rendszerlekérdezéssel kell megerősíteni."
                ),
            ],
        )

    # ----------------------------------------------------------
    # SELF
    # ----------------------------------------------------------

    if decision == "SELF":
        return stable_variant(
            scenario,
            facts,
            [
                (
                    f"{subject.capitalize()} a megadott, ellenőrzött "
                    "tényekből megítélhető, ezért nincs szükség élő "
                    "rendszeradatra vagy állapotmódosításra."
                ),
                (
                    "A feladat tervezési vagy döntési jellegű, és a "
                    "szükséges információ már rendelkezésre áll; "
                    "eszközös lekérdezés nem szükséges."
                ),
                (
                    f"{about.capitalize()} minden szükséges információ "
                    "rendelkezésre áll, ezért külső rendszerlekérdezés "
                    "vagy végrehajtási művelet nem szükséges."
                ),
            ],
        )

    raise ValueError(
        f"UNHANDLED_REASON_BRANCH:{decision}:{operation}"
    )


def derive_target(scenario, facts):
    target = dict(
        _V10_DERIVE_TARGET(
            scenario,
            facts,
        )
    )

    before = {
        k: v
        for k, v in target.items()
        if k != "reason"
    }

    target["reason"] = derive_reason(
        scenario,
        facts,
        target,
    )

    after = {
        k: v
        for k, v in target.items()
        if k != "reason"
    }

    if before != after:
        raise ValueError(
            "V12_NON_REASON_TARGET_MUTATION"
        )

    return target


base.sample_policy_facts = v10.sample_policy_facts
base.derive_target = derive_target
base.build_generation_prompt = v10.v9.build_generation_prompt
base.validate_user_text = v10.v9.validate_user_text


if __name__ == "__main__":
    base.main()
