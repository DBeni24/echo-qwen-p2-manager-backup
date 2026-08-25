import generate_manager_policy_dataset_v05_safe_v09 as v9
import generate_manager_policy_dataset_v05_safe_v08 as v8
import generate_manager_policy_dataset_v05_safe_v05 as v5
import generate_manager_policy_dataset_v05_safe_v03 as v3
import generate_manager_policy_dataset_v05_safe_v01 as base


base.VERSION = "V05_SAFE_POLICY_V10_COMPATIBILITY"

_BASE_SAMPLE = v5.sample_policy_facts
_V08_SCENARIO_SENTENCE = v8.scenario_sentence
_V08_OPERATION_REQUEST = v8.operation_request


# ======================================================================
# SCENARIO <-> POLICY FACT COMPATIBILITY
# ======================================================================

def validate_policy_compatibility(scenario, facts):
    op = facts["operation_type"]
    evidence = facts["evidence_quality"]

    # ----------------------------------------------------------
    # Globális invariánsok
    # ----------------------------------------------------------

    if facts["secret_exposure_requested"]:
        if op != "SECRET_DISCLOSURE":
            raise ValueError(
                "COMPAT_SECRET_EXPOSURE_OPERATION"
            )
    elif op == "SECRET_DISCLOSURE":
        raise ValueError(
            "COMPAT_DISCLOSURE_WITHOUT_SECRET_EXPOSURE"
        )

    if facts["destructive"]:
        if op != "EXECUTE_DESTRUCTIVE":
            raise ValueError(
                "COMPAT_DESTRUCTIVE_OPERATION"
            )

    if op == "EXECUTE_DESTRUCTIVE":
        if not facts["destructive"]:
            raise ValueError(
                "COMPAT_DESTRUCTIVE_FLAG_MISSING"
            )

    if op in {
        "EXECUTE_REVERSIBLE",
        "EXECUTE_DESTRUCTIVE",
    }:
        if not facts["explicit_human_approval_gate"]:
            raise ValueError(
                "COMPAT_EXECUTION_APPROVAL_MISSING"
            )

    # ----------------------------------------------------------
    # Scenario-specifikus invariánsok
    # ----------------------------------------------------------

    if scenario == "workflow_planning":
        # Ennek a scenario-családnak ténylegesen tervezési
        # feladatot kell tanítania.
        if op != "PLAN_ONLY":
            raise ValueError(
                "COMPAT_WORKFLOW_NOT_PLAN_ONLY"
            )

    elif scenario == "rag_conflict":
        # A skeleton két egymásnak ellentmondó forrást állít,
        # tehát a policy fact is CONFLICTING legyen.
        if op != "VALIDATE":
            raise ValueError(
                "COMPAT_RAG_NOT_VALIDATE"
            )

        if evidence != "CONFLICTING":
            raise ValueError(
                "COMPAT_RAG_NOT_CONFLICTING"
            )

    elif scenario == "false_state_detection":
        if op != "VALIDATE":
            raise ValueError(
                "COMPAT_FALSE_STATE_NOT_VALIDATE"
            )

    elif scenario == "approval_required":
        if op not in {
            "EXECUTE_REVERSIBLE",
            "EXECUTE_DESTRUCTIVE",
        }:
            raise ValueError(
                "COMPAT_APPROVAL_NOT_EXECUTION"
            )

        if not facts["explicit_human_approval_gate"]:
            raise ValueError(
                "COMPAT_APPROVAL_GATE_MISSING"
            )

        if facts["production_scope"] != "PRODUCTION":
            raise ValueError(
                "COMPAT_APPROVAL_NOT_PRODUCTION"
            )

    elif scenario == "secret_handling":
        if op not in {
            "PLAN_ONLY",
            "VALIDATE",
            "SECRET_DISCLOSURE",
        }:
            raise ValueError(
                "COMPAT_SECRET_BAD_OPERATION"
            )

        if op == "PLAN_ONLY":
            if facts["requires_tool"] or facts["requires_live_data"]:
                raise ValueError(
                    "COMPAT_SECRET_PLAN_REQUIRES_TOOL"
                )

        if op == "VALIDATE":
            if not (
                facts["requires_tool"]
                and facts["requires_live_data"]
            ):
                raise ValueError(
                    "COMPAT_SECRET_VALIDATE_NO_TOOL"
                )

        if op == "SECRET_DISCLOSURE":
            if not facts["secret_exposure_requested"]:
                raise ValueError(
                    "COMPAT_SECRET_DISCLOSURE_FLAG"
                )

    elif scenario == "checkpoint_review":
        if op != "VALIDATE":
            raise ValueError(
                "COMPAT_CHECKPOINT_NOT_VALIDATE"
            )

    elif scenario == "provider_routing":
        if op not in {
            "PLAN_ONLY",
            "READ_ONLY",
            "VALIDATE",
        }:
            raise ValueError(
                "COMPAT_PROVIDER_BAD_OPERATION"
            )

    elif scenario == "tool_selection":
        if op not in {
            "PLAN_ONLY",
            "READ_ONLY",
            "VALIDATE",
        }:
            raise ValueError(
                "COMPAT_TOOL_SELECTION_BAD_OPERATION"
            )

    elif scenario == "worker_assignment":
        if op not in {
            "DELEGATE",
            "READ_ONLY",
            "VALIDATE",
        }:
            raise ValueError(
                "COMPAT_WORKER_BAD_OPERATION"
            )

        if op == "DELEGATE":
            if not facts["delegation_requested"]:
                raise ValueError(
                    "COMPAT_WORKER_DELEGATION_MISSING"
                )
        else:
            if facts["delegation_requested"]:
                raise ValueError(
                    "COMPAT_WORKER_UNEXPECTED_DELEGATION"
                )

    else:
        raise ValueError(
            f"COMPAT_UNKNOWN_SCENARIO:{scenario}"
        )

    return facts


# ======================================================================
# FACT SAMPLER FIX
# ======================================================================

def sample_policy_facts(rng, scenario):
    facts = dict(
        _BASE_SAMPLE(
            rng,
            scenario,
        )
    )

    # ----------------------------------------------------------
    # 1. workflow_planning
    #
    # A scenario valódi tervezést kér.
    # A régi tool_read/evidence_conflict profile-ok hasznos
    # változatosságát megtartjuk, de a rendszer-művelet PLAN_ONLY.
    # ----------------------------------------------------------

    if scenario == "workflow_planning":
        old_profile = facts["profile"]

        if old_profile == "tool_read":
            facts["profile"] = "planning_live"
            facts["operation_type"] = "PLAN_ONLY"

            # Tool/live megmarad:
            # a tervhez előbb aktuális workflow-adat szükséges.
            facts["requires_tool"] = True
            facts["requires_live_data"] = True
            facts["external_side_effect"] = False
            facts["destructive"] = False
            facts["delegation_requested"] = False
            facts["explicit_human_approval_gate"] = False

        elif old_profile == "evidence_conflict":
            facts["profile"] = "planning_evidence"
            facts["operation_type"] = "PLAN_ONLY"

            facts["requires_tool"] = True
            facts["requires_live_data"] = True
            facts["external_side_effect"] = False
            facts["destructive"] = False
            facts["delegation_requested"] = False
            facts["explicit_human_approval_gate"] = False

        elif old_profile == "safe_self":
            # Már eleve helyes PLAN_ONLY.
            pass

        else:
            raise ValueError(
                f"WORKFLOW_UNKNOWN_PROFILE:{old_profile}"
            )

    # ----------------------------------------------------------
    # 2. rag_conflict
    #
    # Ha valódi forrásellentmondást írunk a user promptba,
    # a policy-fact ne lehessen PARTIAL/UNVERIFIED.
    # ----------------------------------------------------------

    if scenario == "rag_conflict":
        facts["evidence_quality"] = "CONFLICTING"
        facts["operation_type"] = "VALIDATE"
        facts["requires_tool"] = True
        facts["requires_live_data"] = True

    validate_policy_compatibility(
        scenario,
        facts,
    )

    # A V03 validator innen olvassa a policy contextet.
    v3.CURRENT_CONTEXT = {
        "scenario": scenario,
        "facts": facts,
    }

    return facts


# ======================================================================
# SECRET_HANDLING SAFE_SELF SKELETON FIX
# ======================================================================

def scenario_sentence(scenario, facts):
    if (
        scenario == "secret_handling"
        and facts["operation_type"] == "PLAN_ONLY"
    ):
        return (
            "A megadott <API_KEY> kezeléséhez biztonságos "
            "eljárási terv szükséges. A terv az azonosító, "
            "jogosultsági és rotációs szabályokra épüljön, "
            "és ne kérjen érzékeny tartalom megjelenítését."
        )

    if scenario == "rag_conflict":
        # Az evidence_sentence mondja ki magát az eltérést.
        # Itt csak a két forrás meglétét rögzítjük, így nem
        # ismételjük ugyanazt kétszer a deterministic draftban.
        return (
            "Ugyanarról az elemről egy belső dokumentum és "
            "egy másik adatforrás áll rendelkezésre."
        )

    return _V08_SCENARIO_SENTENCE(
        scenario,
        facts,
    )


def operation_request(scenario, facts):
    if (
        scenario == "tool_selection"
        and facts["operation_type"] == "READ_ONLY"
    ):
        return (
            "Nézd meg az aktuális diagnosztikai adatokat, "
            "és foglald össze, hogy az elérhető eszközök "
            "milyen vizsgálatokra alkalmasak. Ebben a lépésben "
            "csak az alkalmassági információkat gyűjtsd össze."
        )

    return _V08_OPERATION_REQUEST(
        scenario,
        facts,
    )


# A V08 compose_draft ezt a modulszintű függvényt használja.
v8.scenario_sentence = scenario_sentence
v8.operation_request = operation_request


# ======================================================================
# PATCH GENERATOR
# ======================================================================

base.sample_policy_facts = sample_policy_facts
base.derive_target = v9.derive_target
base.build_generation_prompt = v9.build_generation_prompt
base.validate_user_text = v9.validate_user_text


if __name__ == "__main__":
    base.main()
