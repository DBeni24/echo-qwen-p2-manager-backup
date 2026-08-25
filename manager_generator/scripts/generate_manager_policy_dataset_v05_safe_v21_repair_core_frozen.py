import generate_manager_policy_dataset_v05_safe_v17_repair_core_frozen as v17
import generate_manager_policy_dataset_v05_safe_v20_quality_frozen as v20


VERSION = "V05_SAFE_POLICY_V21_REPAIR_CORE_V20"


DIRECT_ACCEPT = v17.DIRECT_ACCEPT
REPAIR = v17.REPAIR
REJECT = v17.REJECT


def route_user(
    user,
    scenario,
    facts,
):
    result = v20.classify_user_quality(
        user,
        scenario,
        facts,
    )

    classification = result[
        "classification"
    ]

    if classification == v20.PASS:
        route = DIRECT_ACCEPT

    elif classification == v20.REPAIRABLE:
        route = REPAIR

    elif classification == v20.SEMANTIC_REJECT:
        route = REJECT

    else:
        raise ValueError(
            "UNKNOWN_V18_CLASSIFICATION:"
            f"{classification}"
        )

    return {
        "route": route,
        "classification": classification,
        "code": result["code"],
        "source_draft": result["source_draft"],
        "fragment": result.get("fragment"),
    }


def mandatory_anchors(
    scenario,
    facts,
):
    return v17.mandatory_anchors(
        scenario,
        facts,
    )


def build_repair_prompt(
    bad_user,
    scenario,
    facts,
    quality_result,
):
    return v17.build_repair_prompt(
        bad_user,
        scenario,
        facts,
        quality_result,
    )


def validate_repaired_user(
    repaired_user,
    scenario,
    facts,
):
    result = v20.classify_user_quality(
        repaired_user,
        scenario,
        facts,
    )

    if (
        result["classification"]
        != v20.PASS
    ):
        raise ValueError(
            "REPAIR_NOT_PASS:"
            f"{result['classification']}:"
            f"{result['code']}"
        )

    return {
        "classification":
            result["classification"],

        "code":
            result["code"],

        "source_draft":
            result["source_draft"],
    }


def repair_audit_metadata(
    initial_result,
    repaired_result=None,
):
    return {
        "initial_classification":
            initial_result[
                "classification"
            ],

        "initial_code":
            initial_result[
                "code"
            ],

        "repair_attempted":
            initial_result["route"]
            == REPAIR,

        "repair_final_classification":
            (
                repaired_result[
                    "classification"
                ]
                if repaired_result
                is not None
                else None
            ),

        "repair_final_code":
            (
                repaired_result[
                    "code"
                ]
                if repaired_result
                is not None
                else None
            ),
    }
