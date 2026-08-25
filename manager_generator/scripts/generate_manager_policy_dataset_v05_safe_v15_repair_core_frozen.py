import json

import generate_manager_policy_dataset_v05_safe_v12_frozen as v12
import generate_manager_policy_dataset_v05_safe_v14_quality_frozen as v14


VERSION = "V05_SAFE_POLICY_V15_REPAIR_CORE"


DIRECT_ACCEPT = "DIRECT_ACCEPT"
REPAIR = "REPAIR"
REJECT = "REJECT"


def route_user(
    user,
    scenario,
    facts,
):
    result = v14.classify_user_quality(
        user,
        scenario,
        facts,
    )

    classification = result[
        "classification"
    ]

    if classification == v14.PASS:
        route = DIRECT_ACCEPT

    elif classification == v14.REPAIRABLE:
        route = REPAIR

    elif classification == v14.SEMANTIC_REJECT:
        route = REJECT

    else:
        raise ValueError(
            "UNKNOWN_V14_CLASSIFICATION:"
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
    return list(
        v12.v10.v9.mandatory_anchors(
            scenario,
            facts,
        )
    )


def build_repair_prompt(
    bad_user,
    scenario,
    facts,
    quality_result,
):
    if (
        quality_result["route"]
        != REPAIR
    ):
        raise ValueError(
            "REPAIR_PROMPT_FOR_NONREPAIRABLE_INPUT"
        )

    source_draft = (
        quality_result[
            "source_draft"
        ]
    )

    anchors = mandatory_anchors(
        scenario,
        facts,
    )

    anchors_text = "\n".join(
        f"- {anchor}"
        for anchor in anchors
    )

    quality_code = quality_result[
        "code"
    ]

    return f"""
Az alábbi magyar felhasználói kérés nyelvileg hibás vagy gépies.
Javítsd természetes, világos szakmai magyarra.

FONTOS:
- kizárólag a megfogalmazást javítsd;
- az eredeti technikai jelentést ne változtasd meg;
- ne adj hozzá új tényt;
- ne adj hozzá új rendszerállapotot;
- ne adj hozzá új eseményt;
- ne adj hozzá időpontot vagy időablakot;
- ne adj hozzá garanciát vagy következményt;
- ne adj hozzá új műveletet;
- ne változtass ellenőrzést végrehajtássá;
- ne változtass olvasási feladatot tervezéssé;
- ne változtass végrehajtást puszta tervvé;
- ne változtass production scope-ot;
- ne változtass jóváhagyási igényt;
- ne változtass secret-kezelési jelentést;
- ne írj belső policy mezőket vagy generálási utasításokat;
- legalább 3 teljes mondatot használj;
- a végeredmény 220 és 750 karakter közötti legyen.

A MEGENGEDETT TÉNYTARTALOM determinisztikus forrása:

--- SOURCE START ---
{source_draft}
--- SOURCE END ---

A JAVÍTANDÓ FELHASZNÁLÓI KÉRÉS:

--- USER START ---
{bad_user}
--- USER END ---

A korábbi quality gate hibakódja:
{quality_code}

A következő jelentéshordozó kifejezések maradjanak meg:

{anchors_text}

Kizárólag egyetlen JSON objektumot adj vissza:

{{"user":"a javított magyar felhasználói kérés"}}
""".strip()


def validate_repaired_user(
    repaired_user,
    scenario,
    facts,
):
    """
    Repair után nincs második repair.

    Kizárólag a teljes V14 PASS eredmény fogadható el.
    """

    result = v14.classify_user_quality(
        repaired_user,
        scenario,
        facts,
    )

    if (
        result["classification"]
        != v14.PASS
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
