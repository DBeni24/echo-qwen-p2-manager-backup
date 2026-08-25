import re

import generate_manager_policy_dataset_v05_safe_v14_quality_frozen as v14


VERSION = "V05_SAFE_POLICY_V16_QUALITY_EXTENSION"


PASS = v14.PASS
REPAIRABLE = v14.REPAIRABLE
SEMANTIC_REJECT = v14.SEMANTIC_REJECT


# ======================================================================
# V16 SEMANTIC EXTENSIONS
#
# High precision only.
#
# Important rule:
# a matched phrase is considered unsupported only when the same semantic
# fragment is not already present in the deterministic source draft.
# ======================================================================

UNSUPPORTED_URGENCY_PATTERNS = [
    (
        r"\bazonnal\b",
        "UNSUPPORTED_URGENCY_IMMEDIATE",
    ),
    (
        r"\bhaladéktalanul\b",
        "UNSUPPORTED_URGENCY_IMMEDIATE",
    ),
    (
        r"\bsürgősen\b",
        "UNSUPPORTED_URGENCY",
    ),
]


UNSUPPORTED_HISTORY_PATTERNS = [
    (
        r"\bkorábbi\s+döntés\w*\b",
        "UNSUPPORTED_DECISION_HISTORY",
    ),
    (
        r"\belőző\s+döntés\w*\b",
        "UNSUPPORTED_DECISION_HISTORY",
    ),
    (
        r"\bkorábbi\s+eredmény\w*\b",
        "UNSUPPORTED_RESULT_HISTORY",
    ),
]


UNSUPPORTED_GUARANTEE_PATTERNS = [
    (
        r"\bhibamentes\w*\b",
        "UNSUPPORTED_SUCCESS_GUARANTEE",
    ),
    (
        r"\bgarantált\w*\b",
        "UNSUPPORTED_GUARANTEE",
    ),
    (
        r"\bbiztosan\s+(?:működ\w*|sikerül\w*)\b",
        "UNSUPPORTED_SUCCESS_GUARANTEE",
    ),
]


UNSUPPORTED_CONCRETE_EVIDENCE_PATTERNS = [
    (
        r"\badatbázis[-\s]?bizonyíték\w*\b",
        "UNSUPPORTED_DATABASE_EVIDENCE_SOURCE",
    ),
]


# ======================================================================
# NO-LIVE CONTRADICTION
#
# If facts say that no fresh/live data is required, the rewrite must not
# silently add a demand for current worker/system state.
# ======================================================================

NO_LIVE_CURRENT_STATE_PATTERNS = [
    (
        r"\b(?:worker\w*\s+)?"
        r"(?:aktuális|jelenlegi|legutóbbi)\s+"
        r"(?:állapot|státusz|kapacitás|terhelés)\w*\b",
        "UNEXPECTED_LIVE_STATE_REQUIREMENT",
    ),
    (
        r"\b(?:aktuális|jelenlegi|legutóbbi)\s+"
        r"(?:rendszerállapot|rendszeradat|metrika|napló)\w*\b",
        "UNEXPECTED_LIVE_STATE_REQUIREMENT",
    ),
]


# ======================================================================
# REPAIRABLE LANGUAGE / ROLE CONTAMINATION
# ======================================================================

REPAIRABLE_ROLE_PATTERNS = [
    (
        r"\bszívesen\s+segítek\b",
        "ROLE_CONTAMINATION_ASSISTANT_VOICE",
    ),
    (
        r"\börömmel\s+segítek\b",
        "ROLE_CONTAMINATION_ASSISTANT_VOICE",
    ),
    (
        r"\bpersze[,! ]+\s*segítek\b",
        "ROLE_CONTAMINATION_ASSISTANT_VOICE",
    ),
]


REPAIRABLE_LANGUAGE_PATTERNS = [
    (
        r"\bhelyes\s+lépés\s+sorrend\w*\b",
        "LANGUAGE_BAD_STEP_ORDER",
    ),
    (
        r"\bfeladatonkhoz\b",
        "LANGUAGE_BAD_TASK_SUFFIX",
    ),
]


# ======================================================================
# HELPERS
# ======================================================================

def normalize(text):
    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()


def find_new_pattern(
    user,
    source_draft,
    patterns,
):
    """
    Return the first matched fragment that is present in the generated
    user text but not literally present in the deterministic source.

    These pattern sets intentionally target concepts that should normally
    survive almost verbatim if they were part of the source.
    """

    low_user = normalize(user)
    low_source = normalize(
        source_draft
    )

    for pattern, code in patterns:
        match = re.search(
            pattern,
            low_user,
            re.I,
        )

        if match is None:
            continue

        fragment = normalize(
            match.group(0)
        )

        if fragment not in low_source:
            return {
                "code": code,
                "fragment": match.group(0),
            }

    return None


def semantic_result(
    code,
    fragment,
    source_draft,
):
    return {
        "classification":
            SEMANTIC_REJECT,

        "code":
            code,

        "fragment":
            fragment,

        "source_draft":
            source_draft,
    }


def repairable_result(
    code,
    fragment,
    source_draft,
):
    return {
        "classification":
            REPAIRABLE,

        "code":
            code,

        "fragment":
            fragment,

        "source_draft":
            source_draft,
    }


# ======================================================================
# CLASSIFIER
# ======================================================================

def classify_user_quality(
    user,
    scenario,
    facts,
):
    # ----------------------------------------------------------
    # 1. Frozen V14 remains authoritative.
    # ----------------------------------------------------------

    base_result = (
        v14.classify_user_quality(
            user,
            scenario,
            facts,
        )
    )

    if (
        base_result["classification"]
        != v14.PASS
    ):
        return base_result

    source_draft = (
        base_result[
            "source_draft"
        ]
    )

    # ----------------------------------------------------------
    # 2. requires_live_data=False contradiction
    # ----------------------------------------------------------

    if not facts["requires_live_data"]:
        issue = find_new_pattern(
            user,
            source_draft,
            NO_LIVE_CURRENT_STATE_PATTERNS,
        )

        if issue is not None:
            return semantic_result(
                issue["code"],
                issue["fragment"],
                source_draft,
            )

    # ----------------------------------------------------------
    # 3. Unsupported urgency
    # ----------------------------------------------------------

    issue = find_new_pattern(
        user,
        source_draft,
        UNSUPPORTED_URGENCY_PATTERNS,
    )

    if issue is not None:
        return semantic_result(
            issue["code"],
            issue["fragment"],
            source_draft,
        )

    # ----------------------------------------------------------
    # 4. Unsupported history
    # ----------------------------------------------------------

    issue = find_new_pattern(
        user,
        source_draft,
        UNSUPPORTED_HISTORY_PATTERNS,
    )

    if issue is not None:
        return semantic_result(
            issue["code"],
            issue["fragment"],
            source_draft,
        )

    # ----------------------------------------------------------
    # 5. Unsupported guarantees
    # ----------------------------------------------------------

    issue = find_new_pattern(
        user,
        source_draft,
        UNSUPPORTED_GUARANTEE_PATTERNS,
    )

    if issue is not None:
        return semantic_result(
            issue["code"],
            issue["fragment"],
            source_draft,
        )

    # ----------------------------------------------------------
    # 6. Unsupported concrete evidence source
    # ----------------------------------------------------------

    issue = find_new_pattern(
        user,
        source_draft,
        UNSUPPORTED_CONCRETE_EVIDENCE_PATTERNS,
    )

    if issue is not None:
        return semantic_result(
            issue["code"],
            issue["fragment"],
            source_draft,
        )

    # ----------------------------------------------------------
    # 7. User/assistant role contamination
    # ----------------------------------------------------------

    issue = find_new_pattern(
        user,
        source_draft,
        REPAIRABLE_ROLE_PATTERNS,
    )

    if issue is not None:
        return repairable_result(
            issue["code"],
            issue["fragment"],
            source_draft,
        )

    # ----------------------------------------------------------
    # 8. Additional Hungarian quality
    # ----------------------------------------------------------

    issue = find_new_pattern(
        user,
        source_draft,
        REPAIRABLE_LANGUAGE_PATTERNS,
    )

    if issue is not None:
        return repairable_result(
            issue["code"],
            issue["fragment"],
            source_draft,
        )

    return {
        "classification":
            PASS,

        "code":
            "QUALITY_PASS",

        "source_draft":
            source_draft,
    }
