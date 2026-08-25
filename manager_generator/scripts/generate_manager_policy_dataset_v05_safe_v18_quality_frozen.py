import re

import generate_manager_policy_dataset_v05_safe_v16_quality_frozen as v16


VERSION = "V05_SAFE_POLICY_V18_QUALITY_EXTENSION"


PASS = v16.PASS
REPAIRABLE = v16.REPAIRABLE
SEMANTIC_REJECT = v16.SEMANTIC_REJECT


# ======================================================================
# V18 HIGH-PRECISION SEMANTIC EXTENSIONS
# ======================================================================

NUMBER = (
    r"(?:egy|két|három|négy|öt|hat|hét|nyolc|kilenc|tíz|\d+)"
)


UNSUPPORTED_TASK_CONSTRAINT_PATTERNS = [
    (
        rf"\blegalább\s+{NUMBER}\s+lépés\w*\b",
        "UNSUPPORTED_MINIMUM_STEP_CONSTRAINT",
    ),
]


UNSUPPORTED_SOURCE_QUALIFIER_PATTERNS = [
    (
        r"\bkülső\s+adatforrás\w*\b",
        "UNSUPPORTED_EXTERNAL_SOURCE_QUALIFIER",
    ),
]


UNSUPPORTED_EVIDENCE_STRENGTH_PATTERNS = [
    (
        r"\bbizonyíték(?:ok)?\s+hiány\w*\b",
        "UNSUPPORTED_EVIDENCE_ABSENCE",
    ),
    (
        r"\bnincs(?:en)?\s+"
        r"(?:megfelelő\s+|megbízható\s+)?"
        r"bizonyíték\w*\b",
        "UNSUPPORTED_EVIDENCE_ABSENCE",
    ),
]


# ======================================================================
# V18 REPAIRABLE QUALITY / PROMPT-LEAK EXTENSIONS
# ======================================================================

REPAIRABLE_PROMPT_LEAK_PATTERNS = [
    (
        rf"\blegalább\s+{NUMBER}\s+"
        r"(?:teljes\s+)?mondat\w*\b",
        "PROMPT_LEAK_SENTENCE_COUNT",
    ),
    (
        rf"\b{NUMBER}\s*[-–]\s*{NUMBER}\s+"
        r"(?:teljes\s+)?mondat\w*\b",
        "PROMPT_LEAK_SENTENCE_COUNT",
    ),
]


REPAIRABLE_LANGUAGE_PATTERNS = [
    (
        r"\bérinti\s+a\s+több\s+",
        "LANGUAGE_BAD_QUANTIFIER_ARTICLE",
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
    Find any matching fragment in the generated user text that is not
    already supported literally by the deterministic source draft.

    Important:
    iterate over ALL matches. A supported earlier occurrence must not
    hide a later unsupported occurrence of the same pattern family.
    """

    low_user = normalize(user)
    low_source = normalize(source_draft)

    for pattern, code in patterns:
        for match in re.finditer(
            pattern,
            low_user,
            re.I,
        ):
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
    issue,
    source_draft,
):
    return {
        "classification": SEMANTIC_REJECT,
        "code": issue["code"],
        "fragment": issue["fragment"],
        "source_draft": source_draft,
    }


def repairable_result(
    issue,
    source_draft,
):
    return {
        "classification": REPAIRABLE,
        "code": issue["code"],
        "fragment": issue["fragment"],
        "source_draft": source_draft,
    }


# ======================================================================
# CLASSIFIER
# ======================================================================

def classify_user_quality(
    user,
    scenario,
    facts,
):
    # Frozen V16 remains authoritative.
    result = v16.classify_user_quality(
        user,
        scenario,
        facts,
    )

    if result["classification"] != v16.PASS:
        return result

    source_draft = result["source_draft"]

    # ----------------------------------------------------------
    # New task constraints
    # ----------------------------------------------------------

    issue = find_new_pattern(
        user,
        source_draft,
        UNSUPPORTED_TASK_CONSTRAINT_PATTERNS,
    )

    if issue is not None:
        return semantic_result(
            issue,
            source_draft,
        )

    # ----------------------------------------------------------
    # New evidence/source properties
    # ----------------------------------------------------------

    issue = find_new_pattern(
        user,
        source_draft,
        UNSUPPORTED_SOURCE_QUALIFIER_PATTERNS,
    )

    if issue is not None:
        return semantic_result(
            issue,
            source_draft,
        )

    # ----------------------------------------------------------
    # Evidence-strength drift
    # PARTIAL/UNVERIFIED must not silently become "no evidence".
    # ----------------------------------------------------------

    issue = find_new_pattern(
        user,
        source_draft,
        UNSUPPORTED_EVIDENCE_STRENGTH_PATTERNS,
    )

    if issue is not None:
        return semantic_result(
            issue,
            source_draft,
        )

    # ----------------------------------------------------------
    # Generator/meta instruction leakage
    # Safe to route to one repair attempt.
    # ----------------------------------------------------------

    issue = find_new_pattern(
        user,
        source_draft,
        REPAIRABLE_PROMPT_LEAK_PATTERNS,
    )

    if issue is not None:
        return repairable_result(
            issue,
            source_draft,
        )

    # ----------------------------------------------------------
    # High-precision Hungarian language defects
    # ----------------------------------------------------------

    issue = find_new_pattern(
        user,
        source_draft,
        REPAIRABLE_LANGUAGE_PATTERNS,
    )

    if issue is not None:
        return repairable_result(
            issue,
            source_draft,
        )

    return {
        "classification": PASS,
        "code": "QUALITY_PASS",
        "source_draft": source_draft,
    }
