import re

import generate_manager_policy_dataset_v05_safe_v18_quality_frozen as v18


VERSION = "V05_SAFE_POLICY_V20_QUALITY_EXTENSION"


PASS = v18.PASS
REPAIRABLE = v18.REPAIRABLE
SEMANTIC_REJECT = v18.SEMANTIC_REJECT


# ======================================================================
# V20 SEMANTIC EXTENSIONS
# ======================================================================

UNSUPPORTED_FUTURE_EXECUTION_PATTERNS = [
    (
        r"\b(?:konkrét\s+)?"
        r"(?:vizsgálat|feladat|művelet)\w*\s+"
        r"végrehajtás\w*\s+később\s+"
        r"(?:következ\w*|történ\w*|lesz)\b",
        "UNSUPPORTED_FUTURE_EXECUTION",
    ),
]


NO_TOOL_DOCUMENT_LOOKUP_PATTERNS = [
    (
        r"\bdokumentációból\s+"
        r"(?:ellenőriz\w*|kikeres\w*|megerősít\w*)\b",
        "UNEXPECTED_DOCUMENT_LOOKUP_NO_TOOL",
    ),
    (
        r"\bdokumentációt\s+"
        r"(?:ellenőriz\w*|átvizsgál\w*|lekér\w*)\b",
        "UNEXPECTED_DOCUMENT_LOOKUP_NO_TOOL",
    ),
]


UNSUPPORTED_SCOPE_PATTERNS = [
    (
        r"\bminden\s+tényező\w*\s+"
        r"(?:pontosan\s+)?"
        r"átvizsgál\w*\b",
        "UNSUPPORTED_SCOPE_ALL_FACTORS",
    ),
]


# ======================================================================
# V20 REPAIRABLE META / LANGUAGE EXTENSIONS
# ======================================================================

REPAIRABLE_META_PATTERNS = [
    (
        r"\búj\s+rendszer\w*\s+vagy\s+"
        r"termék\w*\s+említ\w*\b",
        "PROMPT_LEAK_NO_NEW_SYSTEM_PRODUCT",
    ),
    (
        r"\b(?:bármilyen\s+)?új\s+funkció\w*\s+"
        r"vagy\s+állapotszint\w*\s+"
        r"feltételez\w*\b",
        "PROMPT_LEAK_NO_ASSUMED_FUNCTION_STATE",
    ),
]


REPAIRABLE_LANGUAGE_PATTERNS = [
    (
        r"\badatak\b",
        "LANGUAGE_TYPO_DATAK",
    ),
    (
        r"\bcritikus\b",
        "LANGUAGE_TYPO_CRITICAL",
    ),
]


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


def classify_user_quality(
    user,
    scenario,
    facts,
):
    # ----------------------------------------------------------
    # Frozen V18 remains authoritative.
    # ----------------------------------------------------------

    result = v18.classify_user_quality(
        user,
        scenario,
        facts,
    )

    if result["classification"] != v18.PASS:
        return result

    source_draft = result["source_draft"]

    # ----------------------------------------------------------
    # Unsupported future execution/scheduling claim.
    # ----------------------------------------------------------

    issue = v18.find_new_pattern(
        user,
        source_draft,
        UNSUPPORTED_FUTURE_EXECUTION_PATTERNS,
    )

    if issue is not None:
        return semantic_result(
            issue,
            source_draft,
        )

    # ----------------------------------------------------------
    # requires_tool=False must not silently become a
    # documentation lookup / information-retrieval task.
    # ----------------------------------------------------------

    if not facts["requires_tool"]:
        issue = v18.find_new_pattern(
            user,
            source_draft,
            NO_TOOL_DOCUMENT_LOOKUP_PATTERNS,
        )

        if issue is not None:
            return semantic_result(
                issue,
                source_draft,
            )

    # ----------------------------------------------------------
    # Scope broadening.
    # ----------------------------------------------------------

    issue = v18.find_new_pattern(
        user,
        source_draft,
        UNSUPPORTED_SCOPE_PATTERNS,
    )

    if issue is not None:
        return semantic_result(
            issue,
            source_draft,
        )

    # ----------------------------------------------------------
    # Generator/meta instruction leakage.
    # ----------------------------------------------------------

    issue = v18.find_new_pattern(
        user,
        source_draft,
        REPAIRABLE_META_PATTERNS,
    )

    if issue is not None:
        return repairable_result(
            issue,
            source_draft,
        )

    # ----------------------------------------------------------
    # High-precision language defects.
    # ----------------------------------------------------------

    issue = v18.find_new_pattern(
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
