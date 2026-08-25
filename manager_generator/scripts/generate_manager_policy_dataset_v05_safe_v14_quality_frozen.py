import re

import generate_manager_policy_dataset_v05_safe_v03 as v3
import generate_manager_policy_dataset_v05_safe_v12_frozen as v12
import generate_manager_policy_dataset_v05_safe_v13_frozen as v13


VERSION = "V05_SAFE_POLICY_V14_QUALITY_CLASSIFIER"


# ======================================================================
# RESULT CLASSES
# ======================================================================

PASS = "PASS"
REPAIRABLE = "REPAIRABLE"
SEMANTIC_REJECT = "SEMANTIC_REJECT"


# ======================================================================
# LANGUAGE QUALITY
#
# High-precision patterns only.
# These are deliberately NOT semantic failures.
# ======================================================================

REPAIRABLE_LANGUAGE_PATTERNS = [
    (
        r"\bintegrum\w*\b",
        "LANGUAGE_INTEGRUM_GARBAGE",
    ),
    (
        r"\brendszerszemélyzet\w*\b",
        "LANGUAGE_SYSTEM_PERSONNEL_GARBAGE",
    ),
    (
        r"\btöbb\s+tagos\s+munkaerő\b",
        "LANGUAGE_WORKFORCE_GARBAGE",
    ),
    (
        r"\bsegítségetekhez\b",
        "LANGUAGE_BAD_CASE_SUFFIX",
    ),
    (
        r"\bjelenlegi\s+aktuális\b",
        "LANGUAGE_REDUNDANT_CURRENT",
    ),
    (
        r"\bkapcsolatban\s+lévő\s+adat\w*\b",
        "LANGUAGE_AWKWARD_RELATED_DATA",
    ),
    (
        r"\bjóváhagyás\s+elvégzés\w*\b",
        "LANGUAGE_AWKWARD_APPROVAL",
    ),
    (
        r"<api_key>\s+kapcsolat\w*\b",
        "LANGUAGE_AWKWARD_APIKEY_RELATION",
    ),
    (
        r"\bworker-ről\b",
        "LANGUAGE_AWKWARD_WORKER_SUFFIX",
    ),
]


# ======================================================================
# UNSUPPORTED FACT / STATE CLAIMS
#
# These are intentionally conservative:
#
# if the deterministic source did not contain such a claim, the rewrite
# must not invent it.
#
# We classify these as SEMANTIC_REJECT, not REPAIRABLE.
# ======================================================================

UNSUPPORTED_FACT_PATTERNS = [
    (
        r"\béppen\s+most\b",
        "UNSUPPORTED_TEMPORAL_CLAIM_NOW",
    ),
    (
        r"\bmostanában\b",
        "UNSUPPORTED_TEMPORAL_CLAIM_RECENTLY",
    ),
    (
        r"\baz\s+elmúlt\b.{0,50}"
        r"\b(?:nap|napok|héten|hét|hetek|hónap|hónapok)\w*\b",
        "UNSUPPORTED_TEMPORAL_HISTORY",
    ),
    (
        r"\bmúlt\s+(?:héten|hónapban|napokban)\b",
        "UNSUPPORTED_TEMPORAL_HISTORY",
    ),
    (
        r"\btegnap\b",
        "UNSUPPORTED_TEMPORAL_CLAIM_YESTERDAY",
    ),
    (
        r"\ba\s+rendszer\s+(?:nem\s+)?érte\s+el\b",
        "UNSUPPORTED_SYSTEM_OUTCOME",
    ),
    (
        r"\büzemeltetés\s+folytonossága\s+"
        r"(?:biztosított|garantált)\b",
        "UNSUPPORTED_CONTINUITY_GUARANTEE",
    ),
    (
        r"\b(?:szolgáltatás|rendszer)\w*.{0,35}"
        r"\b(?:leállt|meghibásodott|összeomlott)\b",
        "UNSUPPORTED_SYSTEM_FAILURE",
    ),
    (
        r"\b(?:hiba|incidens|probléma)\w*.{0,35}"
        r"\b(?:történt|jelentkezett|lépett\s+fel)\b",
        "UNSUPPORTED_INCIDENT_ASSERTION",
    ),
    (
        r"\b(?:hibamentes|garantáltan\s+működő|"
        r"biztosan\s+működő)\b",
        "UNSUPPORTED_POSITIVE_STATE_ASSERTION",
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


def compose_source_draft(
    scenario,
    facts,
):
    # V10 frozen patches the V08 draft components.
    return (
        v12.v10.v9.v8.compose_draft(
            scenario,
            facts,
        )
    )


def active_semantic_validate(
    user,
    scenario,
    facts,
):
    """
    Execute the exact current frozen semantic validator chain.
    """

    v3.CURRENT_CONTEXT = {
        "scenario": scenario,
        "facts": facts,
    }

    return (
        v12.v10.v9.validate_user_text(
            user
        )
    )


def detect_unsupported_fact(
    user,
    source_draft,
):
    """
    Only flag a pattern when it appears in the rewrite but NOT
    in the deterministic source draft.

    This avoids treating an explicitly supplied source fact as
    hallucinated.
    """

    low_user = normalize(user)
    low_source = normalize(
        source_draft
    )

    for pattern, code in UNSUPPORTED_FACT_PATTERNS:
        matches = list(
            re.finditer(
                pattern,
                low_user,
                re.I,
            )
        )

        for match in matches:
            fragment = normalize(
                match.group(0)
            )

            if fragment not in low_source:
                return {
                    "code": code,
                    "fragment": match.group(0),
                }

    return None


def detect_language_issue(user):
    low = normalize(user)

    for pattern, code in REPAIRABLE_LANGUAGE_PATTERNS:
        match = re.search(
            pattern,
            low,
            re.I,
        )

        if match:
            return {
                "code": code,
                "fragment": match.group(0),
            }

    return None


# ======================================================================
# CLASSIFIER
# ======================================================================

def classify_user_quality(
    user,
    scenario,
    facts,
):
    """
    Priority:

    1. Existing semantic validator
    2. Unsupported/new facts
    3. Language quality
    4. PASS
    """

    source_draft = compose_source_draft(
        scenario,
        facts,
    )

    # ----------------------------------------------------------
    # 1. Existing frozen semantic validator
    # ----------------------------------------------------------

    try:
        active_semantic_validate(
            user,
            scenario,
            facts,
        )

    except Exception as exc:
        code = str(exc)

        # Existing language-quality failures are repairable.
        if code.startswith(
            "LANGUAGE_QUALITY_"
        ):
            return {
                "classification": REPAIRABLE,
                "code": code,
                "source_draft": source_draft,
            }

        return {
            "classification": SEMANTIC_REJECT,
            "code": code,
            "source_draft": source_draft,
        }

    # ----------------------------------------------------------
    # 2. Unsupported facts / state claims
    # ----------------------------------------------------------

    unsupported = detect_unsupported_fact(
        user,
        source_draft,
    )

    if unsupported is not None:
        return {
            "classification": SEMANTIC_REJECT,
            "code": unsupported["code"],
            "fragment": unsupported["fragment"],
            "source_draft": source_draft,
        }

    # ----------------------------------------------------------
    # 3. Hungarian language quality
    # ----------------------------------------------------------

    language = detect_language_issue(
        user
    )

    if language is not None:
        return {
            "classification": REPAIRABLE,
            "code": language["code"],
            "fragment": language["fragment"],
            "source_draft": source_draft,
        }

    # ----------------------------------------------------------
    # 4. PASS
    # ----------------------------------------------------------

    return {
        "classification": PASS,
        "code": "QUALITY_PASS",
        "source_draft": source_draft,
    }
