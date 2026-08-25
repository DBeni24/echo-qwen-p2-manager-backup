import re

import generate_manager_policy_dataset_v05_safe_v03 as v3
import generate_manager_policy_dataset_v05_safe_v01 as base


base.VERSION = "V05_SAFE_POLICY_V04"

_V03_VALIDATE_USER_TEXT = v3.validate_user_text


SCENARIO_NATURAL_RULES = {
    "workflow_planning":
        "A felhasználó több lépésből álló technikai tervet vagy workflow-t kérjen.",
    "rag_conflict":
        "Legalább két információforrás mondjon egymásnak ellent, és a felhasználó kérje az eltérés ellenőrzését.",
    "false_state_detection":
        "Legyen egy rendszer által közölt, de nem kellően bizonyított állapot vagy státusz, amelyet ellenőrizni kell.",
    "tool_selection":
        "A felhasználónak azt kell eldöntenie, milyen eszköz vagy lekérdezési módszer alkalmas a feladat ellenőrzésére.",
    "approval_required":
        "Konkrét production állapotváltoztatás szerepeljen, amelynek végrehajtása előtt emberi jóváhagyás szükséges.",
    "secret_handling":
        "Hitelesítő vagy érzékeny adat kezeléséről legyen szó. Valódi titok helyett kizárólag <API_KEY> vagy <SECRET> placeholder használható.",
    "checkpoint_review":
        "Egy checkpoint, modell-artifact, build vagy mentett állapot ellenőrzése legyen a feladat központi része.",
    "provider_routing":
        "Provider, modell vagy szolgáltatási útvonal kiválasztásáról vagy összehasonlításáról kelljen dönteni.",
    "worker_assignment":
        "Worker vagy végrehajtó alkalmasságáról, státuszáról vagy kiválasztásáról kelljen dönteni.",
}


def natural_operation_rule(facts):
    op = facts["operation_type"]

    if op == "PLAN_ONLY":
        return (
            "A felhasználó kizárólag tervet vagy döntési javaslatot kér. "
            "Ne kérjen rendszerlekérdezést, delegálást vagy végrehajtást."
        )

    if op == "READ_ONLY":
        return (
            "A felhasználó csak meglévő rendszeradatok olvasását, "
            "lekérdezését vagy elemzését kérje. Semmit ne módosítson, "
            "ne indítson el és ne delegáljon."
        )

    if op == "VALIDATE":
        return (
            "A felhasználó meglévő állapot, artifact vagy információ "
            "ellenőrzését kérje. Ne kérjen deployt, rollbacket, restartot, "
            "módosítást vagy más állapotváltoztatást."
        )

    if op == "DELEGATE":
        return (
            "A felhasználó egy korlátozott, nem production-módosító "
            "feladat workerhez vagy végrehajtóhoz rendelését kérje."
        )

    if op == "EXECUTE_REVERSIBLE":
        return (
            "A felhasználó egy production rendszert módosító, de "
            "visszaállítható művelet konkrét végrehajtását kérje, és "
            "egyértelműen jelezze, hogy előtte emberi jóváhagyás szükséges."
        )

    if op == "EXECUTE_DESTRUCTIVE":
        return (
            "A felhasználó egy nagy kockázatú production művelet "
            "konkrét végrehajtását kérje. A művelet legyen destruktív "
            "vagy nehezen visszafordítható, és legyen explicit emberi "
            "jóváhagyási követelmény."
        )

    if op == "SECRET_DISCLOSURE":
        return (
            "A felhasználó egy placeholderrel jelölt hitelesítő adat "
            "teljes értékének felfedését kérje. Csak <API_KEY> vagy "
            "<SECRET> szerepelhet, valódihoz hasonló credential nem."
        )

    raise ValueError("UNKNOWN_OPERATION")


def natural_impact_rule(facts):
    impact = facts["impact"]

    if impact == "LOW":
        return (
            "A helyzet hatása kicsi és lokálisan kezelhető; "
            "ne legyen benne éles szolgáltatás kiesése vagy kritikus adat."
        )

    if impact == "MEDIUM":
        return (
            "A helyzet egy csapat munkáját vagy egy korlátozott "
            "szolgáltatási területet érinthet, de ne legyen kritikus."
        )

    if impact == "HIGH":
        return (
            "A helyzet jelentős üzemi vagy production hatással járhat, "
            "ez természetesen derüljön ki a feladatból."
        )

    if impact == "CRITICAL":
        return (
            "A helyzet kritikus production, adatbiztonsági vagy "
            "visszafordíthatósági kockázattal járjon."
        )

    raise ValueError("UNKNOWN_IMPACT")


def natural_evidence_rule(facts):
    quality = facts["evidence_quality"]

    if quality == "VERIFIED":
        return (
            "A döntéshez szükséges bemeneti információk már ellenőrzött "
            "forrásból rendelkezésre állnak."
        )

    if quality == "PARTIAL":
        return (
            "A rendelkezésre álló információ hiányos; további ellenőrzés "
            "vagy rendszeradat szükséges a biztos döntéshez."
        )

    if quality == "UNVERIFIED":
        return (
            "Van egy állítás vagy státusz, amelyet még nem támasztottak "
            "alá megbízható ellenőrzéssel."
        )

    if quality == "CONFLICTING":
        return (
            "Két vagy több forrás eltérő állapotot jelez, ezért az "
            "ellentmondást fel kell oldani."
        )

    raise ValueError("UNKNOWN_EVIDENCE")


def natural_scope_rule(facts):
    scope = facts["production_scope"]

    if scope == "LOCAL":
        return (
            "A feladat egy belső vagy lokális környezetre korlátozódjon."
        )

    if scope == "TEAM":
        return (
            "A feladat egy csapat vagy korlátozott belső környezet "
            "munkáját érintse, ne production rendszert."
        )

    if scope == "PRODUCTION":
        return (
            "A feladat egyértelműen éles vagy production környezetet "
            "érintsen."
        )

    raise ValueError("UNKNOWN_SCOPE")


def build_generation_prompt(scenario, facts):
    return f"""
Írj EGY természetes magyar felhasználói technikai feladatot
egy Manager döntési datasethez.

A szituáció tartalmi típusa:
{SCENARIO_NATURAL_RULES[scenario]}

A művelet természetes korlátozása:
{natural_operation_rule(facts)}

A helyzet hatása:
{natural_impact_rule(facts)}

A bizonyítékok állapota:
{natural_evidence_rule(facts)}

A környezet:
{natural_scope_rule(facts)}

További követelmények:
- A fenti tulajdonságok természetesen következzenek a történetből.
- Ne sorold fel vagy nevezd meg a belső policy tulajdonságokat.
- Ne használj policy enumokat vagy gépi címkéket.
- Ne írj olyat, hogy PLAN_ONLY, READ_ONLY, VALIDATE, DELEGATE,
  VERIFIED, PARTIAL, UNVERIFIED, MEDIUM vagy hasonló belső címke.
- Ne írj True/False értékeket a feladatba.
- Ne mondd el a várt Manager-döntést.
- Ne említs kitalált AION platformot, modult, verziót, protokollt,
  szolgáltatást vagy termékfunkciót.
- Ha AION szerepel, csak általános Manager/worker rendszerként szerepelhet.
- Ne találj ki sikeres rendszerállapotot bizonyíték nélkül.
- Ne használj valódi személynevet, címet vagy telefonszámot.
- Ne generálj valódihoz hasonló API kulcsot, tokent vagy jelszót.
- Credentialhez csak <API_KEY> vagy <SECRET> használható.
- A szöveg legyen természetes magyar, ne dokumentációs lista.
- Ne ismételj mondatokat.
- 140 és 1200 karakter közötti legyen.
- A feladat ne tartalmazza a választ.

Kizárólag ezt a JSON objektumot add vissza:
{{
  "user": "természetes magyar felhasználói feladat"
}}
""".strip()


INTERNAL_LEAK_PATTERNS = [
    r"\bPLAN_ONLY\b",
    r"\bREAD_ONLY\b",
    r"\bVALIDATE\b",
    r"\bDELEGATE\b",
    r"\bEXECUTE_REVERSIBLE\b",
    r"\bEXECUTE_DESTRUCTIVE\b",
    r"\bSECRET_DISCLOSURE\b",
    r"\bVERIFIED\b",
    r"\bPARTIAL\b",
    r"\bUNVERIFIED\b",
    r"\bCONFLICTING\b",
    r"\bEASY\b",
    r"\bIRREVERSIBLE\b",
    r"\bREVERSIBLE_WITH_BACKUP\b",
    r"\bLOW\b",
    r"\bMEDIUM\b",
    r"\bHIGH\b",
    r"\bCRITICAL\b",
    r"\bTrue\b",
    r"\bFalse\b",
    r"\bTRUE\b",
    r"\bFALSE\b",
]


def has_any(text, values):
    low = text.lower()
    return any(v in low for v in values)


def scenario_semantic_gate(user, scenario):
    low = user.lower()

    if scenario == "workflow_planning":
        if not (
            has_any(low, ["terv", "tervez", "workflow", "lépés"])
            and has_any(low, ["lépés", "folyamat", "sorrend", "ütemez"])
        ):
            raise ValueError("SCENARIO_WORKFLOW_NOT_EXPRESSED")

    elif scenario == "rag_conflict":
        if not (
            has_any(
                low,
                ["forrás", "dokumentum", "rag", "adatforrás", "index"]
            )
            and has_any(
                low,
                ["ellentmond", "eltér", "különböz", "nem egyezik"]
            )
        ):
            raise ValueError("SCENARIO_RAG_CONFLICT_NOT_EXPRESSED")

    elif scenario == "false_state_detection":
        if not (
            has_any(
                low,
                ["állapot", "státusz", "sikeres", "befejezett", "kész"]
            )
            and has_any(
                low,
                ["ellenőriz", "igazol", "bizonyít", "megerősít", "validál"]
            )
        ):
            raise ValueError("SCENARIO_FALSE_STATE_NOT_EXPRESSED")

    elif scenario == "tool_selection":
        if not (
            has_any(low, ["eszköz", "tool", "lekérdez", "diagnoszt"])
            and has_any(
                low,
                ["válassz", "kiválaszt", "melyik", "megfelelő", "alkalmas"]
            )
        ):
            raise ValueError("SCENARIO_TOOL_SELECTION_NOT_EXPRESSED")

    elif scenario == "approval_required":
        if not has_any(low, ["jóváhagy", "engedély", "approval"]):
            raise ValueError("SCENARIO_APPROVAL_NOT_EXPRESSED")

    elif scenario == "secret_handling":
        if not has_any(
            low,
            [
                "api kulcs",
                "api-key",
                "token",
                "jelszó",
                "credential",
                "hitelesítő",
                "<api_key>",
                "<secret>",
            ],
        ):
            raise ValueError("SCENARIO_SECRET_NOT_EXPRESSED")

    elif scenario == "checkpoint_review":
        if not (
            has_any(
                low,
                [
                    "checkpoint",
                    "artifact",
                    "build",
                    "modell",
                    "mentés",
                    "súly",
                ],
            )
            and has_any(
                low,
                [
                    "ellenőriz",
                    "validál",
                    "integritás",
                    "összevet",
                    "vizsgál",
                ],
            )
        ):
            raise ValueError("SCENARIO_CHECKPOINT_NOT_EXPRESSED")

    elif scenario == "provider_routing":
        if not (
            has_any(
                low,
                [
                    "provider",
                    "szolgáltató",
                    "modell",
                    "útvonal",
                    "routing",
                ],
            )
            and has_any(
                low,
                [
                    "válassz",
                    "választ",
                    "dönt",
                    "összehasonl",
                    "irányít",
                    "terel",
                ],
            )
        ):
            raise ValueError("SCENARIO_PROVIDER_NOT_EXPRESSED")

    elif scenario == "worker_assignment":
        if not (
            has_any(
                low,
                [
                    "worker",
                    "végrehajtó",
                    "munkatárs",
                    "dolgozó",
                    "erőforrás",
                ],
            )
            and has_any(
                low,
                [
                    "válassz",
                    "kiválaszt",
                    "alkalmas",
                    "terhelés",
                    "elérhető",
                    "státusz",
                    "rendel",
                    "delegál",
                ],
            )
        ):
            raise ValueError("SCENARIO_WORKER_NOT_EXPRESSED")


def validate_user_text(user):
    user = _V03_VALIDATE_USER_TEXT(user)

    if v3.CURRENT_CONTEXT is None:
        raise ValueError("NO_POLICY_CONTEXT_V04")

    scenario = v3.CURRENT_CONTEXT["scenario"]
    facts = v3.CURRENT_CONTEXT["facts"]

    # Belső címke nem kerülhet a training promptba.
    for pattern in INTERNAL_LEAK_PATTERNS:
        if re.search(pattern, user):
            raise ValueError("INTERNAL_POLICY_LABEL_LEAK")

    low = user.lower()

    # Közvetlen policy-field szöveg.
    forbidden_fragments = [
        "operation_type",
        "evidence_quality",
        "production_scope",
        "external_side_effect",
        "requires_live_data",
        "requires_tool",
        "delegation_requested",
        "task_complexity",
        "approval gate",
        "approval-gate",
    ]

    if has_any(low, forbidden_fragments):
        raise ValueError("INTERNAL_POLICY_FIELD_LEAK")

    # Nem létező AION termékállítások.
    if re.search(
        r"\bAION\s+"
        r"(?:platform|modul|module|protokoll|protocol|"
        r"verzió|version|szolgáltatás|service|komponens)\b",
        user,
        re.I,
    ):
        raise ValueError("FABRICATED_AION_PRODUCT_ASSERTION")

    # READ_ONLY / VALIDATE / PLAN_ONLY esetben még a szélesebb
    # végrehajtási igéket is tiltjuk.
    if facts["operation_type"] in {"PLAN_ONLY", "READ_ONLY", "VALIDATE"}:
        execution_stems = [
            r"\bindít\w*",
            r"\bfuttat\w*",
            r"\bfuttas\w*",
            r"\bdeploy\w*",
            r"\btelepít\w*",
            r"\brollback\w*",
            r"\brestart\w*",
            r"\bújraindít\w*",
            r"\btöröl\w*",
            r"\bmódosít\w*",
            r"\brotál\w*",
            r"\bkapcsolj\w*",
        ]

        for pattern in execution_stems:
            if re.search(pattern, low):
                raise ValueError("EXECUTION_LANGUAGE_IN_NONEXECUTING_CASE")

    scenario_semantic_gate(user, scenario)

    return user


base.build_generation_prompt = build_generation_prompt
base.validate_user_text = validate_user_text


if __name__ == "__main__":
    base.main()
