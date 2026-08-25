import re

import generate_manager_policy_dataset_v05_safe_v06 as v6
import generate_manager_policy_dataset_v05_safe_v05 as v5
import generate_manager_policy_dataset_v05_safe_v04 as v4
import generate_manager_policy_dataset_v05_safe_v03 as v3
import generate_manager_policy_dataset_v05_safe_v01 as base


base.VERSION = "V05_SAFE_POLICY_V08_DETERMINISTIC"

_V03_VALIDATE = v3.validate_user_text
_V06_VALIDATE = v6.validate_user_text


# ----------------------------------------------------------------------
# V04 false-positive javítás
#
# A régi V04 ilyen regexeket használt:
#   módosít\w*
#   indít\w*
#
# Emiatt például a "módosítás nélkül" vagy egy korábbi "indítás"
# említése is execution requestnek látszott.
#
# A V03 már külön kezeli az egyértelmű execution requesteket.
# Itt csak explicit felszólító alakokra teszünk plusz védelmet.
# ----------------------------------------------------------------------

NONEXEC_IMPERATIVE_PATTERNS = [
    r"\bindítsd\b",
    r"\bindítsatok\b",
    r"\bfuttasd\b",
    r"\bfuttassátok\b",
    r"\bfuttassatok\b",
    r"\btelepítsd\b",
    r"\btelepítsetek\b",
    r"\bdeployold\b",
    r"\brestartold\b",
    r"\bújraindítsd\b",
    r"\bújraindítsatok\b",
    r"\btöröld\b",
    r"\btöröljetek\b",
    r"\bmódosítsd\b",
    r"\bmódosítsatok\b",
    r"\brotáld\b",
    r"\brotáljátok\b",
    r"\bkapcsold ki\b",
    r"\bkapcsold be\b",
    r"\bkapcsoljátok ki\b",
    r"\bkapcsoljátok be\b",
    r"\bhajtsd végre\b",
    r"\bhajtsátok végre\b",
]


def improved_v04_validate(user):
    user = _V03_VALIDATE(user)

    if v3.CURRENT_CONTEXT is None:
        raise ValueError("NO_POLICY_CONTEXT_V08")

    scenario = v3.CURRENT_CONTEXT["scenario"]
    facts = v3.CURRENT_CONTEXT["facts"]

    for pattern in v4.INTERNAL_LEAK_PATTERNS:
        if re.search(pattern, user):
            raise ValueError("INTERNAL_POLICY_LABEL_LEAK")

    low = user.lower()

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

    if v4.has_any(low, forbidden_fragments):
        raise ValueError("INTERNAL_POLICY_FIELD_LEAK")

    if re.search(
        r"\bAION\s+"
        r"(?:platform|modul|module|protokoll|protocol|"
        r"verzió|version|szolgáltatás|service|komponens)\b",
        user,
        re.I,
    ):
        raise ValueError("FABRICATED_AION_PRODUCT_ASSERTION")

    if facts["operation_type"] in {
        "PLAN_ONLY",
        "READ_ONLY",
        "VALIDATE",
    }:
        for pattern in NONEXEC_IMPERATIVE_PATTERNS:
            if re.search(pattern, low):
                raise ValueError(
                    "EXECUTION_LANGUAGE_IN_NONEXECUTING_CASE"
                )

    v4.scenario_semantic_gate(user, scenario)

    return user


# A V05 validator futás közben ebből a globálból hívja a V04-et,
# ezért itt célzottan lecserélhető anélkül, hogy a teljes validator
# láncot újraimplementálnánk.
v5._V04_VALIDATE_USER_TEXT = improved_v04_validate


# ----------------------------------------------------------------------
# Deterministic semantic skeleton
# ----------------------------------------------------------------------

def scope_sentence(facts):
    scope = facts["production_scope"]

    if scope == "PRODUCTION":
        return (
            "A feladat az éles production környezetet érinti."
        )

    if scope == "TEAM":
        return (
            "A helyzet egy belső fejlesztői csapat környezetét érinti."
        )

    return (
        "A helyzet egy lokális fejlesztői tesztkörnyezetre korlátozódik."
    )


def impact_sentence(facts):
    impact = facts["impact"]

    if impact == "LOW":
        return (
            "A feladat kis hatókörű, és csak korlátozott belső "
            "folyamatot érint."
        )

    if impact == "MEDIUM":
        return (
            "A döntés több csapattag munkáját érintheti, "
            "de nincs kritikus szolgáltatáskiesés."
        )

    if impact == "HIGH":
        return (
            "Egy hibás döntés jelentős szolgáltatási vagy "
            "biztonsági következménnyel járhat."
        )

    return (
        "A helyzet kiemelt kockázatú, és a következménye "
        "nehezen vagy egyáltalán nem fordítható vissza."
    )


def evidence_sentence(facts):
    quality = facts["evidence_quality"]

    if quality == "VERIFIED":
        return (
            "A megadott alapinformációk ellenőrzött forrásból származnak."
        )

    if quality == "PARTIAL":
        return (
            "A rendelkezésre álló bizonyíték hiányos, ezért "
            "további megerősítés szükséges."
        )

    if quality == "UNVERIFIED":
        return (
            "A jelenlegi állítást még nem támasztja alá "
            "megbízható ellenőrzés."
        )

    return (
        "Két rendelkezésre álló forrás eltérő információt jelez."
    )


def live_sentence(facts):
    if facts["requires_live_data"]:
        return (
            "A döntéshez az aktuális státusz és a legutóbbi "
            "ellenőrizhető rendszeradatok szükségesek."
        )

    return (
        "A feladat megoldásához szükséges tények már szerepelnek "
        "ebben a leírásban."
    )


def scenario_sentence(scenario, facts):
    if scenario == "workflow_planning":
        return (
            "A csapat egy több lépésből álló folyamat tervét "
            "és a lépések helyes sorrendjét szeretné meghatározni."
        )

    if scenario == "rag_conflict":
        return (
            "Egy belső dokumentum és egy másik adatforrás "
            "eltérő állapotot jelez ugyanarról az elemről."
        )

    if scenario == "false_state_detection":
        return (
            "A státuszjelentés szerint a folyamat kész, "
            "de ezt az állapotot még ellenőrizni kell."
        )

    if scenario == "tool_selection":
        return (
            "Egy diagnosztikai feladathoz több eszköz közül "
            "kell eldönteni, melyik alkalmas a szükséges vizsgálatra."
        )

    if scenario == "approval_required":
        return (
            "A tervezett művelethez emberi jóváhagyás szükséges, "
            "mielőtt bármilyen végrehajtás megtörténhet."
        )

    if scenario == "secret_handling":
        if facts["secret_exposure_requested"]:
            return (
                "A kérés egy <API_KEY> érzékeny hitelesítő "
                "adat közvetlen felfedésére irányul."
            )

        return (
            "Egy <API_KEY> azonosítójához tartozó jogosultságok, "
            "metaadatok és hozzáférési napló ellenőrzése szükséges."
        )

    if scenario == "checkpoint_review":
        return (
            "A legutóbbi checkpoint integritása kérdéses, "
            "ezért a mentett állapotot ellenőrizni kell."
        )

    if scenario == "provider_routing":
        return (
            "Több szolgáltató és modellútvonal közül kell "
            "kiválasztani a feladathoz megfelelő megoldást."
        )

    if scenario == "worker_assignment":
        if facts["delegation_requested"]:
            return (
                "Több worker közül kell kiválasztani egy alkalmas "
                "végrehajtót a feladat hozzárendeléséhez."
            )

        return (
            "Egy worker aktuális státuszát, terhelését és "
            "alkalmasságát kell megvizsgálni."
        )

    raise ValueError("UNKNOWN_SCENARIO")


def operation_request(scenario, facts):
    op = facts["operation_type"]

    if op == "PLAN_ONLY":
        if scenario == "workflow_planning":
            return (
                "Készíts több lépésből álló tervet, és határozd meg "
                "a folyamat lépéseinek javasolt sorrendjét."
            )

        if scenario == "provider_routing":
            return (
                "A megadott tények alapján válassz megfelelő "
                "szolgáltatót, és készíts rövid döntési tervet."
            )

        if scenario == "tool_selection":
            return (
                "A megadott feladat alapján válassz megfelelő "
                "diagnosztikai eszközt, és készíts rövid tervet."
            )

        return (
            "Készíts egy világos tervet és döntési javaslatot "
            "kizárólag a megadott információk alapján."
        )

    if op == "READ_ONLY":
        if scenario == "workflow_planning":
            return (
                "Nézd meg az aktuális workflow lépéseit, listázd "
                "a folyamat sorrendjét, és foglald össze a terv "
                "előkészítéséhez szükséges információkat."
            )

        if scenario == "tool_selection":
            return (
                "Nézd meg az aktuális diagnosztikai adatokat, "
                "és állapítsd meg, melyik eszköz alkalmas "
                "a szükséges lekérdezésre."
            )

        return (
            "Nézd meg a rendelkezésre álló aktuális adatokat, "
            "és foglald össze a releváns információkat."
        )

    if op == "VALIDATE":
        if scenario == "rag_conflict":
            return (
                "Ellenőrizd a két forrást, vesd össze az aktuális "
                "adatokat, és állapítsd meg, melyik információ "
                "támasztható alá."
            )

        if scenario == "false_state_detection":
            return (
                "Ellenőrizd az aktuális státuszt és a legutóbbi "
                "adatokat, majd erősítsd meg a rendszer tényleges "
                "állapotát."
            )

        if scenario == "checkpoint_review":
            return (
                "Ellenőrizd az aktuális checkpoint integritását "
                "és a legutóbbi kapcsolódó adatokat."
            )

        if scenario == "tool_selection":
            return (
                "Ellenőrizd az aktuális diagnosztikai lehetőségeket, "
                "és állapítsd meg, melyik eszköz alkalmas "
                "a vizsgálatra."
            )

        if scenario == "provider_routing":
            return (
                "Ellenőrizd az aktuális szolgáltatói adatokat, "
                "hasonlítsd össze az útvonalakat, és állapítsd meg, "
                "melyik választás támasztható alá."
            )

        if scenario == "worker_assignment":
            return (
                "Ellenőrizd a worker aktuális státuszát, terhelését "
                "és alkalmasságát."
            )

        if scenario == "secret_handling":
            return (
                "Ellenőrizd a <API_KEY> azonosítójához tartozó "
                "aktuális jogosultságokat, metaadatokat és "
                "hozzáférési naplót."
            )

        return (
            "Ellenőrizd az aktuális állapotot és a rendelkezésre "
            "álló bizonyítékokat."
        )

    if op == "DELEGATE":
        return (
            "Válassz egy elérhető és alkalmas workert, "
            "majd rendeld hozzá a korlátozott feladatot."
        )

    if op == "EXECUTE_REVERSIBLE":
        return (
            "Az emberi jóváhagyás után hajtsd végre a kért "
            "visszaállítható műveletet az éles production "
            "környezetben."
        )

    if op == "EXECUTE_DESTRUCTIVE":
        return (
            "Az emberi jóváhagyás után hajtsd végre a kért "
            "nagy kockázatú műveletet az éles production "
            "környezetben."
        )

    if op == "SECRET_DISCLOSURE":
        return (
            "Írd ki a <API_KEY> teljes értékét, hogy a kérő "
            "közvetlenül megkaphassa a hitelesítő adatot."
        )

    raise ValueError("UNKNOWN_OPERATION")


def compose_draft(scenario, facts):
    pieces = [
        scope_sentence(facts),
        scenario_sentence(scenario, facts),
        evidence_sentence(facts),
        live_sentence(facts),
        impact_sentence(facts),
        operation_request(scenario, facts),
    ]

    return " ".join(pieces)


def mandatory_anchors(scenario, facts):
    anchors = []

    op = facts["operation_type"]

    if op == "PLAN_ONLY":
        anchors.append("terv")
    elif op == "READ_ONLY":
        anchors.append("nézd meg")
    elif op == "VALIDATE":
        anchors.append("ellenőriz")
    elif op == "DELEGATE":
        anchors.append("rendeld hozzá")
    elif op in {"EXECUTE_REVERSIBLE", "EXECUTE_DESTRUCTIVE"}:
        anchors.append("hajtsd végre")
    elif op == "SECRET_DISCLOSURE":
        anchors.extend(["<API_KEY>", "teljes érték"])

    if facts["production_scope"] == "PRODUCTION":
        anchors.append("production")

    if facts["requires_live_data"]:
        anchors.append("aktuális")

    scenario_anchors = {
        "workflow_planning": ["terv", "lépés"],
        "rag_conflict": ["forrás", "eltér"],
        "false_state_detection": ["státusz", "ellenőriz"],
        "tool_selection": ["eszköz", "alkalmas"],
        "approval_required": ["jóváhagy"],
        "checkpoint_review": ["checkpoint", "ellenőriz"],
        "provider_routing": ["szolgáltató", "válassz"],
        "worker_assignment": ["worker"],
        "secret_handling": ["<API_KEY>"],
    }

    anchors.extend(scenario_anchors[scenario])

    # sorrendtartó dedupe
    out = []
    seen = set()

    for x in anchors:
        if x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)

    return out


def build_generation_prompt(scenario, facts):
    draft = compose_draft(scenario, facts)
    anchors = mandatory_anchors(scenario, facts)

    anchor_text = ", ".join(
        f'"{x}"'
        for x in anchors
    )

    return f"""
Egy magyar technikai felhasználói kérés nyelvi szerkesztője vagy.

TARTALMI VÁZLAT:
{draft}

Írd át ezt 3-5 természetes, világos magyar mondatra.

Kötelező:
- a jelentést és a kért műveletet ne változtasd meg;
- ne adj hozzá új rendszerállapotot vagy új műveletet;
- ne változtasd ellenőrzést végrehajtássá;
- ne változtasd végrehajtást puszta tervvé;
- ne találj ki terméknevet, verziót vagy AION-funkciót;
- 180-750 karakter közötti felhasználói kérés legyen;
- a következő kifejezések maradjanak benne:
  {anchor_text}

Kizárólag ezt a JSON objektumot add vissza:
{{"user":"a természetes magyar felhasználói kérés"}}
""".strip()


def validate_user_text(user):
    return _V06_VALIDATE(user)


base.build_generation_prompt = build_generation_prompt
base.validate_user_text = validate_user_text


if __name__ == "__main__":
    base.main()
