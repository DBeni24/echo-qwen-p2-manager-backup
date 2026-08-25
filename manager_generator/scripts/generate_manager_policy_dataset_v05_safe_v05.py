import re

import generate_manager_policy_dataset_v05_safe_v04 as v4
import generate_manager_policy_dataset_v05_safe_v03 as v3
import generate_manager_policy_dataset_v05_safe_v01 as base


base.VERSION = "V05_SAFE_POLICY_V05"

_V04_VALIDATE_USER_TEXT = v4.validate_user_text
_V03_SAMPLE_POLICY_FACTS = v3.sample_policy_facts
_V04_BUILD_GENERATION_PROMPT = v4.build_generation_prompt


def sample_policy_facts(rng, scenario):
    facts = _V03_SAMPLE_POLICY_FACTS(rng, scenario)

    # Credential infrastruktúra tényleges ellenőrzése nálunk
    # ne legyen MEDIUM kockázatú.
    if (
        scenario == "secret_handling"
        and facts["operation_type"] == "VALIDATE"
    ):
        facts["impact"] = "HIGH"
        facts["task_complexity"] = "COMPLEX"

    return facts


def build_generation_prompt(scenario, facts):
    prompt = _V04_BUILD_GENERATION_PROMPT(scenario, facts)

    extra = []

    if not facts["requires_tool"] and not facts["requires_live_data"]:
        extra.append(
            """
Mivel ehhez a helyzethez nem szükséges rendszerlekérdezés:
- minden szükséges tény már szerepeljen magában a felhasználói feladatban;
- ne kérje logok, naplók, konfigurációk, adatbázisok, metrikák,
  státuszok vagy más külső rendszeradat megnyitását vagy vizsgálatát;
- a kérés kizárólag a megadott tényekből megoldható legyen.
""".strip()
        )

    if scenario == "secret_handling":
        if facts["secret_exposure_requested"]:
            extra.append(
                """
Ez kifejezetten tiltandó secret-felfedési példa.
Csak <API_KEY> vagy <SECRET> placeholder szerepelhet.
Valódihoz hasonló secretet ne generálj.
""".strip()
            )
        else:
            extra.append(
                """
Ez NEM secret-felfedési kérés.
A hitelesítő adat teljes értékét soha ne kérje.
Ha credentialt kell ellenőrizni, kizárólag annak azonosítóját,
metaadatait, scope-ját, rotációs állapotát, jogosultságait vagy
hozzáférési naplóit vizsgálja, az érték felfedése nélkül.
Ne írj olyan helyzetet, amely már bizonyított credential leaket,
adatkiszivárgást vagy kompromittált kulcsot állít.
""".strip()
            )

    if facts["impact"] in {"LOW", "MEDIUM"}:
        extra.append(
            """
A helyzet ne legyen kritikus incidens.
Ne szerepeljen benne adatvesztés, adatszivárgás, credential compromise,
kritikus szolgáltatáskiesés vagy sürgős production vészhelyzet.
""".strip()
        )

    if extra:
        prompt += "\n\nTovábbi konzisztencia-szabályok:\n" + "\n\n".join(extra)

    return prompt


def contains_any(text, terms):
    low = text.lower()
    return any(term in low for term in terms)


def validate_user_text(user):
    user = _V04_VALIDATE_USER_TEXT(user)

    if v3.CURRENT_CONTEXT is None:
        raise ValueError("NO_POLICY_CONTEXT_V05")

    facts = v3.CURRENT_CONTEXT["facts"]
    scenario = v3.CURRENT_CONTEXT["scenario"]
    low = user.lower()

    # ------------------------------------------------------------
    # 1. requires_tool=False => ne kérjen külső evidence-vizsgálatot.
    # ------------------------------------------------------------

    if not facts["requires_tool"] and not facts["requires_live_data"]:
        inspect_terms = [
            "ellenőrizd",
            "ellenőrizze",
            "vizsgáld",
            "vizsgálja",
            "elemezd",
            "elemezze",
            "nézd meg",
            "tekintsd meg",
            "kérdezd le",
            "lekérdez",
            "keress rá",
            "olvasd be",
        ]

        external_evidence_terms = [
            "napló",
            "log",
            "konfiguráció",
            "config",
            "adatbázis",
            "metrika",
            "státusz",
            "rendszeresemény",
            "monitoring",
            "checkpoint",
            "api",
            "fájl",
        ]

        if (
            contains_any(low, inspect_terms)
            and contains_any(low, external_evidence_terms)
        ):
            raise ValueError("NO_TOOL_EXTERNAL_INSPECTION_REQUEST")

    # ------------------------------------------------------------
    # 2. LOW/MEDIUM fact ne váljon high/critical incidenssé.
    # ------------------------------------------------------------

    if facts["impact"] in {"LOW", "MEDIUM"}:
        high_severity_terms = [
            "adatveszt",
            "adatszivárg",
            "kiszivárg",
            "kompromittált",
            "kompromittálód",
            "jogosulatlan hozzáfér",
            "illetéktelen hozzáfér",
            "biztonsági incidens",
            "kritikus leállás",
            "kritikus kiesés",
            "érzékeny információkhoz hozzáfér",
            "érzékeny információk elérhető",
            "azonnal le kell zárni",
            "vészhelyzet",
        ]

        if contains_any(low, high_severity_terms):
            raise ValueError("IMPACT_ESCALATION_BEYOND_POLICY_FACTS")

    # ------------------------------------------------------------
    # 3. Secret esetben ne lehessen véletlenül secret retrieval.
    # ------------------------------------------------------------

    if scenario == "secret_handling":
        if not facts["secret_exposure_requested"]:
            disclosure_patterns = [
                r"\badd meg\b.{0,80}\b(?:kulcs|token|jelszó|secret)",
                r"\bírd ki\b.{0,80}\b(?:kulcs|token|jelszó|secret)",
                r"\bmutasd meg\b.{0,80}\b(?:kulcs|token|jelszó|secret)",
                r"\bfedd fel\b.{0,80}\b(?:kulcs|token|jelszó|secret)",
                r"\bkeress rá\b.{0,80}\b(?:hitelesítő adat|kulcs érték|token érték)",
                r"\bteljes érték",
            ]

            for pattern in disclosure_patterns:
                if re.search(pattern, low):
                    raise ValueError("UNEXPECTED_SECRET_VALUE_RETRIEVAL")

            # A normál secret-auditban legyen egyértelmű,
            # hogy metaadat/azonosító/policy jellegű vizsgálatról van szó.
            safe_secret_terms = [
                "azonosító",
                "metaadat",
                "rotáció",
                "jogosultság",
                "scope",
                "hozzáférési napló",
                "hozzáférés napló",
                "secret manager",
                "hitelesítő adat kezel",
                "kulcs kezel",
            ]

            if not contains_any(low, safe_secret_terms):
                raise ValueError("SECRET_VALIDATION_SCOPE_AMBIGUOUS")

    return user


base.sample_policy_facts = sample_policy_facts
base.build_generation_prompt = build_generation_prompt
base.validate_user_text = validate_user_text


if __name__ == "__main__":
    base.main()
