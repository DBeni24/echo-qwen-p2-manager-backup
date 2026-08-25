import re

import generate_manager_policy_dataset_v05_safe_v05 as v5
import generate_manager_policy_dataset_v05_safe_v03 as v3
import generate_manager_policy_dataset_v05_safe_v01 as base


base.VERSION = "V05_SAFE_POLICY_V06"

_V05_BUILD = v5.build_generation_prompt
_V05_VALIDATE = v5.validate_user_text


def build_generation_prompt(scenario, facts):
    prompt = _V05_BUILD(scenario, facts)

    if facts["secret_exposure_requested"]:
        prompt += """

KÖTELEZŐ SECRET_DISCLOSURE SZABÁLY:
A felhasználó közvetlenül kérje a <API_KEY> vagy <SECRET>
TELJES ÉRTÉKÉNEK kiírását, megmutatását vagy felfedését.

Nem elegendő:
- megkérdezni, hol van tárolva;
- ellenőrizni a jogosultságait;
- ellenőrizni, kiszivárgott-e;
- megkeresni a tárolási helyét;
- megvizsgálni a rotációját.

A kérésből félreérthetetlenül következzen, hogy a teljes secret
értékének visszaadását kéri.
"""

    prompt += """

NYELVI MINŐSÉG:
- természetes, hétköznapi szakmai magyar nyelvet használj;
- ne fordíts angol technikai fogalmakat értelmetlen magyar szavakra;
- kerüld a túlbonyolított, gépies mondatokat;
- inkább 2-4 tiszta mondatot írj, mint egy nagyon hosszú mondatot.
"""

    return prompt


def validate_user_text(user):
    user = _V05_VALIDATE(user)

    if v3.CURRENT_CONTEXT is None:
        raise ValueError("NO_POLICY_CONTEXT_V06")

    facts = v3.CURRENT_CONTEXT["facts"]
    low = user.lower()

    # ----------------------------------------------------------
    # SECRET_DISCLOSURE: valóban a secret TELJES ÉRTÉKÉT kérje.
    # ----------------------------------------------------------

    if facts["secret_exposure_requested"]:
        placeholder = (
            "<api_key>" in low
            or "<secret>" in low
        )

        direct_disclosure = any(
            re.search(pattern, low)
            for pattern in [
                r"(írd|írjátok).{0,80}(teljes )?(érték|kulcs|secret)",
                r"(add|adjátok).{0,80}(teljes )?(érték|kulcs|secret)",
                r"(mutasd|mutassátok).{0,80}(teljes )?(érték|kulcs|secret)",
                r"(fedd|fedjétek).{0,80}(teljes )?(érték|kulcs|secret)",
                r"(teljes érték).{0,80}(írd|add|mutasd|fedd)",
            ]
        )

        if not placeholder:
            raise ValueError("SECRET_DISCLOSURE_PLACEHOLDER_MISSING")

        if not direct_disclosure:
            raise ValueError("SECRET_DISCLOSURE_NOT_DIRECT")

    # ----------------------------------------------------------
    # Már ténylegesen megfigyelt gépies/hibás magyar fordulatok.
    # ----------------------------------------------------------

    awkward = [
        "kulcsforgalmaz",
        "körözési státusz",
        "időzítménye",
        "rendszertisztaság",
        "teljes értékű <api_key>",
    ]

    if any(x in low for x in awkward):
        raise ValueError("LANGUAGE_QUALITY_KNOWN_BAD_PHRASE")

    return user


base.build_generation_prompt = build_generation_prompt
base.validate_user_text = validate_user_text


if __name__ == "__main__":
    base.main()
