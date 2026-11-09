#!/usr/bin/env python3
"""Shared, deliberately conservative metadata rules for Stage 1 filtering."""

from __future__ import annotations

import re


PERSON_RE = re.compile(
    r"\b(?:person|people|man|men|woman|women|boy|girl|child|children|baby|"
    r"gentleman|lady|bride|groom|worker|chef|player|rider|skier|surfer|"
    r"doctor|nurse|soldier|athlete|dancer|performer)\b",
    re.IGNORECASE,
)
ACCESSORY_RE = re.compile(
    r"\b(?:glasses|sunglasses|goggles|hat|cap|helmet|necklace|earring|earrings|"
    r"bracelet|watch|scarf|tie|bowtie|backpack|handbag|purse|ring|rings)\b",
    re.IGNORECASE,
)
CLOTHING_RE = re.compile(
    r"\b(?:clothes|clothing|shirt|t-shirt|tee|blouse|sweater|hoodie|jacket|coat|"
    r"sweatshirt|pullover|cardigan|button-down|button down|tank top|jumpsuit|"
    r"dress|skirt|pants|trousers|shorts|jeans|shoe|shoes|boot|boots|sneaker|"
    r"sneakers|sock|socks|underwear|bra|bikini|swimsuit|uniform|vest|apron|"
    r"garment|outfit|top|jersey|gown)\b",
    re.IGNORECASE,
)
APPEARANCE_RE = re.compile(
    r"\b(?:face|identity|facial|hair|hairstyle|haircut|beard|mustache|moustache|"
    r"eyebrow|eyebrows|age|skin tone)\b",
    re.IGNORECASE,
)
BACKGROUND_RE = re.compile(
    r"\b(?:background|backdrop|sky|landscape|scene|setting|environment)\b",
    re.IGNORECASE,
)
ROBOT_CG_RE = re.compile(
    r"\b(?:robot|robotic|android|humanoid|cyborg|cartoon|animated|animation|"
    r"puppet|3d|cgi|game character|video game character|low-poly|low poly)\b",
    re.IGNORECASE,
)
HOLDING_RE = re.compile(
    r"\b(?:hold|holds|holding|held|grip|grips|gripping|carry|carries|carrying|"
    r"wield|wields|wielding)\b",
    re.IGNORECASE,
)
WEARABLE_RE = re.compile(
    r"\b(?:wear|wears|wearing|worn|ring|watch|glove|gloves|bracelet|wrist)\b",
    re.IGNORECASE,
)
RECO_HANDHELD_OBJECT_ONLY_RE = re.compile(
    r"^remove (?:the |a |an )?(?:guitar|open book|basket|glowing object|black shoe|"
    r"white shoes|white rabbit|large dark rectangular object)\b",
    re.IGNORECASE,
)
STYLE_RE = re.compile(
    r"\b(?:style transfer|stylize|stylise|pencil sketch|oil painting|watercolor|"
    r"watercolour|cel-shading|anime style|cartoon style)\b",
    re.IGNORECASE,
)
CAMERA_RE = re.compile(
    r"\b(?:zoom in|zoom out|pan left|pan right|tilt up|tilt down|camera movement|"
    r"camera angle|camera motion|dolly|truck left|truck right)\b",
    re.IGNORECASE,
)
EDIT_VERB_RE = re.compile(r"\b(?:add|remove|replace|change|alter|modify)\b", re.IGNORECASE)
TARGET_TAIL_RE = re.compile(
    r"\b(?:with|from|to|near|next to|beside|held by|worn by|wearing|located|"
    r"standing|sitting|in front of|behind|in|on)\b",
    re.IGNORECASE,
)
POSSESSIVE_PERSON_RE = re.compile(
    r"\b(?:person|people|man|men|woman|women|boy|girl|child|gentleman|lady|worker|player)'s\b",
    re.IGNORECASE,
)
MULTI_STEP_RE = re.compile(
    r"\b(?:add|remove|replace|change|alter|modify)\b.+?\b(?:and then|then|also)\b.+?"
    r"\b(?:add|remove|replace|change|alter|modify)\b",
    re.IGNORECASE,
)
MALFORMED_PERSON_IS_WEARING_RE = re.compile(
    r"^\s*(?:add|remove)\s+(?:the\s+)?"
    r"(?:person|man|woman|boy|girl|child|gentleman|lady)\b"
    r"[^.!?]{0,180}\b(?:is|are)\s+wearing\b(?P<worn>[^.!?]{1,180})",
    re.IGNORECASE,
)


def _first_match(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return match.start() if match else None


def classify_target_scope(instruction: str) -> str:
    """Classify the first edited entity, avoiding garment words in person descriptions."""
    text = instruction.strip()
    # Some GOKU captions use the ungrammatical template "Remove the woman is
    # wearing a ... shirt". In that template the garment, not the person, is
    # the edited entity. Keep the override narrow so a normal instruction such
    # as "Remove a woman wearing a shirt" remains a person edit.
    malformed_wear = MALFORMED_PERSON_IS_WEARING_RE.search(text)
    if malformed_wear:
        worn = malformed_wear.group("worn")
        worn_candidates = {
            "accessory": _first_match(ACCESSORY_RE, worn),
            "clothes": _first_match(CLOTHING_RE, worn),
        }
        present_worn = [
            (position, scope)
            for scope, position in worn_candidates.items()
            if position is not None
        ]
        if present_worn:
            return min(present_worn)[1]
    verb = EDIT_VERB_RE.search(text)
    if verb:
        text = text[verb.end() :].strip(" :,-")
    tail = TARGET_TAIL_RE.search(text)
    target_phrase = text[: tail.start()] if tail else text
    candidates = {
        "person": _first_match(PERSON_RE, target_phrase),
        "accessory": _first_match(ACCESSORY_RE, target_phrase),
        "clothes": _first_match(CLOTHING_RE, target_phrase),
        "appearance": _first_match(APPEARANCE_RE, target_phrase),
        "background": _first_match(BACKGROUND_RE, target_phrase),
    }
    if POSSESSIVE_PERSON_RE.search(target_phrase):
        candidates["person"] = None
    present = [(position, scope) for scope, position in candidates.items() if position is not None]
    if not present:
        return "object"
    return min(present)[1]


def metadata_rejection_reasons(dataset: str, operation: str, instruction: str) -> list[str]:
    reasons: list[str] = []
    text = instruction.strip()
    if not text:
        return ["empty_instruction"]
    scope = classify_target_scope(text)
    if dataset in {"insave", "goku"} and operation in {"add", "remove"} and scope == "clothes":
        reasons.append("excluded_garment_add_remove")
    if STYLE_RE.search(text):
        reasons.append("excluded_style")
    if CAMERA_RE.search(text):
        reasons.append("excluded_camera")
    if MULTI_STEP_RE.search(text):
        reasons.append("excluded_multi_step")
    return reasons


def is_robot_cg(instruction: str) -> bool:
    return bool(ROBOT_CG_RE.search(instruction))


def is_handheld_object_remove(branch: str, target_scope: str, instruction: str) -> bool:
    """Match only removal of the held object, never removal of its holder."""
    if not HOLDING_RE.search(instruction) or WEARABLE_RE.search(instruction):
        return False
    if branch == "goku_remove":
        return target_scope == "object"
    if branch == "reco_remove":
        return bool(RECO_HANDHELD_OBJECT_ONLY_RE.search(instruction))
    return False
