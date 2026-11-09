"""Conservative guard against naming an unseen entity from a definition/memory.

This is an extra admission check, not a substitute for the semantic evidence graph.
"""
from __future__ import annotations

import re


def normalized(text):
    text = re.sub(r"['’]", "", text.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_support(question, sources, answer):
    frontier = "\n".join([question, *sources])
    value = normalized(answer)
    value = re.sub(r"^(the|an|a) ", "", value)
    visible = normalized(frontier)
    if value and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", visible):
        return {"supported": True, "basis": "visible_answer"}
    if value in ("yes", "no", "true", "false"):
        return {"supported": True, "basis": "boolean_requires_semantic_review"}
    if re.fullmatch(r"[\d.,+\-/\s]+", answer.strip()) and re.search(r"\d", answer):
        return {"supported": True, "basis": "numeric_requires_semantic_review"}
    # Surface country-name expansions, not geographical facts such as
    # New South Wales -> Australia or a definition -> an unseen war name.
    aliases = {
        "united states": r"\bU\.S\.(?:A\.)?|\bUSA\b|\bUS\b|\bAmerican\b",
        "united states of america": r"\bU\.S\.(?:A\.)?|\bUSA\b|\bUS\b|\bAmerican\b",
        "united kingdom": r"\bU\.K\.|\bUK\b",
    }
    pattern = aliases.get(value)
    if pattern and re.search(pattern, frontier):
        return {"supported": True, "basis": "surface_country_alias_requires_semantic_review"}
    # Lists, reordered dates, and short explanatory answers may compose already
    # visible terms. Their relationships still require the full evidence review.
    grammar = set("a an the and or of in on at to from for by with is are was were be been it its his her their who which that this as won wins win winner victory answer".split())
    terms = set(value.split()) - grammar
    if terms and terms <= set(visible.split()):
        return {"supported": True, "basis": "visible_terms_require_semantic_review"}
    return {"supported": False, "basis": "unseen_named_answer_requires_retrieval"}
