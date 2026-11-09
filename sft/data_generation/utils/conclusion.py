"""Detect an already completed answer separately from evidence sufficiency."""
from __future__ import annotations

import json
import re
from .logic import boolean_hint, check_short_circuit


ASSERTED_ATOMS = """
Extract what the ASSISTANT HAS ASSERTED, not what is factually proven. Source 1
is its CURRENT reasoning, not a retrieved document. The question only identifies
the two subjects and the requested predicate. Never treat the question as facts.
For each subject, has the assistant definitely claimed the predicate true/false?
A definite 'A is not economic-based' or 'A is about violence, not economics'
claims false for 'economic-based documentaries', even if the next sentence says
it still needs to check B. Do not require evidence about B to extract A's claim.
Do not evaluate whether the claim has factual evidence or a final answer message.
Plans ('verify whether A is P'), tentative hypotheses, conditionals, quotations
of somebody else's guess, and retracted statements are null, not assertions.
'Not definitively P', 'not necessarily P', 'cannot confirm P', and 'no evidence
for P' express uncertainty, not a definite claim that P is false. In particular,
'sometimes two species, or four or five' leaves 'just two species' unresolved.
Return AND for both/and, OR for either/or, NONE for a relational comparison.
Copy the predicate EXACTLY from the question, including negation and numbers.
Every non-null value needs a short contiguous exact quote from source 1 that
states this subject's claim, not the plan to look it up. No source 0 evidence.
Return JSON {"operator":"AND"|"OR"|"NONE","predicate":string,
"claims":[{"entity":string,"truth":true|false|null,
"evidence":[{"source":1,"quote":string}]}]}.
INPUT:\n"""

VERIFY_ASSERTION = """
Does this quoted assistant reasoning DEFINITELY ASSERT this ONE entity/predicate
truth value? Check the commitment, NOT whether it is true in the real world or
supported by retrieval. 'A is about violence, not economics' asserts false for
economic-based; a plan to verify, a conditional or a tentative guess does not.
Ignore plans to check another subject: they cannot undo a definite assertion
about this one. Do not demand a final answer message or external citations.
Negating certainty ('not definitively P', 'not necessarily P', 'cannot confirm P')
does NOT assert not-P. A disjunction allowing both P and not-P stays unknown.
Return JSON {"supported":true|false,"reason":string}.
INPUT:\n"""

PROMPT = """
Inspect the assistant's reasoning BEFORE a tool call. Decide whether it has
ALREADY ASSERTED the COMPLETE answer, or explicitly acknowledged that all facts
needed for the answer are known, but is still planning to search. This is about
what the assistant has concluded, NOT whether its conclusion has good evidence.
Do not excuse a completed answer as needing citation, verification, confidence,
or an exact quote. If the conclusion uses memory or an unsupported inference, it
is STILL a completed conclusion; the data must be quarantined rather than teach
searching after stating the answer. No gold or future evidence is available.

Examples of complete conclusions: 'The city must be Z after excluding Y';
'Z matches the platform described; I will verify its headquarters'; 'the date is
already shown, but I will confirm'; 'the required facts are available' even when
the short answer isn't repeated. Presuppositions count: when asked what P and Q
have in common, calling them 'the two film directors' already supplies a shared
profession. For an exhaustive choice, excluding all other candidates completes
the answer under the question's premise; do not demand reconfirming the survivor.

Do NOT flag a partial bridge or missing branch: 'I know the author; now find the
author's birthplace', or 'P was born in 1900; I need Q's date to compare ages'.
A search plan, question restatement, or tentative hypothesis is incomplete.
Identify the requested unknown. A definite identification such as 'The actor is
X' when asked 'What actor stars in Film B and played Role A?' already ASSERTS
the answer, even if the next sentence says it will verify Film B. An unverified
descriptive constraint does not undo that assertion. Do not require a separate
final-response message to recognize a conclusion in reasoning. In contrast,
'X is in Film B; I still need who played Role A' only states a partial relation,
and 'The actor may be X' is tentative. If the question asks for the actor's birth
year, 'The actor is X; now find the birth year' remains a partial bridge.
For AND No one counterexample suffices; for
OR Yes one example suffices. Do not invent a conclusion the reasoning never made.
Lack of certainty is not a negative premise: 'not definitively just two' or
'sometimes two, or four or five' does not assert that there are not two species.

Return ONLY JSON {"state":"complete"|"incomplete"|"uncertain", "reason":string,
"answer":string|null,"quote_turn":integer|null,"quote":string|null}.
For complete copy a contiguous exact quote from the reasoning history that states
the answer or acknowledges its completion. The answer itself may be null when
the reasoning only acknowledges that the full answer has already been found.
INPUT:\n"""


def conclusion_payload(question, steps, current_reasoning):
    return {"question": question, "reasoning_history": [
        {"turn": index, "text": step["think"]} for index, step in enumerate(steps, 1)
    ] + [{"turn": len(steps) + 1, "text": current_reasoning}]}


def explicit_actor_conclusion(payload):
    """Recognize a narrow, unambiguous assignment to the requested identity.

    The model sometimes demands another constraint even after 'The actor is X'.
    Abstain on multiple unknowns, bridge actors, tentative wording or retraction.
    """
    question = payload["question"]
    if (len(re.findall(r"\b(?:what|which|who|whom|when|where|how)\b", question, re.I)) != 1
            or len(re.findall(r"\bactor\b", question, re.I)) != 1
            or not re.search(r"\b(?:what|which)\s+(?:[A-Za-z]+\s+)?actor\b", question, re.I)):
        return None
    current = payload["reasoning_history"][-1]
    text = current["text"]
    match = re.search(r"(?:^|(?<=[.!?])\s+)The actor is ([A-Z][A-Za-z'-]*(?: [A-Z][A-Za-z'-]*){1,5})\.(?=\s|$)", text)
    if not match or match.group(1).lower() in question.lower():
        return None
    if (match.group(1).split()[0].lower() in ("not", "probably", "perhaps", "maybe")
            or re.search(r"\b(?:wrong|mistaken|retract\w*|withdraw\w*)\b", text[match.end():], re.I)
            or re.search(r"\b(?:different|another) actor\b", text[match.end():], re.I)):
        return None
    return {"state": "complete", "answer": match.group(1), "quote_turn": current["turn"],
            "quote": match.group(0).strip(), "rule": "explicit_requested_actor_identity",
            "reason": "The sole requested actor identity is definitely asserted; checking another descriptive constraint does not retract it."}


def asserted_boolean_conclusion(llm, payload, judge):
    if not boolean_hint(payload["question"]):
        return None
    current = payload["reasoning_history"][-1]
    if re.search(r"\b(?:wrong|mistaken|retract\w*|withdraw\w*)\b", current["text"], re.I):
        return None
    # This evidence realm contains reasoning only. Its proof establishes a
    # commitment, never factual support; readiness must still verify real sources.
    asserted = check_short_circuit(llm, {"question": payload["question"],
        "sources": [{"source": 1, "text": current["text"]}]}, judge,
        atom_prompt=ASSERTED_ATOMS, verification_prompt=VERIFY_ASSERTION)
    if asserted["status"] != "stop":
        return None
    evidence = asserted["decisive_claim"]["evidence"]
    if any(item["source"] != 1 for item in evidence):
        return None
    # The semantic verifier has repeatedly promoted a negation of certainty to
    # a definite false premise. Abstain on that quoted wording; the ordinary
    # full-history check can still recognize a different completed answer.
    epistemic = (r"\bnot\s+(?:definit\w*|necessar\w*|certain\w*|clear\w*|"
                 r"prov\w*|establish\w*|confirm\w*|known|sure)\b|"
                 r"\b(?:cannot|can't)\s+(?:confirm|determine|establish|prove|tell)\b|"
                 r"\bno evidence\b|\b(?:unclear|unknown|uncertain)\b")
    if any(re.search(epistemic, item["quote"], re.I) for item in evidence):
        return None
    return {"state": "complete", "answer": asserted["answer"],
            "quote_turn": current["turn"], "quote": evidence[0]["quote"],
            "reason": "A verified assertion already decides the boolean question; its factual support is checked separately.",
            "rule": "asserted_boolean_short_circuit", "assertion_review": asserted}


def check_conclusion(llm, payload, judge):
    explicit = explicit_actor_conclusion(payload)
    if explicit is not None:
        return explicit
    asserted = asserted_boolean_conclusion(llm, payload, judge)
    if asserted is not None:
        return asserted
    result = judge(llm, PROMPT + json.dumps(payload, ensure_ascii=False))
    if result.get("state") == "incomplete" and isinstance(result.get("reason"), str) and result["reason"].strip():
        return result
    sources = {item["turn"]: item["text"] for item in payload["reasoning_history"]}
    turn, quote = result.get("quote_turn"), result.get("quote")
    if (result.get("state") == "complete" and type(turn) is int
            and isinstance(quote, str) and quote.strip() and quote in sources.get(turn, "")):
        return result
    return {"state": "uncertain", "reason": "unresolved_or_invalid_conclusion_quote", "review": result}
