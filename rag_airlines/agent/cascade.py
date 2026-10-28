"""Deterministic intent routing for the public policy RAG pipeline."""

import re

def classify_public_department(question):
    """Deterministic department routing for public policy questions."""
    text = " ".join(question.casefold().split())
    # EU-law terms identify the disruptions/legal corpus even when the user
    # does not repeat the regulation number (for example, "extraordinary
    # circumstances and compensation").
    if re.search(r"\b(?:extraordinary circumstances|eu law|eu261|ec261|261\s*/\s*2004)\b", text):
        return "disruptions"
    if re.search(r"\b(?:eu|ec)[\s-]?261(?:\s*/\s*2004)?\b|\b261\s*/\s*2004\b|\bdenied boarding\b", text):
        return "disruptions"
    if re.search(r"\b(refund|refunds|reimbursement)\b", text) and re.search(
        r"\b(ticket|tickets|fare|cancel(?:led|ed|lation)?)\b", text
    ):
        return "refunds_claims"
    if re.search(r"\b(menu|meal|food|drink|beverage|onboard|in.flight)\b", text):
        return "cabin_onboard"
    if re.search(r"\b(baggage|bag|luggage|checked bag|carry.on)\b", text):
        if re.search(r"\b(lost|missing|damaged|damage|delayed|claim|reimburse|compensation)\b", text):
            return "refunds_claims"
        return "baggage"
    if re.search(r"\b(points?|miles|rewards?|loyalty|redemption|redeem)\b", text):
        return "rewards_loyalty"
    if re.search(r"\b(upgrade|upgrades|business class)\b", text):
        return "upgrades"
    if re.search(r"\b(?:flight )?(?:delay|delayed|cancell?ation|cancelled|canceled)\b", text) and re.search(
        r"\b(compensation|reimbursement|refund|refunds|care|accommodation|assistance)\b", text
    ):
        return "disruptions"
    if re.search(r"\b(refund|refunds|reimbursement|compensation|claim|claims)\b", text):
        return "refunds_claims"
    if re.search(r"\b(delay|delayed|cancell?ation|cancelled|canceled|missed connection|stranded|hotel)\b", text):
        return "disruptions"
    if re.search(r"\b(check.in|travel documents?|passport|boarding|codeshare)\b", text):
        return "general_policies"
    return None


def needs_customer_workflow(question):
    """Detect requests needing personal records or live operational data."""
    text = " ".join(question.casefold().split())
    if re.search(r"\b(?:flight status|where is (?:my )?flight|is my flight .{0,30}delayed)\b", text):
        return True
    if re.search(r"\b(?:upgrade|upgrades)\b", text) and re.search(
        r"\b(?:my offer|offer|availability|inventory|waitlist|booking)\b", text
    ):
        return True
    if re.search(r"\b(?:points?|miles|rewards?)\b", text) and re.search(
        r"\b(?:my balance|balance|how many|missing|my points|my miles)\b", text
    ):
        return True
    return False


def route_public_intent(question):
    """Return one deterministic retrieval route, or None when unsupported."""
    text = " ".join(question.casefold().split())
    if re.search(
        r"\b(?:which parts|what parts|distinguish|separate)\b.*\b(?:eu law|law|aeronova policy|company policy|source|sources)\b",
        text,
    ) or re.search(r"\b(?:eu law|aeronova policy)\b.*\b(?:which|what|source|sources)\b", text):
        return "inspect_metadata"
    if re.search(r"\band\b", text):
        topics = sum(
            bool(re.search(pattern, text))
            for pattern in (
                r"\b(?:menu|meal|food|drink|beverage|onboard|in.flight)\b",
                r"\b(?:baggage|bags?|luggage|checked bag|carry.on)\b",
                r"\b(?:upgrade|upgrades|business class)\b",
                r"\b(?:points?|miles|rewards?|loyalty|redemption|redeem)\b",
                r"\b(?:refund|refunds|reimbursement|claim|claims)\b",
            )
        )
        if topics > 1:
            return None
    return classify_public_department(question)
