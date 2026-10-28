"""Directional airport aliases shared by menu ingestion and retrieval."""

import re

ALIASES = {
    "dxb": "DXB",
    "dubai": "DXB",
    "lhr": "LHR",
    "london": "LHR",
    "london heathrow": "LHR",
    "heathrow": "LHR",
    "bom": "BOM",
    "mumbai": "BOM",
    "sin": "SIN",
    "singapore": "SIN",
}
_PLACES = "|".join(re.escape(name) for name in sorted(ALIASES, key=len, reverse=True))
_ROUTE = re.compile(rf"\b({_PLACES})\s*(?:to|->|→|-)\s*({_PLACES})\b", re.I)


def extract_route(text):
    matches = _ROUTE.findall(text)
    routes = {f"{ALIASES[a.lower()]}-{ALIASES[b.lower()]}" for a, b in matches}
    return next(iter(routes)) if len(routes) == 1 else None


def menu_route(question):
    if re.search(r"\b(menu|menus|meal|meals|food|drink|drinks|dining|dessert|desserts)\b", question, re.I):
        return extract_route(question)
    return None
