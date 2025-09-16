# adapters/people_doc_builder.py
from __future__ import annotations
import re
from typing import Any, Dict, List, Iterable

def _s(x: Any) -> str:
    return str(x).strip() if x is not None else ""

def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip()).strip("_").lower()
    return s or "value"

def _norm_phone(raw: str | None) -> str | None:
    if not raw:
        return None
    digits = re.sub(r"[^\d+]", "", raw)
    return digits or None

def flatten_json_to_open_facts(person: Dict[str, Any]) -> List[str]:
    """
    Convert any nested sub-JSON (e.g., person['data_points'][*]['source_json']) to
    path-based open:* facts like:
      Person "Alice" open:positions__idx0__title "Partner".
    """
    name = person.get("full_name") or person.get("person_id") or "unknown_person"
    pid  = person.get("person_id") or ""

    facts = []
    if pid:
        facts.append(f'Person "{name}" has_person_id "{pid}".')

    def walk(prefix: str, node: Any) -> None:
        if node is None:
            return
        if isinstance(node, (str, int, float, bool)):
            val = str(node).strip()
            if val != "":
                rel = _slug(prefix)
                facts.append(f'Person "{name}" open:{rel} "{val}".')
            return
        if isinstance(node, list):
            for i, item in enumerate(node):
                walk(f"{prefix}__idx{i}" if prefix else f"idx{i}", item)
            return
        if isinstance(node, dict):
            for k, v in node.items():
                key = _slug(k)
                walk(f"{prefix}__{key}" if prefix else key, v)
            return

    for dp in person.get("data_points", []):
        src = dp.get("source_json")
        if src:
            walk("", src)

    # dedupe while keeping order
    seen = set()
    uniq = []
    for f in facts:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq

def precise_facts_from_record(person: Dict[str, Any]) -> List[str]:
    """
    Optional: emit a few precise relations that you want to bias toward (email, phone, company, title).
    OIE will also see these and SC will map/verify.
    """
    name = person.get("full_name") or person.get("person_id") or "unknown_person"
    pid  = person.get("person_id") or ""
    lines: List[str] = []

    if pid:
        lines.append(f'Person "{name}" has_person_id "{pid}".')

    emails: List[str] = []
    phones: List[str] = []
    company = None
    title   = None
    country = None
    links: List[str] = []

    for dp in person.get("data_points", []):
        src = dp.get("source_json") or {}
        if src.get("primary_email"): emails.append(str(src["primary_email"]).strip())
        if src.get("primary_phone_number"): phones.append(str(src["primary_phone_number"]).strip())
        if (src.get("current_company")): company = _s(src["current_company"])
        if (src.get("current_title")):   title   = _s(src["current_title"])
        g = src.get("geo") or {}
        if g.get("country"): country = _s(g["country"])
        if src.get("profile_url"): links.append(_s(src["profile_url"]))
        if src.get("linkedin_id"): links.append(f'https://www.linkedin.com/in/{_s(src["linkedin_id"])}')

    # normalize & dedupe
    emails = sorted({e.lower() for e in emails if e})
    phones = [p for p in {_norm_phone(p) for p in phones} if p]
    links  = sorted({u for u in links if u})

    if title:
        lines.append(f'Person "{name}" has_title "{title}".')
    if company:
        lines.append(f'Person "{name}" works_at "{company}".')
    if country:
        lines.append(f'Person "{name}" located_in_country "{country}".')

    for e in emails:
        lines.append(f'Person "{name}" has_email "{e}".')
    for p in phones:
        lines.append(f'Person "{name}" has_phone "{p}".')
    for u in links:
        lines.append(f'Person "{name}" has_profile_url "{u}".')

    return lines

def build_doc_for_person(person: Dict[str, Any],
                         use_precise: bool = True,
                         cap_open_facts: int = 500) -> str:
    """Concatenate precise facts (optional) + open:* facts (capped) → final doc."""
    parts: List[str] = []
    if use_precise:
        parts.extend(precise_facts_from_record(person))
    open_facts = flatten_json_to_open_facts(person)
    if cap_open_facts and len(open_facts) > cap_open_facts:
        open_facts = open_facts[:cap_open_facts]
    parts.extend(open_facts)
    # Ensure at least one provenance line:
    pid = person.get("person_id") or ""
    if pid:
        parts.append(f'Provenance person_id "{pid}".')
    return "\n".join(parts)

def build_docs_from_people(people: Iterable[Dict[str, Any]],
                           use_precise: bool = True,
                           cap_open_facts: int = 500) -> List[str]:
    return [build_doc_for_person(p, use_precise=use_precise, cap_open_facts=cap_open_facts) for p in people]
