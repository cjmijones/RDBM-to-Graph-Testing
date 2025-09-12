#!/usr/bin/env python3
import os, json, time, sqlite3, uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import requests
import pandas as pd
from jsonschema import Draft202012Validator
from dotenv import load_dotenv

# ============================
# DB helpers
# ============================
def rows_to_dicts(cur) -> List[Dict[str, Any]]:
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]

def fetch_person_ids(conn, limit=50) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM people ORDER BY random() LIMIT ?;", (limit,))
    return [str(r[0]) for r in cur.fetchall()]

def fetch_slice_for_person(conn, person_id: str) -> Dict[str, Any]:
    cur = conn.cursor()
    pack: Dict[str, Any] = {}

    # core
    cur.execute("SELECT * FROM people WHERE id = ?;", (person_id,))
    pack["people"] = rows_to_dicts(cur)

    # direct
    cur.execute("SELECT * FROM people_features WHERE person_id = ?;", (person_id,))
    pf = rows_to_dicts(cur); pack["people_features"] = pf
    pf_ids  = [row["pf_id"] for row in pf]
    feat_ids= [row["feature_id"] for row in pf]

    # secondary joins
    if pf_ids:
        q = f"SELECT * FROM people_features_data_points WHERE pf_id IN ({','.join(['?']*len(pf_ids))});"
        cur.execute(q, pf_ids)
        pack["people_features_data_points"] = rows_to_dicts(cur)
    else:
        pack["people_features_data_points"] = []

    if feat_ids:
        q = f"SELECT * FROM features_schema WHERE feature_id IN ({','.join(['?']*len(feat_ids))});"
        cur.execute(q, feat_ids)
        pack["features_schema"] = rows_to_dicts(cur)
    else:
        pack["features_schema"] = []

    # data_points (by original_person_id)
    cur.execute("SELECT * FROM data_points WHERE original_person_id = ?;", (person_id,))
    pack["data_points"] = rows_to_dicts(cur)

    # org affiliations
    cur.execute("SELECT * FROM people_orgs WHERE person_id = ?;", (person_id,))
    po = rows_to_dicts(cur); pack["people_orgs"] = po
    org_ids = list({row["org_id"] for row in po})
    if org_ids:
        q = f"SELECT * FROM orgs WHERE id IN ({','.join(['?']*len(org_ids))});"
        cur.execute(q, org_ids)
        pack["orgs"] = rows_to_dicts(cur)
    else:
        pack["orgs"] = []

    # events
    cur.execute("SELECT * FROM people_events WHERE person_id = ?;", (person_id,))
    pe = rows_to_dicts(cur); pack["people_events"] = pe
    event_ids = list({row["event_id"] for row in pe if row.get("event_id")})
    if event_ids:
        q = f"SELECT * FROM events WHERE id IN ({','.join(['?']*len(event_ids))});"
        cur.execute(q, event_ids)
        pack["events"] = rows_to_dicts(cur)
    else:
        pack["events"] = []

    return pack

# ============================
# JSON guards / normalization
# ============================
def coerce_min_contract(patch: dict) -> dict:
    if not isinstance(patch.get("nodes"), list):       patch["nodes"] = []
    if not isinstance(patch.get("edges"), list):       patch["edges"] = []
    if not isinstance(patch.get("provenance"), list):  patch["provenance"] = []  # critical
    return patch

def _ensure_str(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) if x is not None else ""

def _primary_type(t: Any) -> str:
    # Accept string or list of strings; choose first when list
    if isinstance(t, list) and t:
        return str(t[0])
    return str(t) if t is not None else "Thing"

def _merge_obj(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(dst or {})
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_obj(out[k], v)
        else:
            out[k] = v
    return out

def normalize_patch_to_schema(patch: Dict[str, Any], *, fallback_person_id: Optional[str]=None) -> Dict[str, Any]:
    """Normalize LLM output so it passes the schema and preserves richness in labels/external_ids."""
    nodes = []
    for n in patch.get("nodes", []):
        n = dict(n)
        # id
        n["id"] = _ensure_str(n.get("id") or uuid.uuid4().hex)
        # type -> primary string; keep extra types in labels.types
        t = n.get("type")
        primary = _primary_type(t)
        extra_types = t[1:] if isinstance(t, list) and len(t) > 1 else []
        n["type"] = primary

        labels = dict(n.get("labels") or {})
        if extra_types:
            labels["types"] = [str(x) for x in extra_types]

        # hoist common fields into labels/external_ids then remove from top
        top_to_labels = [
            ("name", "name"),
            ("fullName", "fullName"),
            ("description", "description"),
            ("current_title", "current_title"),
            ("homeLocation", "homeLocation"),
            ("gender", "gender"),
            ("avatar", "avatar_url"),
            ("avatar_url", "avatar_url"),
            ("avatarUrl", "avatar_url"),
            ("username", "username")
        ]
        for src, dst in top_to_labels:
            if src in n and n[src] is not None:
                labels[dst] = n.pop(src)

        external_ids = dict(n.get("external_ids") or {})
        for key in ["linkedin_id", "linkedinId", "orcid", "twitter", "github", "google_scholar", "wikidata"]:
            if key in n and n[key] is not None:
                external_ids[key] = n.pop(key)

        n["labels"] = labels
        n["external_ids"] = external_ids
        # keep only whitelisted top-level keys (but schema is permissive anyway)
        nodes.append(n)

    edges = []
    for e in patch.get("edges", []):
        if not isinstance(e, dict): continue
        sub = _ensure_str(e.get("sub"))
        pred= _ensure_str(e.get("pred"))
        obj = _ensure_str(e.get("obj"))
        if not (sub and pred and obj):  # skip incomplete edges
            continue
        new_e = {"sub": sub, "pred": pred, "obj": obj}
        quals = e.get("qualifiers")
        if isinstance(quals, dict) and quals:
            # keep common qualifiers and any additional (dates, roles, titles, confidence)
            new_e["qualifiers"] = dict(quals)
        edges.append(new_e)

    prov = []
    for p in patch.get("provenance", []):
        if not isinstance(p, dict): continue
        t = p.get("triple") or {}
        ev= p.get("evidence") or {}
        triple = {}
        for k in ("sub","pred","obj"):
            if k in t and t[k] is not None:
                triple[k] = _ensure_str(t[k])
        evidence_allowed = {"source_table","source_pk","data_point_id","source_url","quote","confidence","collected_at"}
        evidence = {k: ev.get(k) for k in evidence_allowed if k in ev}
        if triple:
            prov.append({"triple": triple, "evidence": evidence})

    # If provenance is empty but we have edges, synthesize minimal provenance
    if not prov and edges:
        for e in edges:
            prov.append({
                "triple": {"sub": e["sub"], "pred": e["pred"], "obj": e["obj"]},
                "evidence": {"source_table": "people", "source_pk": fallback_person_id}
            })

    return {"nodes": nodes, "edges": edges, "provenance": prov}

# ============================
# LLM calls (with logging & fallbacks)
# ============================
def call_mistral(messages, api_key, model, mistral_url, max_tokens, use_json_object=True):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,   # best for strict JSON
        "max_tokens": max_tokens
    }
    if use_json_object:
        payload["response_format"] = {"type": "json_object"}
    resp = requests.post(mistral_url, headers=headers, json=payload, timeout=120)
    if not resp.ok:
        try:
            err = resp.json()
        except Exception:
            err = {"raw": resp.text}
        raise requests.HTTPError(f"{resp.status_code} {resp.reason} | detail: {err}", response=resp)
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def safe_call_mistral(messages, api_key, model, mistral_url, max_tokens):
    try:
        return call_mistral(messages, api_key, model, mistral_url, max_tokens, use_json_object=True)
    except requests.HTTPError as e:
        detail = (str(e) or "").lower()
        # If response_format is not accepted or too long, retry without it and smaller tokens
        if "response_format" in detail or "invalid" in detail:
            return call_mistral(messages, api_key, model, mistral_url, max_tokens, use_json_object=False)
        if "context" in detail or "max_tokens" in detail or "length" in detail or "too many" in detail:
            # chop the longest line (usually the big context) and reduce tokens
            slim = []
            for m in messages:
                c = m.get("content","")
                if c.startswith("JSON context:"):
                    continue
                slim.append(m)
            return call_mistral(slim, api_key, model, mistral_url, max_tokens=max(512, max_tokens//2), use_json_object=True)
        raise

# ============================
# Context building
# ============================
def _cap(rows, n): 
    return rows if len(rows) <= n else rows[:n]

def make_user_context(person_pack: Dict[str, Any]) -> str:
    # Caps keep the request inside context limits but still rich
    ctx = {
        "focus_person_rows": _cap(person_pack.get("people", []), 1),
        "people_features": _cap(person_pack.get("people_features", []), 30),
        "feature_definitions": _cap(person_pack.get("features_schema", []), 30),
        "feature_evidence": _cap(person_pack.get("people_features_data_points", []), 60),
        "data_points": _cap(person_pack.get("data_points", []), 60),
        "org_links": _cap(person_pack.get("people_orgs", []), 30),
        "org_rows": _cap(person_pack.get("orgs", []), 30),
        "event_links": _cap(person_pack.get("people_events", []), 40),
        "event_rows": _cap(person_pack.get("events", []), 40),
    }
    return json.dumps(ctx, ensure_ascii=False)

# ============================
# Validation
# ============================
def is_valid_patch(patch: Dict[str, Any], validator: Draft202012Validator) -> Tuple[bool, List[str]]:
    errors = sorted(validator.iter_errors(patch), key=lambda e: e.path)
    if errors:
        msgs = []
        for e in errors:
            loc = ".".join([str(x) for x in e.path])
            msgs.append(f"{loc}: {e.message}")
        return False, msgs
    return True, []

# ============================
# Merge
# ============================
def merge_graph(G: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    if "nodes" not in G:       G["nodes"] = {}
    if "edges" not in G:       G["edges"] = []
    if "provenance" not in G:  G["provenance"] = []

    id_map: Dict[str, str] = {}

    def canon_key(n: Dict[str, Any]) -> str:
        t = n.get("type", "?")
        if isinstance(t, list) and t: t = t[0]
        name = (n.get("labels", {}) or {}).get("fullName") \
            or (n.get("labels", {}) or {}).get("name") \
            or (n.get("labels", {}) or {}).get("label")
        name = (name or "").strip().lower()
        return f"{t}::{name}" if name else f"{t}::id:{n['id']}"

    existing_keys = { canon_key(v): k for k, v in G["nodes"].items() }

    for n in patch.get("nodes", []):
        key = canon_key(n)
        if key in existing_keys:
            cid = existing_keys[key]
            merged = {**G["nodes"][cid]}
            merged["labels"] = _merge_obj(merged.get("labels") or {}, n.get("labels") or {})
            merged["external_ids"] = _merge_obj(merged.get("external_ids") or {}, n.get("external_ids") or {})
            G["nodes"][cid] = merged
            id_map[n["id"]] = cid
        else:
            cid = n["id"]
            if cid in G["nodes"]:
                cid = f"{cid}-{uuid.uuid4().hex[:6]}"
            G["nodes"][cid] = n
            id_map[n["id"]] = cid

    for e in patch.get("edges", []):
        sub = id_map.get(e["sub"], e["sub"])
        obj = id_map.get(e["obj"], e["obj"])
        new_e = {"sub": sub, "pred": e["pred"], "obj": obj}
        if "qualifiers" in e and e["qualifiers"]:
            new_e["qualifiers"] = e["qualifiers"]
        G["edges"].append(new_e)

    for p in patch.get("provenance", []):
        t = dict(p.get("triple") or {})
        t["sub"] = id_map.get(t.get("sub"), t.get("sub"))
        t["obj"] = id_map.get(t.get("obj"), t.get("obj"))
        G["provenance"].append({"triple": t, "evidence": p.get("evidence", {})})

    return G

# ============================
# Core exportable function
# ============================
def build_kg(
    *,
    api_key: str,
    model: str = "mistral-small-latest",
    mistral_url: str = "https://api.mistral.ai/v1/chat/completions",
    db_path: str = "mock_dataset/mock_people.db",
    out_dir: str = "kg_out",
    kg_schema: Dict[str, Any],
    system_prompt: str,
    user_instructions: str,
    person_limit: int = 50,
    max_tokens: int = 4096,
    sleep_sec: float = 0.4
) -> Dict[str, Any]:
    """
    Build a YAGO/Schema.org–aligned KG from a local SQLite slice using a Mistral model.
    Returns graph_state: {"nodes": {id->node}, "edges": [...], "provenance": [...]}
    and writes artifacts to out_dir (graph.jsonl, graph_state.json, CSVs).
    """
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    validator = Draft202012Validator(kg_schema)

    conn = sqlite3.connect(db_path)
    conn.row_factory = None

    person_ids = fetch_person_ids(conn, limit=person_limit)
    print(f"[info] building KG for {len(person_ids)} people from {db_path}")

    graph_state: Dict[str, Any] = {"nodes": {}, "edges": [], "provenance": []}
    graph_jsonl = (out_path / "graph.jsonl").open("a", encoding="utf-8")

    for i, pid in enumerate(person_ids, 1):
        pack = fetch_slice_for_person(conn, pid)
        user_ctx = make_user_context(pack)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_instructions},
            {"role": "user",   "content": f"JSON context:\n{user_ctx}"},
            {"role": "user",   "content": "Output the KG JSON now."}
        ]

        try:
            raw = safe_call_mistral(messages, api_key=api_key, model=model, mistral_url=mistral_url, max_tokens=max_tokens)
        except Exception as e:
            print(f"[warn] LLM call failed for {pid}: {e}")
            continue

        def try_parse(raw_txt: str) -> Dict[str, Any]:
            return json.loads(raw_txt)

        # Parse + coerce + normalize
        try:
            patch = try_parse(raw)
            patch = coerce_min_contract(patch)
            patch = normalize_patch_to_schema(patch, fallback_person_id=pid)
        except json.JSONDecodeError:
            fix_msgs = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your previous message was not valid JSON. Return ONLY valid JSON per the schema. Include nodes, edges, provenance arrays (empty if none)."}
            ]
            try:
                fixed = safe_call_mistral(fix_msgs, api_key=api_key, model=model, mistral_url=mistral_url, max_tokens=max_tokens)
                patch = try_parse(fixed)
                patch = coerce_min_contract(patch)
                patch = normalize_patch_to_schema(patch, fallback_person_id=pid)
            except Exception as e:
                print(f"[warn] JSON repair failed for {pid}: {e}")
                continue

        ok, errs = is_valid_patch(patch, validator)
        if not ok:
            repair_msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "The following JSON failed schema validation. Return a corrected JSON ONLY (no commentary). Always include nodes, edges, provenance arrays; 'type' must be a single string like 'schema:Person'."},
                {"role": "user", "content": json.dumps(patch, ensure_ascii=False)},
                {"role": "user", "content": "Errors:\n" + "\n".join(errs)}
            ]
            try:
                fixed = safe_call_mistral(repair_msgs, api_key=api_key, model=model, mistral_url=mistral_url, max_tokens=max_tokens)
                patch2 = json.loads(fixed)
                patch2 = coerce_min_contract(patch2)  # <-- ensure provenance exists
                patch2 = normalize_patch_to_schema(patch2, fallback_person_id=pid)
                ok2, errs2 = is_valid_patch(patch2, validator)
                if not ok2:
                    print(f"[warn] validation still failing for {pid}: {errs2[:3]}")
                    continue
                patch = patch2
            except Exception as e:
                print(f"[warn] repair call failed for {pid}: {e}")
                continue

        # Save last patch for debugging
        (out_path / "latest_patch.json").write_text(json.dumps(patch, indent=2, ensure_ascii=False), encoding="utf-8")

        # Merge + audit trail
        graph_state = merge_graph(graph_state, patch)
        graph_jsonl.write(json.dumps({"person_id": pid, "patch": patch}, ensure_ascii=False) + "\n")
        graph_jsonl.flush()

        print(f"[ok] merged person {i}/{len(person_ids)}  id={pid}  nodes={len(patch.get('nodes',[]))} edges={len(patch.get('edges',[]))}")
        time.sleep(sleep_sec)

    graph_jsonl.close()

    # Export final graph artifacts
    nodes_df = pd.DataFrame([{"id": nid, **(node or {})} for nid, node in graph_state["nodes"].items()])
    edges_df = pd.DataFrame(graph_state["edges"])
    prov_df  = pd.DataFrame(graph_state["provenance"])

    nodes_df.to_csv(out_path / "graph_nodes.csv", index=False)
    edges_df.to_csv(out_path / "graph_edges.csv", index=False)
    prov_df.to_csv(out_path / "graph_provenance.csv", index=False)

    (out_path / "graph_state.json").write_text(json.dumps(graph_state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] graph built → {out_path.resolve()}")
    print(f"      nodes={len(graph_state['nodes'])} edges={len(graph_state['edges'])} prov={len(graph_state['provenance'])}")
    return graph_state

# ============================
# Schema (relaxed but typed)
# ============================
DEFAULT_KG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["nodes", "edges", "provenance"],
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type"],
                "properties": {
                    "id": {"type": "string"},
                    # accept string or list of strings; we normalize to a primary string anyway
                    "type": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}}
                        ]
                    },
                    "labels": {"type": "object"},
                    "external_ids": {"type": "object"}
                },
                "additionalProperties": True
            }
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["sub", "pred", "obj"],
                "properties": {
                    "sub": {"type": "string"},
                    "pred": {"type": "string"},
                    "obj": {"type": "string"},
                    "qualifiers": {"type": "object"}
                },
                "additionalProperties": True
            }
        },
        "provenance": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "triple": {
                        "type": "object",
                        "properties": {
                            "sub": {"type": "string"},
                            "pred": {"type": "string"},
                            "obj": {"type": "string"}
                        },
                        "additionalProperties": True
                    },
                    "evidence": {
                        "type": "object",
                        "properties": {
                            "source_table": {"type": "string"},
                            "source_pk": {"type": ["string", "number"]},
                            "data_point_id": {"type": ["string", "null"]},
                            "source_url": {"type": ["string", "null"]},
                            "quote": {"type": ["string", "null"]},
                            "confidence": {"type": ["number", "null"]},
                            "collected_at": {"type": ["string", "null"]}
                        },
                        "additionalProperties": True
                    }
                },
                "additionalProperties": True
            }
        }
    },
    "additionalProperties": True
}

# ============================
# Prompts
# ============================
DEFAULT_SYSTEM_PROMPT = """You convert relational rows into a compact YAGO/Schema.org knowledge subgraph.

Output contract:
- Output ONLY one JSON object with top-level keys: nodes (array), edges (array), provenance (array). Use empty arrays if none.
- Each node must have: id (string), type (single string like "schema:Person"). You may include EXTRA helpful attributes inside labels/external_ids.
- Prefer types like: schema:Person, schema:Organization (use EducationalOrganization if obvious), schema:Event, schema:Language, yago:Award, yago:Gender, yago:BeliefSystem, schema:Place.
- Prefer property directions: schema:parentOrganization, schema:superEvent.
- For Person: use schema:alumniOf, schema:worksFor, schema:memberOf, schema:affiliation, schema:knowsLanguage, schema:award, schema:gender, schema:homeLocation, schema:participant.

Richness & qualifiers:
- Populate node.labels with: name/fullName, description, current_title, homeLocation, gender, username, avatar_url, plus any useful typed fields (e.g., languages, awards).
- On edges, include qualifiers when evident: role/title, startDate, endDate, location, confidence.
- Use external_ids for linkable identifiers: linkedin_id, twitter, github, wikidata, orcid, google_scholar.

Provenance:
- For each asserted edge, include a lightweight provenance object with triple {sub,pred,obj} and evidence {source_table, source_pk, data_point_id, source_url, quote, confidence} when available in rows.

Do not invent facts. If unsure, omit. Return valid JSON only (no markdown, no comments)."""

DEFAULT_USER_INSTRUCTIONS = """Build a preliminary graph for the focus person and directly referenced entities visible in the JSON context.
Use the available rows; the context may be sampled for brevity. Include nodes, edges, and provenance arrays (empty is allowed)."""

# ============================
# CLI
# ============================
if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parent_env = here.parent / ".env"
    load_dotenv(dotenv_path=parent_env) if parent_env.exists() else load_dotenv()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Set MISTRAL_API_KEY in your environment.")

    model = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
    mistral_url = os.environ.get("MISTRAL_URL", "https://api.mistral.ai/v1/chat/completions")
    db_path = os.environ.get("KG_SQLITE_PATH", "mock_dataset/mock_people.db")
    out_dir = os.environ.get("KG_OUT_DIR", "kg_out")
    max_tokens = int(os.environ.get("KG_MAX_TOKENS", "4096"))
    person_limit = int(os.environ.get("KG_PERSON_LIMIT", "50"))

    build_kg(
        api_key=api_key,
        model=model,
        mistral_url=mistral_url,
        db_path=db_path,
        out_dir=out_dir,
        kg_schema=DEFAULT_KG_SCHEMA,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_instructions=DEFAULT_USER_INSTRUCTIONS,
        person_limit=person_limit,
        max_tokens=max_tokens,
        sleep_sec=0.4
    )
