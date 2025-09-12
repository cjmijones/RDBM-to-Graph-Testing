#!/usr/bin/env python3
import os, json, time, sqlite3, uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple
import requests
import pandas as pd
from jsonschema import Draft202012Validator
from dotenv import load_dotenv

# ----------------------------
# DB helpers
# ----------------------------
def rows_to_dicts(cur) -> List[Dict[str, Any]]:
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]

def fetch_person_ids(conn, limit=50) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM people ORDER BY random() LIMIT ?;", (limit,))
    return [r[0] for r in cur.fetchall()]

def fetch_slice_for_person(conn, person_id: str) -> Dict[str, Any]:
    cur = conn.cursor()
    pack: Dict[str, Any] = {}

    # core
    cur.execute("SELECT * FROM people WHERE id = ?;", (person_id,))
    pack["people"] = rows_to_dicts(cur)

    # direct
    cur.execute("SELECT * FROM people_features WHERE person_id = ?;", (person_id,))
    pf = rows_to_dicts(cur); pack["people_features"] = pf
    pf_ids = [row["pf_id"] for row in pf]
    feat_ids = [row["feature_id"] for row in pf]

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
    event_ids = list({row["event_id"] for row in pe if row["event_id"]})
    if event_ids:
        q = f"SELECT * FROM events WHERE id IN ({','.join(['?']*len(event_ids))});"
        cur.execute(q, event_ids)
        pack["events"] = rows_to_dicts(cur)
    else:
        pack["events"] = []

    return pack

def coerce_min_contract(patch: dict) -> dict:
    if not isinstance(patch.get("nodes"), list):
        patch["nodes"] = []
    if not isinstance(patch.get("edges"), list):
        patch["edges"] = []
    if not isinstance(patch.get("provenance"), list):
        patch["provenance"] = []   # <-- critical
    return patch

# ----------------------------
# LLM call
# ----------------------------
def call_mistral(messages, api_key, model, mistral_url, max_tokens):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,            # more reliable JSON
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}  # may be rejected; see fallback below
    }
    resp = requests.post(mistral_url, headers=headers, json=payload, timeout=120)
    if not resp.ok:
        # surface the server’s explanation
        try:
            err = resp.json()
        except Exception:
            err = {"raw": resp.text}
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} | detail: {err}",
            response=resp
        )
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ----------------------------
# Context + merge helpers
# ----------------------------
def make_user_context(person_pack: Dict[str, Any]) -> str:
    def j(obj): return json.dumps(obj, ensure_ascii=False)
    ctx = {
        "focus_person_rows": person_pack.get("people", []),
        "people_features": person_pack.get("people_features", []),
        "feature_definitions": person_pack.get("features_schema", []),
        "feature_evidence": person_pack.get("people_features_data_points", []),
        "data_points": person_pack.get("data_points", []),
        "org_links": person_pack.get("people_orgs", []),
        "org_rows": person_pack.get("orgs", []),
        "event_links": person_pack.get("people_events", []),
        "event_rows": person_pack.get("events", []),
    }
    return j(ctx)

def is_valid_patch(patch: Dict[str, Any], validator: Draft202012Validator) -> Tuple[bool, List[str]]:
    errors = sorted(validator.iter_errors(patch), key=lambda e: e.path)
    if errors:
        msgs = []
        for e in errors:
            loc = ".".join([str(x) for x in e.path])
            msgs.append(f"{loc}: {e.message}")
        return False, msgs
    return True, []

def merge_graph(G: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    if "nodes" not in G:
        G["nodes"] = {}
    if "edges" not in G:
        G["edges"] = []
    if "provenance" not in G:
        G["provenance"] = []

    id_map: Dict[str, str] = {}

    def canon_key(n: Dict[str, Any]) -> str:
        t = n.get("type","?")
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
            merged_labels = {**(merged.get("labels") or {}), **(n.get("labels") or {})}
            merged["labels"] = merged_labels
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
        t = p.get("triple", {})
        t["sub"] = id_map.get(t.get("sub"), t.get("sub"))
        t["obj"] = id_map.get(t.get("obj"), t.get("obj"))
        G["provenance"].append({"triple": t, "evidence": p.get("evidence", {})})

    return G

# ----------------------------
# Core exportable function
# ----------------------------
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
    max_tokens: int = 1024,
    sleep_sec: float = 0.4
) -> Dict[str, Any]:
    """
    Build a YAGO/Schema.org–adherent KG from a local SQLite slice using a Mistral model.

    Returns the final graph_state dict: {"nodes": {id->node}, "edges": [...], "provenance": [...]}
    and writes artifacts to out_dir (graph.jsonl, graph_state.json, csvs).
    """
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    validator = Draft202012Validator(kg_schema)

    # Connect
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
            {"role": "user", "content": user_instructions},
            {"role": "user", "content": f"JSON context:\n{user_ctx}"},
            {"role": "user", "content": "Output the KG JSON now."}
        ]

        try:
            raw = call_mistral(messages, api_key=api_key, model=model, mistral_url=mistral_url, max_tokens=max_tokens)
        except Exception as e:
            print(f"[warn] LLM call failed for {pid}: {e}")
            continue

        # Parse + validate + light repair
        def try_parse(raw_txt: str) -> Dict[str, Any]:
            return json.loads(raw_txt)

        try:
            patch = try_parse(raw)
            patch = coerce_min_contract(patch)
        except json.JSONDecodeError:
            fix_msgs = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your previous message was not valid JSON. Return ONLY valid JSON per the schema."}
            ]
            try:
                fixed = call_mistral(fix_msgs, api_key=api_key, model=model, mistral_url=mistral_url, max_tokens=max_tokens)
                patch = try_parse(fixed)
                patch = coerce_min_contract(patch)
            except Exception as e:
                print(f"[warn] JSON repair failed for {pid}: {e}")
                continue

        ok, errs = is_valid_patch(patch, validator)
        if not ok:
            repair_msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "The following JSON failed schema validation. Return a corrected JSON only."},
                {"role": "user", "content": json.dumps(patch, ensure_ascii=False)},
                {"role": "user", "content": "Errors:\n" + "\n".join(errs)}
            ]
            try:
                fixed = call_mistral(repair_msgs, api_key=api_key, model=model, mistral_url=mistral_url, max_tokens=max_tokens)
                patch2 = json.loads(fixed)
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

# ----------------------------
# Defaults for easy CLI usage
# ----------------------------
DEFAULT_KG_SCHEMA = {
    "type": "object",
    "required": ["nodes", "edges", "provenance"],
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                # keep minimal contract
                "required": ["id", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "labels": {"type": "object"},
                    "external_ids": {"type": "object"}
                },
                # ALLOW extra keys like name, avatar_url, linkedin_id, etc.
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
                # ALLOW extra keys if the model includes them
                "additionalProperties": True
            }
        },
        "provenance": {
            "type": "array",
            "items": {
                "type": "object",
                # Make both optional to avoid rejecting useful patches.
                # We'll accept partial provenance and fix it later.
                "properties": {
                    "triple": {
                        "type": "object",
                        "properties": {
                            "sub": {"type": "string"},
                            "pred": {"type": "string"},
                            "obj": {"type": "string"}
                        },
                        # allow additional keys like source_predicate, time, etc.
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
    # allow top-level extras if needed in the future
    "additionalProperties": True
}


DEFAULT_SYSTEM_PROMPT = """You convert relational rows into a small YAGO/Schema.org compliant knowledge subgraph.

Rules:
- Use classes with examples like: schema:Person, schema:Organization (and EducationalOrganization if obvious), schema:Event, schema:Language, yago:Award, yago:Gender, yago:BeliefSystem, schema:Place.
- Prefer property directions: use schema:parentOrganization (not childOrganization); use schema:superEvent (not subEvent).
- For Person edges, prefer: schema:alumniOf, schema:worksFor, schema:memberOf, schema:affiliation, schema:knowsLanguage, schema:award, schema:gender, schema:homeLocation, schema:participant (for events).
- You may add additional schema types as long as they align with YAGO best practices
- Emit only valid JSON with keys: nodes, edges, provenance (no extra keys).
- Node ids must be strings; reuse ids inside your output when linking.
- Example: For organizations detected as universities/colleges, type them as EducationalOrganization.
- Add lightweight provenance for edges where possible (table name, source pk, data_point_id, url/quote if present).
- Do not invent unknown facts; if unsure, omit.
"""

DEFAULT_USER_INSTRUCTIONS = """Build a preliminary graph for the focus person and any directly referenced entities implied by the rows.
Return JSON matching the provided JSON Schema, nothing else."""

# ----------------------------
# CLI shim (optional)
# ----------------------------
if __name__ == "__main__":
    # Load env for quick runs
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
    max_tokens = int(os.environ.get("KG_MAX_TOKENS", "1024"))
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
