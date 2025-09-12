#!/usr/bin/env python3
import os, json, time, sqlite3, uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple
import requests
from jsonschema import validate, Draft202012Validator
import pandas as pd
from dotenv import load_dotenv

# ----------------------------
# Config
# ----------------------------
import os

# Check if we're in a Colab environment
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

OUT_DIR = Path("../kg_out")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Load environment variables
if is_colab():
    # In Colab, use userdata schema for secrets
    try:
        from google.colab import userdata
        # Map userdata keys to environment variables
        
        KG_SQLITE_PATH = userdata.get('KG_SQLITE_PATH', "mock_dataset/mock_people.db")
        MISTRAL_API_KEY = userdata.get('MISTRAL_API_KEY')
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY is required but not found in userdata")
        MISTRAL_MODEL = userdata.get('MISTRAL_MODEL', "mistral-small-latest")
        MISTRAL_URL = userdata.get('MISTRAL_URL', "https://api.mistral.ai/v1/chat/completions")
        
        # Budget control
        KG_MAX_TOKENS = userdata.get('KG_MAX_TOKENS', "1024")

    except Exception as e:
        print(f"Warning: Could not load userdata in Colab: {e}")
        # Fall back to default values
        pass
else:
    # Standard environment - load from .env file
    parent_env = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=parent_env) if parent_env.exists() else load_dotenv()

    DB_PATH = os.environ.get("KG_SQLITE_PATH", "../mock_dataset/mock_people.db")

    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is required but not found in environment variables")

    MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")  # e.g. mistral-small-latest / open-mistral-7b
    MISTRAL_URL = os.environ.get("MISTRAL_URL", "https://api.mistral.ai/v1/chat/completions")

    # Budget control
    PER_REQUEST_MAX_TOKENS = int(os.environ.get("KG_MAX_TOKENS", "1024"))

# ----------------------------
# Minimal YAGO/Schema.org oriented JSON schema
# (tight enough to keep models honest; broad enough to not fight them)
# ----------------------------
KG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["nodes", "edges", "provenance"],
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type"],
                "properties": {
                    "id": {"type": "string"},  # local IRI/curie, e.g., person:UUID
                    "type": {"type": "string"},  # e.g., "schema:Person", "schema:Organization"
                    "labels": {"type": "object"},  # freeform literals e.g., fullName, url, sameAs
                    "external_ids": {"type": "object"}  # optional: wikidata/schema ids
                },
                "additionalProperties": False
            }
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["sub", "pred", "obj"],
                "properties": {
                    "sub": {"type": "string"},
                    "pred": {"type": "string"},  # e.g., "schema:alumniOf", "schema:worksFor", "schema:participant"
                    "obj": {"type": "string"},
                    "qualifiers": {"type": "object"},  # e.g., startDate, endDate
                },
                "additionalProperties": False
            }
        },
        "provenance": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["triple", "evidence"],
                "properties": {
                    "triple": {"type": "object",
                               "required": ["sub", "pred", "obj"],
                               "properties": {
                                   "sub": {"type": "string"},
                                   "pred": {"type": "string"},
                                   "obj": {"type": "string"},
                               },
                               "additionalProperties": False
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
                        }
                    }
                },
                "additionalProperties": False
            }
        }
    },
    "additionalProperties": False
}

validator = Draft202012Validator(KG_SCHEMA)

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

# ----------------------------
# LLM call (Mistral chat)
# ----------------------------
def call_mistral(messages: List[Dict[str, str]], model: str = MISTRAL_MODEL, max_tokens: int = PER_REQUEST_MAX_TOKENS) -> str:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}  # ask Mistral to return JSON
    }
    resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ----------------------------
# Prompt templates
# ----------------------------
SYSTEM_PROMPT = """You convert relational rows into a small YAGO/Schema.org compliant knowledge subgraph.

Rules:
- Use classes like: schema:Person, schema:Organization (and EducationalOrganization if obvious), schema:Event, schema:Language, yago:Award, yago:Gender, yago:BeliefSystem, schema:Place.
- Prefer property directions: use schema:parentOrganization (not childOrganization); use schema:superEvent (not subEvent).
- For Person edges, prefer: schema:alumniOf, schema:worksFor, schema:memberOf, schema:affiliation, schema:knowsLanguage, schema:award, schema:gender, schema:homeLocation, schema:participant (for events).
- Emit only valid JSON with keys: nodes, edges, provenance (no extra keys).
- Node ids must be strings; reuse ids inside your output when linking.
- For organizations detected as universities/colleges, type them as EducationalOrganization.
- Add lightweight provenance for edges where possible (table name, source pk, data_point_id, url/quote if present).
- Do not invent unknown facts; if unsure, omit.
"""

USER_INSTRUCTIONS = """Build a preliminary graph for the focus person and any directly referenced entities implied by the rows. 
Return JSON matching the provided JSON Schema, nothing else."""

def make_user_context(person_pack: Dict[str, Any]) -> str:
    # Compact the context to stay within cheap-token budgets
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

# ----------------------------
# Validation & merge
# ----------------------------
def is_valid_patch(patch: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = sorted(validator.iter_errors(patch), key=lambda e: e.path)
    if errors:
        msgs = []
        for e in errors:
            loc = ".".join([str(x) for x in e.path])
            msgs.append(f"{loc}: {e.message}")
        return False, msgs
    return True, []

def merge_graph(G: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    # G state: {"nodes": {id->node}, "edges": [..], "provenance": [..]}
    # Simple canonical merge: if a node with same (type, normalized label) exists, reuse its id.
    if "nodes" not in G:
        G["nodes"] = {}
    if "edges" not in G:
        G["edges"] = []
    if "provenance" not in G:
        G["provenance"] = []

    # map temp ids to canonical ids
    id_map: Dict[str, str] = {}

    # Helper to derive a canonical key for de-dupe
    def canon_key(n: Dict[str, Any]) -> str:
        t = n.get("type","?")
        name = (n.get("labels", {}) or {}).get("fullName") \
               or (n.get("labels", {}) or {}).get("name") \
               or (n.get("labels", {}) or {}).get("label")
        name = (name or "").strip().lower()
        return f"{t}::{name}" if name else f"{t}::id:{n['id']}"

    existing_keys = { canon_key(v): k for k, v in G["nodes"].items() }

    # Merge nodes
    for n in patch.get("nodes", []):
        key = canon_key(n)
        if key in existing_keys:
            cid = existing_keys[key]
            # merge labels shallowly
            merged = {**G["nodes"][cid]}
            merged_labels = {**(merged.get("labels") or {}), **(n.get("labels") or {})}
            merged["labels"] = merged_labels
            G["nodes"][cid] = merged
            id_map[n["id"]] = cid
        else:
            cid = n["id"]
            # Ensure unique id
            if cid in G["nodes"]:
                cid = f"{cid}-{uuid.uuid4().hex[:6]}"
            G["nodes"][cid] = n
            id_map[n["id"]] = cid

    # Remap edges to canonical node ids
    for e in patch.get("edges", []):
        sub = id_map.get(e["sub"], e["sub"])
        obj = id_map.get(e["obj"], e["obj"])
        new_e = {"sub": sub, "pred": e["pred"], "obj": obj}
        if "qualifiers" in e and e["qualifiers"]:
            new_e["qualifiers"] = e["qualifiers"]
        G["edges"].append(new_e)

    # provenance passthrough
    for p in patch.get("provenance", []):
        # rewrite IDs inside provenance.triple
        t = p.get("triple", {})
        t["sub"] = id_map.get(t.get("sub"), t.get("sub"))
        t["obj"] = id_map.get(t.get("obj"), t.get("obj"))
        G["provenance"].append({"triple": t, "evidence": p.get("evidence", {})})

    return G

# ----------------------------
# Main loop
# ----------------------------
def main():
    if not MISTRAL_API_KEY:
        raise RuntimeError("Set MISTRAL_API_KEY in your environment.")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = None

    person_ids = fetch_person_ids(conn, limit=50)
    print(f"[info] building KG for {len(person_ids)} people from {DB_PATH}")

    graph_state: Dict[str, Any] = {"nodes": {}, "edges": [], "provenance": []}
    graph_jsonl = (OUT_DIR / "graph.jsonl").open("a", encoding="utf-8")

    for i, pid in enumerate(person_ids, 1):
        pack = fetch_slice_for_person(conn, pid)

        # Build messages
        user_ctx = make_user_context(pack)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_INSTRUCTIONS},
            {"role": "user", "content": f"JSON context:\n{user_ctx}"},
            {"role": "user", "content": "Output the KG JSON now."}
        ]

        # Call LLM
        try:
            raw = call_mistral(messages)
        except Exception as e:
            print(f"[warn] LLM call failed for {pid}: {e}")
            continue

        # Parse + validate + light repair retry
        try:
            patch = json.loads(raw)
        except json.JSONDecodeError:
            # ask the model to fix JSON only once
            fix_msgs = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your previous message was not valid JSON. Return ONLY valid JSON per the schema."}
            ]
            try:
                fixed = call_mistral(fix_msgs)
                patch = json.loads(fixed)
            except Exception as e:
                print(f"[warn] JSON repair failed for {pid}: {e}")
                continue

        ok, errs = is_valid_patch(patch)
        if not ok:
            # one more auto-repair attempt with error hints
            repair_msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "The following JSON failed schema validation. Return a corrected JSON only."},
                {"role": "user", "content": json.dumps(patch, ensure_ascii=False)},
                {"role": "user", "content": "Errors:\n" + "\n".join(errs)}
            ]
            try:
                fixed = call_mistral(repair_msgs)
                patch2 = json.loads(fixed)
                ok2, errs2 = is_valid_patch(patch2)
                if not ok2:
                    print(f"[warn] validation still failing for {pid}: {errs2[:3]}")
                    continue
                patch = patch2
            except Exception as e:
                print(f"[warn] repair call failed for {pid}: {e}")
                continue

        # Write latest patch to file (debugging)
        (OUT_DIR / "latest_patch.json").write_text(json.dumps(patch, indent=2, ensure_ascii=False), encoding="utf-8")

        # Merge into global graph
        graph_state = merge_graph(graph_state, patch)

        # Append to JSONL (audit trail per person)
        graph_jsonl.write(json.dumps({"person_id": pid, "patch": patch}, ensure_ascii=False) + "\n")
        graph_jsonl.flush()

        print(f"[ok] merged person {i}/{len(person_ids)}  id={pid}  nodes={len(patch.get('nodes',[]))} edges={len(patch.get('edges',[]))}")

        # tiny pause to be nice to the API
        time.sleep(0.4)

    graph_jsonl.close()

    # Export final graph as flat CSVs for quick viewing
    nodes_df = pd.DataFrame([{"id": nid, **(node or {})} for nid, node in graph_state["nodes"].items()])
    edges_df = pd.DataFrame(graph_state["edges"])
    prov_df  = pd.DataFrame(graph_state["provenance"])

    nodes_df.to_csv(OUT_DIR / "graph_nodes.csv", index=False)
    edges_df.to_csv(OUT_DIR / "graph_edges.csv", index=False)
    prov_df.to_csv(OUT_DIR / "graph_provenance.csv", index=False)

    (OUT_DIR / "graph_state.json").write_text(json.dumps(graph_state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] graph built → {OUT_DIR.resolve()}")
    print(f"      nodes={len(graph_state['nodes'])} edges={len(graph_state['edges'])} prov={len(graph_state['provenance'])}")

if __name__ == "__main__":
    main()
