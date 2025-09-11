import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from dotenv import load_dotenv
from pathlib import Path

import uuid, json, numpy as np
from decimal import Decimal

# Load .env (your existing logic)
parent_env = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=parent_env) if parent_env.exists() else load_dotenv()

raw_url = os.environ.get("DATABASE_URL")
if not raw_url:
    raise RuntimeError("DATABASE_URL is not set. Add it to your .env or environment.")

# --- Normalize to an explicit driver + ensure sslmode=require ---
# Option A: psycopg2
driver = "postgresql+psycopg2"
# Option B (psycopg3): driver = "postgresql+psycopg"

url = make_url(raw_url)

# If driver isn’t specified, set it
if "+" not in url.drivername:
    # url.drivername is often 'postgresql' or 'postgres'
    base = "postgresql"  # normalize 'postgres' -> 'postgresql'
    url = url.set(drivername=f"{base}+{driver.split('+',1)[1]}")

# Supabase needs SSL; add if missing
query = dict(url.query)
query.setdefault("sslmode", "require")
url = url.set(query=query)

engine = create_engine(url)

# Pull into Pandas DataFrame
# df = pd.read_sql("SELECT id, full_name, detailed_summary FROM people LIMIT 5;", engine)
# print(df)

def _parquet_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to types that pyarrow/pandas can write to parquet safely."""
    out = df.copy()

    for col in out.columns:
        s = out[col]

        # 1) UUIDs -> strings
        if s.dtype == "object" and s.map(lambda v: isinstance(v, uuid.UUID)).any():
            out[col] = s.astype(str)
            continue

        # 2) numpy arrays -> lists (rare but happens)
        if s.dtype == "object" and s.map(lambda v: isinstance(v, np.ndarray)).any():
            out[col] = s.map(lambda v: v.tolist() if isinstance(v, np.ndarray) else v)
            # fall through in case lists/dicts remain

        # 3) dict/list (e.g., JSONB) -> JSON string
        if s.dtype == "object" and s.map(lambda v: isinstance(v, (dict, list))).any():
            out[col] = s.map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
            continue

        # 4) Decimal -> float (or string if you want exactness preserved)
        if s.dtype == "object" and s.map(lambda v: isinstance(v, Decimal)).any():
            out[col] = s.map(lambda v: float(v) if isinstance(v, Decimal) else v)
            continue

        # 5) Optional: force generic object-ish text columns to pandas "string" dtype
        #    This helps avoid mixed-type surprises.
        if s.dtype == "object" and not s.map(lambda v: isinstance(v, (bytes, bytearray, memoryview))).any():
            # only coerce if column looks text-like (heuristic)
            # if you prefer aggressive coercion, just do: out[col] = s.astype("string")
            # We'll be conservative:
            sample = s.dropna().head(50).tolist()
            if all(isinstance(v, str) for v in sample):
                out[col] = s.astype("string")

    return out

def sql_tuple(ids):
    """Return a SQL tuple literal like ('a','b') or ('only_one') safely for UUIDs we fetched ourselves."""
    vals = [f"'{x}'" for x in ids]
    if not vals:
        return "(NULL)"  # will match nothing
    return "(" + ",".join(vals) + ")"

# 1) sample 50 people (already in your code)
people_sample = pd.read_sql("""
    SELECT id
    FROM public.people
    ORDER BY random()
    LIMIT 50;
""", engine)

person_ids = [str(x) for x in people_sample["id"].tolist()]
person_ids_sql = sql_tuple(person_ids)

# 2) pull person-connected tables (reads only)
people = pd.read_sql(f"SELECT * FROM public.people WHERE id IN {person_ids_sql};", engine)

people_features = pd.read_sql(
    f"SELECT * FROM public.people_features WHERE person_id IN {person_ids_sql};",
    engine
)

people_data_points = pd.read_sql(
    f"SELECT * FROM public.people_data_points WHERE person_id IN {person_ids_sql};",
    engine
)

people_events = pd.read_sql(
    f"SELECT * FROM public.people_events WHERE person_id IN {person_ids_sql};",
    engine
)

people_event_sessions = pd.read_sql(
    f"SELECT * FROM public.people_event_sessions WHERE person_id IN {person_ids_sql};",
    engine
)

people_groups = pd.read_sql(
    f"SELECT * FROM public.people_groups WHERE person_id IN {person_ids_sql};",
    engine
)

people_orgs = pd.read_sql(
    f"SELECT * FROM public.people_orgs WHERE person_id IN {person_ids_sql};",
    engine
)

social_connections = pd.read_sql(
    f"""
    SELECT * FROM public.social_connections
    WHERE from_person_id IN {person_ids_sql}
       OR to_person_id   IN {person_ids_sql};
    """,
    engine
)

mutual_fits = pd.read_sql(
    f"""
    SELECT * FROM public.mutual_fits
    WHERE from_person_id IN {person_ids_sql}
       OR to_person_id   IN {person_ids_sql};
    """,
    engine
)

import_batch_requests = pd.read_sql(
    f"SELECT * FROM public.import_batch_requests WHERE person_id IN {person_ids_sql};",
    engine
)

profile_update_log = pd.read_sql(
    f"SELECT * FROM public.profile_update_log WHERE person_id IN {person_ids_sql};",
    engine
)

personal_contacts = pd.read_sql(
    f"SELECT * FROM public.personal_contacts WHERE person_id IN {person_ids_sql};",
    engine
)

vibe_invitation_records = pd.read_sql(
    f"SELECT * FROM public.vibe_invitation_records WHERE person_id IN {person_ids_sql};",
    engine
)

# 3) secondary tables tied via the above (still read-only)
# people_features_data_points joins through people_features
people_features_data_points = pd.read_sql(
    f"""
    SELECT pfd.*
    FROM public.people_features_data_points pfd
    JOIN public.people_features pf ON pf.pf_id = pfd.pf_id
    WHERE pf.person_id IN {person_ids_sql};
    """,
    engine
)

data_points = pd.read_sql(
    f"SELECT * FROM public.data_points WHERE original_person_id IN {person_ids_sql};",
    engine
)

# 4) parents referenced by the slices (orgs, events, sessions, features_schema)
org_ids = set(people_orgs.get("org_id", pd.Series(dtype=str)).dropna().astype(str))
event_ids = set(people_events.get("event_id", pd.Series(dtype=str)).dropna().astype(str))
session_ids = set(people_event_sessions.get("session_id", pd.Series(dtype=str)).dropna().astype(str))
feature_ids = set(people_features.get("feature_id", pd.Series(dtype=str)).dropna().astype(str))

orgs = pd.read_sql(
    f"SELECT * FROM public.orgs WHERE id IN {sql_tuple(sorted(org_ids))};", engine
) if org_ids else pd.DataFrame()

events = pd.read_sql(
    f"SELECT * FROM public.events WHERE id IN {sql_tuple(sorted(event_ids))};", engine
) if event_ids else pd.DataFrame()

event_sessions = pd.read_sql(
    f"SELECT * FROM public.event_sessions WHERE id IN {sql_tuple(sorted(session_ids))};", engine
) if session_ids else pd.DataFrame()

features_schema = pd.read_sql(
    f"SELECT * FROM public.features_schema WHERE feature_id IN {sql_tuple(sorted(feature_ids))};", engine
) if feature_ids else pd.DataFrame()

# (Optional) if events carry org_id you also want:
extra_org_ids = set(events.get("org_id", pd.Series(dtype=str)).dropna().astype(str)) - org_ids
if extra_org_ids:
    more_orgs = pd.read_sql(
        f"SELECT * FROM public.orgs WHERE id IN {sql_tuple(sorted(extra_org_ids))};", engine
    )
    orgs = pd.concat([orgs, more_orgs], ignore_index=True).drop_duplicates()

# quick sanity prints
for name, df in [
    ("people", people),
    ("people_features", people_features),
    ("people_data_points", people_data_points),
    ("people_events", people_events),
    ("people_event_sessions", people_event_sessions),
    ("people_groups", people_groups),
    ("people_orgs", people_orgs),
    ("social_connections", social_connections),
    ("mutual_fits", mutual_fits),
    ("import_batch_requests", import_batch_requests),
    ("profile_update_log", profile_update_log),
    ("personal_contacts", personal_contacts),
    ("vibe_invitation_records", vibe_invitation_records),
    ("people_features_data_points", people_features_data_points),
    ("data_points", data_points),
    ("orgs", orgs),
    ("events", events),
    ("event_sessions", event_sessions),
    ("features_schema", features_schema),
]:
    print(f"{name:28s} rows={len(df)}")


# Directory for parquet + sqlite outputs
outdir = Path("../mock_dataset")
outdir.mkdir(exist_ok=True)

# Collect all the frames into a dict
frames = {
    "people": people,
    "people_features": people_features,
    "people_data_points": people_data_points,
    "people_events": people_events,
    "people_event_sessions": people_event_sessions,
    "people_groups": people_groups,
    "people_orgs": people_orgs,
    "social_connections": social_connections,
    "mutual_fits": mutual_fits,
    "import_batch_requests": import_batch_requests,
    "profile_update_log": profile_update_log,
    "personal_contacts": personal_contacts,
    "vibe_invitation_records": vibe_invitation_records,
    "people_features_data_points": people_features_data_points,
    "data_points": data_points,
    "orgs": orgs,
    "events": events,
    "event_sessions": event_sessions,
    "features_schema": features_schema,
}

# --- 1) Save to parquet ---
for name, df in frames.items():
    df = _parquet_friendly(df)
    df.to_parquet(outdir / f"{name}.parquet", index=False)
    print(f"[ok] wrote {name}.parquet ({len(df)} rows)")

# --- 2) Save to a SQLite mock DB ---
sqlite_path = outdir / "mock_people.db"
sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")

for name, df in frames.items():
    if df.shape[1] == 0:  # zero columns
        print(f"[skip] {name}: no columns (would create invalid table)")
        continue
    df_clean = _parquet_friendly(df)
    df_clean.to_sql(name, sqlite_engine, if_exists="replace", index=False)
    print(f"[ok] wrote {name} table to sqlite ({len(df_clean)} rows)")

print(f"\n[done] mock dataset available in {outdir.resolve()}")
