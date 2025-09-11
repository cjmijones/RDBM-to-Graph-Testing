#!/usr/bin/env python3
import os
import uuid
import json
from decimal import Decimal
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from dotenv import load_dotenv


# ---------- .env loading (unchanged logic) ----------
parent_env = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=parent_env) if parent_env.exists() else load_dotenv()


# ---------- Helpers ----------
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
        if s.dtype == "object" and not s.map(lambda v: isinstance(v, (bytes, bytearray, memoryview))).any():
            sample = s.dropna().head(50).tolist()
            if all(isinstance(v, str) for v in sample):
                out[col] = s.astype("string")

    return out


def _sql_tuple(ids) -> str:
    """Return a SQL tuple literal like ('a','b') or ('only_one'). If empty, returns (NULL) to match nothing."""
    vals = [f"'{x}'" for x in ids]
    if not vals:
        return "(NULL)"
    return "(" + ",".join(vals) + ")"


def _sample_people(engine, limit: int = 50, seed: Optional[float] = None) -> pd.DataFrame:
    """
    Sample people from the database with optional deterministic random seed.
    If seed is provided, the same set will be returned for the same seed.
    """
    if seed is not None:
        # Ensure seed in [0,1). cast to float; modulo keeps it in range if caller passes something odd.
        seed_val = float(seed) % 1.0
        query = f"""
            SELECT id
            FROM public.people
            ORDER BY (
                SELECT setseed({seed_val})
            ), random()
            LIMIT {int(limit)};
        """
    else:
        query = f"""
            SELECT id
            FROM public.people
            ORDER BY random()
            LIMIT {int(limit)};
        """
    return pd.read_sql(query, engine)


def _fetch_frames(engine, person_ids_sql: str) -> Dict[str, pd.DataFrame]:
    """Fetch all related tables and return as a dict of DataFrames (mirrors your original pulls)."""
    frames: Dict[str, pd.DataFrame] = {}

    # Direct person-tied tables
    frames["people"] = pd.read_sql(f"SELECT * FROM public.people WHERE id IN {person_ids_sql};", engine)
    frames["people_features"] = pd.read_sql(
        f"SELECT * FROM public.people_features WHERE person_id IN {person_ids_sql};", engine
    )
    frames["people_data_points"] = pd.read_sql(
        f"SELECT * FROM public.people_data_points WHERE person_id IN {person_ids_sql};", engine
    )
    frames["people_events"] = pd.read_sql(
        f"SELECT * FROM public.people_events WHERE person_id IN {person_ids_sql};", engine
    )
    frames["people_event_sessions"] = pd.read_sql(
        f"SELECT * FROM public.people_event_sessions WHERE person_id IN {person_ids_sql};", engine
    )
    frames["people_groups"] = pd.read_sql(
        f"SELECT * FROM public.people_groups WHERE person_id IN {person_ids_sql};", engine
    )
    frames["people_orgs"] = pd.read_sql(
        f"SELECT * FROM public.people_orgs WHERE person_id IN {person_ids_sql};", engine
    )
    frames["social_connections"] = pd.read_sql(
        f"""
        SELECT * FROM public.social_connections
        WHERE from_person_id IN {person_ids_sql}
           OR to_person_id   IN {person_ids_sql};
        """,
        engine,
    )
    frames["mutual_fits"] = pd.read_sql(
        f"""
        SELECT * FROM public.mutual_fits
        WHERE from_person_id IN {person_ids_sql}
           OR to_person_id   IN {person_ids_sql};
        """,
        engine,
    )
    frames["import_batch_requests"] = pd.read_sql(
        f"SELECT * FROM public.import_batch_requests WHERE person_id IN {person_ids_sql};", engine
    )
    frames["profile_update_log"] = pd.read_sql(
        f"SELECT * FROM public.profile_update_log WHERE person_id IN {person_ids_sql};", engine
    )
    frames["personal_contacts"] = pd.read_sql(
        f"SELECT * FROM public.personal_contacts WHERE person_id IN {person_ids_sql};", engine
    )
    frames["vibe_invitation_records"] = pd.read_sql(
        f"SELECT * FROM public.vibe_invitation_records WHERE person_id IN {person_ids_sql};", engine
    )

    # Secondary via join
    frames["people_features_data_points"] = pd.read_sql(
        f"""
        SELECT pfd.*
        FROM public.people_features_data_points pfd
        JOIN public.people_features pf ON pf.pf_id = pfd.pf_id
        WHERE pf.person_id IN {person_ids_sql};
        """,
        engine,
    )

    frames["data_points"] = pd.read_sql(
        f"SELECT * FROM public.data_points WHERE original_person_id IN {person_ids_sql};", engine
    )

    # Parent IDs
    people_orgs = frames["people_orgs"]
    people_events = frames["people_events"]
    people_event_sessions = frames["people_event_sessions"]
    people_features = frames["people_features"]

    org_ids = set(people_orgs.get("org_id", pd.Series(dtype=str)).dropna().astype(str))
    event_ids = set(people_events.get("event_id", pd.Series(dtype=str)).dropna().astype(str))
    session_ids = set(people_event_sessions.get("session_id", pd.Series(dtype=str)).dropna().astype(str))
    feature_ids = set(people_features.get("feature_id", pd.Series(dtype=str)).dropna().astype(str))

    frames["orgs"] = (
        pd.read_sql(f"SELECT * FROM public.orgs WHERE id IN {_sql_tuple(sorted(org_ids))};", engine)
        if org_ids
        else pd.DataFrame()
    )
    frames["events"] = (
        pd.read_sql(f"SELECT * FROM public.events WHERE id IN {_sql_tuple(sorted(event_ids))};", engine)
        if event_ids
        else pd.DataFrame()
    )
    frames["event_sessions"] = (
        pd.read_sql(
            f"SELECT * FROM public.event_sessions WHERE id IN {_sql_tuple(sorted(session_ids))};", engine
        )
        if session_ids
        else pd.DataFrame()
    )
    frames["features_schema"] = (
        pd.read_sql(
            f"SELECT * FROM public.features_schema WHERE feature_id IN {_sql_tuple(sorted(feature_ids))};",
            engine,
        )
        if feature_ids
        else pd.DataFrame()
    )

    # If events carry org_id, include any extra orgs
    events_df = frames["events"]
    if not events_df.empty and "org_id" in events_df.columns:
        extra_org_ids = set(events_df["org_id"].dropna().astype(str)) - org_ids
        if extra_org_ids:
            more_orgs = pd.read_sql(
                f"SELECT * FROM public.orgs WHERE id IN {_sql_tuple(sorted(extra_org_ids))};", engine
            )
            frames["orgs"] = (
                pd.concat([frames["orgs"], more_orgs], ignore_index=True).drop_duplicates()
                if not frames["orgs"].empty
                else more_orgs
            )

    return frames


def _write_outputs(frames: Dict[str, pd.DataFrame], outdir: Path, verbose: bool = True) -> Tuple[Path, Path]:
    """Write parquet files and a SQLite DB. Returns (outdir, sqlite_path)."""
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Parquet
    for name, df in frames.items():
        df2 = _parquet_friendly(df)
        (outdir / f"{name}.parquet").parent.mkdir(parents=True, exist_ok=True)
        df2.to_parquet(outdir / f"{name}.parquet", index=False)
        if verbose:
            print(f"[ok] wrote {name}.parquet ({len(df2)} rows)")

    # 2) SQLite
    sqlite_path = outdir / "mock_people.db"
    sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")

    for name, df in frames.items():
        if df.shape[1] == 0:
            if verbose:
                print(f"[skip] {name}: no columns (would create invalid table)")
            continue
        df_clean = _parquet_friendly(df)
        df_clean.to_sql(name, sqlite_engine, if_exists="replace", index=False)
        if verbose:
            print(f"[ok] wrote {name} table to sqlite ({len(df_clean)} rows)")

    return outdir, sqlite_path


# ---------- Exportable main function ----------
def build_mock_dataset(
    *,
    seed: Optional[float] = None,
    outdir: os.PathLike | str = "../mock_dataset",
    limit: int = 50,
    database_url: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    """
    Build a mock dataset by sampling people and exporting related tables to Parquet and a SQLite DB.

    Parameters
    ----------
    seed : Optional[float]
        Deterministic sampling seed for ORDER BY random(). Use a float in [0,1). If None, sampling is non-deterministic.
    outdir : PathLike | str
        Output directory to write Parquet files and 'mock_people.db'.
    limit : int
        Number of people to sample (default 50).
    database_url : Optional[str]
        Override for DATABASE_URL. If not provided, reads from environment (.env already loaded).
    verbose : bool
        If True, prints progress messages.

    Returns
    -------
    Path
        Absolute path to the output directory.
    """
    # Prepare engine
    driver = "postgresql+psycopg2"
    raw_url = database_url or os.environ.get("DATABASE_URL")
    url = make_url(raw_url)

    if "+" not in url.drivername:
        base = "postgresql"
        url = url.set(drivername=f"{base}+{driver.split('+',1)[1]}")

    query = dict(url.query)
    query.setdefault("sslmode", "require")
    url = url.set(query=query)
    engine = create_engine(url)

    # Sample IDs
    people_sample = _sample_people(engine, limit=limit, seed=seed)
    person_ids = [str(x) for x in people_sample["id"].tolist()]
    person_ids_sql = _sql_tuple(person_ids)

    # Fetch all frames
    frames = _fetch_frames(engine, person_ids_sql)

    if verbose:
        for name, df in frames.items():
            print(f"{name:28s} rows={len(df)}")

    # Write outputs
    outdir_path = Path(outdir)
    outdir_path, sqlite_path = _write_outputs(frames, outdir_path, verbose=verbose)

    if verbose:
        print(f"\n[done] mock dataset available in {outdir_path.resolve()}")
        print(f"[info] SQLite DB: {sqlite_path.resolve()}")

    return outdir_path.resolve()


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a mock dataset from a Postgres DB (e.g., Supabase).")
    parser.add_argument(
        "--seed",
        type=float,
        default=None,
        help="Deterministic sampling seed in [0,1). Example: 0.42",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="../mock_dataset",
        help="Output directory for Parquet files and SQLite DB (default: ../mock_dataset)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of people to sample (default: 50)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Override DATABASE_URL; if omitted, reads from environment/.env",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress prints",
    )

    args = parser.parse_args()

    build_mock_dataset(
        seed=args.seed,
        outdir=args.outdir,
        limit=args.limit,
        database_url=args.database_url,
        verbose=not args.quiet,
    )
