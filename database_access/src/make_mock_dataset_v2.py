#!/usr/bin/env python3
import os
import json
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from dotenv import load_dotenv


# ---------- .env loading ----------
parent_env = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=parent_env) if parent_env.exists() else load_dotenv()


# ---------- Helpers ----------
def _to_jsonable(obj):
    """Safe JSON serializer for UUID, Decimal, numpy types, etc."""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    # Fallback
    try:
        return str(obj)
    except Exception:
        return None


def _sql_tuple(ids: List[str]) -> str:
    """Return SQL tuple literal like ('a','b'). If empty, returns (NULL) to match nothing."""
    vals = [f"'{x}'" for x in ids]
    if not vals:
        return "(NULL)"
    return "(" + ",".join(vals) + ")"


def _parquet_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to parquet-safe types."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        s = out[col]
        # UUIDs -> str
        if s.dtype == "object" and s.map(lambda v: isinstance(v, uuid.UUID)).any():
            out[col] = s.astype(str)
            continue
        # numpy array -> list
        if s.dtype == "object" and s.map(lambda v: isinstance(v, np.ndarray)).any():
            out[col] = s.map(lambda v: v.tolist() if isinstance(v, np.ndarray) else v)
        # dict/list -> json string
        if s.dtype == "object" and s.map(lambda v: isinstance(v, (dict, list))).any():
            out[col] = s.map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
            continue
        # Decimal -> float
        if s.dtype == "object" and s.map(lambda v: isinstance(v, Decimal)).any():
            out[col] = s.map(lambda v: float(v) if isinstance(v, Decimal) else v)
            continue
        # plain object -> string dtype if looks like text
        if s.dtype == "object" and not s.map(lambda v: isinstance(v, (bytes, bytearray, memoryview))).any():
            sample = s.dropna().head(50).tolist()
            if all(isinstance(v, str) for v in sample):
                out[col] = s.astype("string")
    return out


def _sample_people(engine, limit: int = 50, seed: Optional[float] = None) -> pd.DataFrame:
    """
    Sample people ids with optional deterministic random seed.
    """
    if seed is not None:
        seed_val = float(seed) % 1.0
        q = f"""
            SELECT id
            FROM public.people
            ORDER BY (
                SELECT setseed({seed_val})
            ), random()
            LIMIT {int(limit)};
        """
    else:
        q = f"""
            SELECT id
            FROM public.people
            ORDER BY random()
            LIMIT {int(limit)};
        """
    return pd.read_sql(q, engine)


def _fetch_core_frames(engine, person_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Fetch ONLY the three core tables: people, people_data_points, data_points
    limited to the chosen people (via people_data_points linkage).
    """
    frames: Dict[str, pd.DataFrame] = {}
    ids_sql = _sql_tuple(person_ids)

    # people
    frames["people"] = pd.read_sql(
        f"SELECT * FROM public.people WHERE id IN {ids_sql};", engine
    )

    # people_data_points
    frames["people_data_points"] = pd.read_sql(
        f"""
        SELECT *
        FROM public.people_data_points
        WHERE person_id IN {ids_sql};
        """,
        engine,
    )

    # data_points (only those referenced in people_data_points)
    dp_ids = set(
        frames["people_data_points"]["data_point_id"].dropna().astype(str)
    ) if not frames["people_data_points"].empty else set()

    if dp_ids:
        dp_sql = _sql_tuple(sorted(dp_ids))
        frames["data_points"] = pd.read_sql(
            f"SELECT * FROM public.data_points WHERE id IN {dp_sql};", engine
        )
    else:
        frames["data_points"] = pd.DataFrame(
            columns=[
                "id",
                "source_type",
                "source_url",
                "collected_at",
                "embedding",
                "source_json",
                "retrieval_method",
                "retrieval_endpoint",
                "embedding_type",
                "agent_message",
                "original_person_id",
                "org_id",
                "import_id",
                "version",
                "content",
                "process_status",
                "source_json_hash",
                "extra_data_hash",
            ]
        )

    return frames


def _build_people_records(
    people: pd.DataFrame,
    pdp: pd.DataFrame,
    dps: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Returns a list of flat person records:

    {
      "person_id": "...",
      "full_name": "...",
      "primary_phone_number": "...",
      "data_points": [
        {"data_point_id": "...", "source_json": {...}},
        ...
      ]
    }
    """
    # Normalize ids
    if "id" in people.columns:
        people["id"] = people["id"].astype(str)
    if "data_point_id" in pdp.columns:
        pdp["data_point_id"] = pdp["data_point_id"].astype(str)
    if "person_id" in pdp.columns:
        pdp["person_id"] = pdp["person_id"].astype(str)
    if "id" in dps.columns:
        dps["id"] = dps["id"].astype(str)

    # Build data_point lookup
    dp_map = dps.set_index("id").to_dict(orient="index")

    # Group dp_ids by person_id
    dp_by_person: Dict[str, List[str]] = {}
    if not pdp.empty:
        for pid, group in pdp.groupby("person_id"):
            dp_by_person[pid] = group["data_point_id"].tolist()

    # Build flat records
    records: List[Dict[str, Any]] = []
    for _, prow in people.iterrows():
        pid = prow["id"]
        rec: Dict[str, Any] = {
            "person_id": pid,
            "full_name": prow.get("full_name"),
            "primary_phone_number": prow.get("primary_phone_number"),
            "data_points": [],
        }
        for dpid in dp_by_person.get(pid, []):
            dp_row = dp_map.get(dpid)
            if dp_row is None:
                continue
            rec["data_points"].append({
                "data_point_id": dpid,
                "source_json": dp_row.get("source_json"),
            })
        records.append(rec)

    return records


def _write_debug_parquets(frames: Dict[str, pd.DataFrame], outdir: Path, verbose: bool = True):
    """Optional: write the three core frames as parquet for inspection."""
    outdir.mkdir(parents=True, exist_ok=True)
    for name in ("people", "people_data_points", "data_points"):
        df = frames[name]
        df2 = _parquet_friendly(df)
        df2.to_parquet(outdir / f"{name}.parquet", index=False)
        if verbose:
            print(f"[ok] wrote {name}.parquet ({len(df2)} rows)")


def build_people_mock_json(
    *,
    seed: Optional[float] = None,
    limit: int = 50,
    outdir: os.PathLike | str = "../mock_dataset",
    json_name: str = "mock_people.json",
    ndjson_name: str = "mock_people.ndjson",
    database_url: Optional[str] = None,
    include_debug_parquets: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Build the mock people JSON:
      - sample N people
      - read only people, people_data_points, data_points
      - write an array of flat records to mock_people.json
      - write one flat record per line to mock_people.ndjson
    """
    # --- engine ---
    driver = "postgresql+psycopg2"
    raw_url = database_url or os.environ.get("DATABASE_URL")
    if not raw_url:
        raise RuntimeError("DATABASE_URL not set (and --database-url not provided).")

    url = make_url(raw_url)
    if "+" not in url.drivername:
        base = "postgresql"
        url = url.set(drivername=f"{base}+{driver.split('+',1)[1]}")
    query = dict(url.query)
    query.setdefault("sslmode", "require")
    url = url.set(query=query)
    engine = create_engine(url)

    # --- sample & fetch three frames ---
    sample_df = _sample_people(engine, limit=limit, seed=seed)
    person_ids = [str(x) for x in sample_df["id"].tolist()]
    if verbose:
        print(f"[info] sampled {len(person_ids)} people")

    frames = _fetch_core_frames(engine, person_ids)

    # --- build flat records ---
    people_records = _build_people_records(
        frames["people"], frames["people_data_points"], frames["data_points"]
    )

    # --- write outputs ---
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    json_path = outdir_path / json_name
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(people_records, f, ensure_ascii=False, indent=2, default=_to_jsonable)
    if verbose:
        print(f"[ok] wrote {json_path.resolve()}")

    ndjson_path = outdir_path / ndjson_name
    with ndjson_path.open("w", encoding="utf-8") as f:
        for rec in people_records:
            f.write(json.dumps(rec, ensure_ascii=False, default=_to_jsonable) + "\n")
    if verbose:
        print(f"[ok] wrote {ndjson_path.resolve()}")

    if include_debug_parquets:
        _write_debug_parquets(frames, outdir_path, verbose=verbose)

    if verbose:
        print(f"[done] mock dataset available in {outdir_path.resolve()}")

    return outdir_path.resolve()


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build a people-centric mock JSON from three core tables.")
    p.add_argument("--seed", type=float, default=None, help="Deterministic sampling seed in [0,1). Example: 0.42")
    p.add_argument("--limit", type=int, default=50, help="Number of people to sample (default: 50)")
    p.add_argument("--outdir", type=str, default="../mock_dataset", help="Output directory (default: ../mock_dataset)")
    p.add_argument("--database-url", type=str, default=None, help="Override DATABASE_URL env var")
    p.add_argument("--json-name", type=str, default="mock_people.json", help="Output JSON filename")
    p.add_argument("--ndjson-name", type=str, default="mock_people.ndjson", help="Output NDJSON filename")
    p.add_argument("--include-debug-parquets", action="store_true", help="Also write people/data_points/pdp parquet")
    p.add_argument("--quiet", action="store_true", help="Suppress progress prints")
    args = p.parse_args()

    build_people_mock_json(
        seed=args.seed,
        limit=args.limit,
        outdir=args.outdir,
        json_name=args.json_name,
        ndjson_name=args.ndjson_name,
        database_url=args.database_url,
        include_debug_parquets=args.include_debug_parquets,
        verbose=not args.quiet,
    )
