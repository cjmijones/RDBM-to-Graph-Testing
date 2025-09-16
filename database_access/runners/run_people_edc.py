# runners/run_people_edc.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

from edc.edc_framework import EDC  # <-- USES YOUR EXISTING EDC PACKAGE (unchanged)
from adapters.people_doc_builder import build_docs_from_people

def to_nodes_edges(canon: List[List[List[str]]],
                   ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = set()
    edges = []
    for row_id, triples in zip(ids, canon):
        for triple in triples:
            if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                continue
            subj, rel, obj = triple[:3]
            if not subj or not rel or not obj:
                continue
            nodes.add(subj); nodes.add(obj)
            edges.append({"source": subj, "relation": rel, "target": obj, "person_id": row_id})
    return pd.DataFrame(sorted(nodes), columns=["node"]), pd.DataFrame(edges)

def load_people_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    people = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            people.append(json.loads(line))
    return people

def main():
    ap = argparse.ArgumentParser(description="Run EDC over person JSON with dynamic-schema adapters.")
    ap.add_argument("--jsonl", type=str, required=True, help="Path to people JSONL (one person record per line).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for EDC artifacts.")
    ap.add_argument("--schema_csv", type=str, default="schemas/people_schema.csv",
                    help="Target schema CSV (can be empty header to start).")
    ap.add_argument("--enrich_schema", action="store_true", help="Allow SD to propose new relations.")
    ap.add_argument("--refinement_iterations", type=int, default=1)
    ap.add_argument("--cap_open_facts", type=int, default=500)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load input
    people = load_people_from_jsonl(Path(args.jsonl))
    ids = [str(p.get("person_id", f"row{i}")) for i, p in enumerate(people)]

    # 2) Build documents (precise facts + open:* facts)
    docs = build_docs_from_people(people, use_precise=True, cap_open_facts=args.cap_open_facts)

    # 3) Configure EDC with your prompt templates and few-shots
    edc = EDC(
        # Models (swap to gpt-* if you prefer API models)
        oie_llm="mistralai/Mistral-7B-Instruct-v0.2",
        sd_llm="mistralai/Mistral-7B-Instruct-v0.2",
        sc_llm="mistralai/Mistral-7B-Instruct-v0.2",
        sc_embedder="intfloat/e5-mistral-7b-instruct",
        sr_embedder="intfloat/e5-mistral-7b-instruct",

        # Prompts & few-shots
        oie_prompt_template_file_path="prompt_templates/oie_template.txt",
        oie_few_shot_example_file_path="few_shot_examples/example/oie_few_shot_examples.txt",

        sd_prompt_template_file_path="prompt_templates/sd_template.txt",
        sd_few_shot_example_file_path="few_shot_examples/example/sd_few_shot_examples.txt",

        sc_prompt_template_file_path="prompt_templates/sc_template.txt",

        oie_refine_prompt_template_file_path="prompt_templates/oie_r_template.txt",
        oie_refine_few_shot_example_file_path="few_shot_examples/example/oie_few_shot_refine_examples.txt",

        ee_prompt_template_file_path="prompt_templates/ee_template.txt",
        ee_few_shot_example_file_path="few_shot_examples/example/ee_few_shot_examples.txt",
        em_prompt_template_file_path="prompt_templates/em_template.txt",

        # IO
        input_text_file_path="datasets/example.txt",   # unused in programmatic flow
        output_dir=str(out_dir),

        # Schema behavior
        target_schema_path=args.schema_csv,
        enrich_schema=args.enrich_schema,

        # Hints/aux
        include_relation_example="self",
        refinement_iterations=args.refinement_iterations,
        loglevel=None,
    )

    # 4) Run EDC extraction
    canon = edc.extract_kg(docs, str(out_dir), refinement_iterations=args.refinement_iterations)

    # 5) Project to nodes / edges for graph load
    nodes_df, edges_df = to_nodes_edges(canon, ids)
    nodes_df.to_csv(out_dir / "nodes.csv", index=False)
    edges_df.to_csv(out_dir / "edges.csv", index=False)

    # Also keep the documents and input IDs for reproducibility
    (out_dir / "docs.txt").write_text("\n\n---\n\n".join(docs), encoding="utf-8")
    (out_dir / "ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")

    print(f"[ok] EDC run complete. Wrote {len(nodes_df)} nodes and {len(edges_df)} edges to {out_dir}")

if __name__ == "__main__":
    main()
