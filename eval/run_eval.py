"""Compare retrieval approaches on a notebook and write a Markdown report.

Runs every question through the RAG pipeline once per retrieval mode and records the retrieved
chunks, timings, and answers. Chat history is not touched. Fill in the "Answer quality" column by
hand after reading the answers (or add your own scoring).

Usage (from the repo root, with backend dependencies installed and .env filled in):
    python eval/run_eval.py --notebook-id <id> --questions eval/questions.json \
        --modes vector hybrid_semantic --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from folio import rag  # noqa: E402
from folio.config import RETRIEVAL_MODES  # noqa: E402
from folio.context import get_context  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook-id", required=True)
    parser.add_argument("--questions", default=str(REPO / "eval" / "questions.json"))
    parser.add_argument("--modes", nargs="+", default=["vector", "hybrid_semantic"], choices=RETRIEVAL_MODES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=str(REPO / "eval" / "results"))
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        sys.exit(f"{questions_path} not found. Copy eval/questions.example.json to eval/questions.json and edit it.")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        sys.exit("The questions file must be a JSON array of strings.")

    ctx = get_context()
    notebook = ctx.store.get_notebook(args.notebook_id)
    runs = []
    for qi, question in enumerate(questions, 1):
        for mode in args.modes:
            print(f"Q{qi} [{mode}] {question}")
            result = rag.answer(ctx, args.notebook_id, question, mode=mode, top_k=args.top_k)
            runs.append({"question_number": qi, "question": question, **result})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"eval-{stamp}.json").write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")
    report = out_dir / f"eval-{stamp}.md"
    report.write_text(render(notebook["name"], args, ctx.settings, runs), encoding="utf-8")
    print(f"\nWrote {report}")


def render(notebook_name: str, args, settings, runs: list[dict]) -> str:
    lines = [
        f"# RAG evaluation - {notebook_name}",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Methods compared: {', '.join(args.modes)}",
        f"- Top-k: {args.top_k}",
        f"- Chunking: recursive character, size {settings.chunk_size}, overlap {settings.chunk_overlap}",
        f"- Embedding model: {settings.embedding_model} ({settings.embedding_dimensions} dims)",
        f"- LLM: {settings.llm_model}",
        "",
        "## Summary",
        "",
        "| Question | Method | Retrieved chunks | Retrieval time | Total response time | Answer quality |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        t = run["timings"]
        lines.append(
            f"| Q{run['question_number']} | {run['mode']} | {len(run['chunks'])} | "
            f"{t['retrieval_s']:.2f} s | {t['total_s']:.2f} s | _fill in_ |"
        )

    for mode in args.modes:
        mode_runs = [r for r in runs if r["mode"] == mode]
        if mode_runs:
            avg = sum(r["timings"]["total_s"] for r in mode_runs) / len(mode_runs)
            lines.append(f"\nAverage total response time for **{mode}**: {avg:.2f} s")

    lines += ["", "## Details", ""]
    for run in runs:
        lines += [f"### Q{run['question_number']} - {run['mode']}", "", f"**Question:** {run['question']}", "",
                  "**Retrieved chunks:**", ""]
        for i, chunk in enumerate(run["chunks"], 1):
            where = rag.location(chunk)
            score = chunk.get("reranker_score") if chunk.get("reranker_score") is not None else chunk.get("score")
            snippet = " ".join((chunk.get("content") or "").split())[:220]
            where_text = f" ({where})" if where else ""
            score_text = f" - score {score:.4f}" if isinstance(score, (int, float)) else ""
            lines.append(f"{i}. {chunk.get('source_name')}{where_text}{score_text}  \n   > {snippet}...")
        lines += ["", "**Answer:**", "", run["answer"], ""]

    lines += [
        "## Conclusions",
        "",
        "_Which method retrieved more relevant chunks, how answer quality differed, what the latency cost was,"
        " and which method you selected and why._",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
