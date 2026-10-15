"""Compare retrieval approaches on a notebook and write a Markdown report.

Runs every question through the RAG pipeline once per retrieval mode and records the retrieved chunks,
timings, and answers. Chat history is not touched.

Each question can say what *should* be retrieved, which lets the report score retrieval automatically:
    {"question": "...", "expected_source": "lecture3", "expected_text": "photosynthesis"}
A chunk counts as relevant when its source name contains `expected_source` and its text contains
`expected_text` (case-insensitive; give either or both). Plain strings work too, without scoring.
Answer quality is still judged by hand: fill in the "Answer quality" column after reading the answers.

Usage (from the repo root, with backend dependencies installed and .env filled in):
    python eval/run_eval.py --notebook-id <id> --modes vector hybrid_semantic --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from folio import rag  # noqa: E402
from folio.config import RETRIEVAL_MODES  # noqa: E402
from folio.context import get_context  # noqa: E402


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        sys.exit(f"{path} not found. Copy eval/questions.example.json to eval/questions.json and edit it.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        sys.exit("The questions file must be a non-empty JSON array.")
    questions = []
    for i, item in enumerate(raw, 1):
        if isinstance(item, str):
            item = {"question": item}
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            sys.exit(f"Question {i} must be a string or an object with a 'question' field.")
        questions.append(item)
    return questions


def has_expectation(q: dict[str, Any]) -> bool:
    return bool(q.get("expected_source") or q.get("expected_text"))


def is_relevant(chunk: dict[str, Any], q: dict[str, Any]) -> bool:
    source = str(q.get("expected_source") or "").lower()
    text = str(q.get("expected_text") or "").lower()
    if source and source not in (chunk.get("source_name") or "").lower():
        return False
    if text and text not in (chunk.get("content") or "").lower():
        return False
    return bool(source or text)


def score_run(chunks: list[dict[str, Any]], q: dict[str, Any]) -> dict[str, Any]:
    if not has_expectation(q):
        return {"scored": False, "relevant": [], "relevant_count": None, "first_rank": None}
    relevant = [is_relevant(c, q) for c in chunks]
    first = next((i for i, r in enumerate(relevant, 1) if r), None)
    return {"scored": True, "relevant": relevant, "relevant_count": sum(relevant), "first_rank": first}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook-id", required=True)
    parser.add_argument("--questions", default=str(REPO / "eval" / "questions.json"))
    parser.add_argument("--modes", nargs="+", default=["vector", "hybrid_semantic"], choices=RETRIEVAL_MODES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=str(REPO / "eval" / "results"))
    parser.add_argument("--rerender", metavar="RESULTS_JSON",
                        help="rebuild the Markdown report from a saved results .json without calling the models again")
    args = parser.parse_args()

    if args.rerender:
        rerender(Path(args.rerender), args)
        return

    questions = load_questions(Path(args.questions))
    ctx = get_context()
    notebook = ctx.store.get_notebook(args.notebook_id)
    runs = []
    for qi, q in enumerate(questions, 1):
        for mode in args.modes:
            print(f"Q{qi} [{mode}] {q['question']}")
            result = rag.answer(ctx, args.notebook_id, q["question"], mode=mode, top_k=args.top_k)
            runs.append({"question_number": qi, **q, **result, "retrieval_score": score_run(result["chunks"], q)})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trim_source_text(runs)
    (out_dir / f"eval-{stamp}.json").write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")
    report = out_dir / f"eval-{stamp}.md"
    report.write_text(render(notebook["name"], args, ctx.settings, runs), encoding="utf-8")
    print(f"\nWrote {report}")


SNIPPET_CHARS = 220


def snippet(text: str | None) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rstrip() + "..."


def trim_source_text(runs: list[dict[str, Any]]) -> None:
    """Keep only short snippets of source text in saved results. Sources can be copyrighted, and results are
    committed to a public repo; relevance is already scored, so the full text isn't needed afterwards."""
    for run in runs:
        for chunk in run.get("chunks", []):
            chunk["content"] = snippet(chunk.get("content"))
        for citation in run.get("citations", []):
            citation["snippet"] = snippet(citation.get("snippet"))


def rerender(results_json: Path, args) -> None:
    runs = json.loads(results_json.read_text(encoding="utf-8"))
    args.modes = list(dict.fromkeys(r["mode"] for r in runs))  # keep the original order
    args.top_k = max(len(r["chunks"]) for r in runs) or args.top_k
    ctx = get_context()
    notebook = ctx.store.get_notebook(args.notebook_id)
    try:  # results files are named eval-YYYYmmdd-HHMMSS.json
        run_date = datetime.strptime(results_json.stem.removeprefix("eval-"), "%Y%m%d-%H%M%S")
    except ValueError:
        run_date = datetime.fromtimestamp(results_json.stat().st_mtime)
    report = results_json.with_suffix(".md")
    report.write_text(render(notebook["name"], args, ctx.settings, runs, run_date), encoding="utf-8")
    print(f"Rewrote {report}")


def mode_summary(runs: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    mode_runs = [r for r in runs if r["mode"] == mode]
    scored = [r["retrieval_score"] for r in mode_runs if r["retrieval_score"]["scored"]]
    n = len(mode_runs)
    return {
        "questions": n,
        "scored": len(scored),
        "hit_rate": sum(1 for s in scored if s["first_rank"]) / len(scored) if scored else None,
        "mrr": sum(1 / s["first_rank"] for s in scored if s["first_rank"]) / len(scored) if scored else None,
        "avg_relevant": sum(s["relevant_count"] for s in scored) / len(scored) if scored else None,
        "avg_retrieval_s": sum(r["timings"]["retrieval_s"] for r in mode_runs) / n,
        "avg_total_s": sum(r["timings"]["total_s"] for r in mode_runs) / n,
    }


def fmt(value: float | None, pattern: str) -> str:
    return "n/a" if value is None else pattern.format(value)


def render(notebook_name: str, args, settings, runs: list[dict[str, Any]], run_date: datetime | None = None) -> str:
    lines = [
        f"# RAG evaluation - {notebook_name}",
        "",
        f"- Date: {run_date or datetime.now():%Y-%m-%d %H:%M}",
        f"- Methods compared: {', '.join(args.modes)}",
        f"- Top-k: {args.top_k}",
        f"- Chunking: recursive character, size {settings.chunk_size}, overlap {settings.chunk_overlap}",
        f"- Embedding model: {settings.embedding_model} ({settings.embedding_dimensions} dims)",
        f"- LLM: {settings.llm_model}",
        "",
        "## Method comparison",
        "",
        "Hit rate = share of scored questions where at least one relevant chunk was retrieved. "
        "MRR = mean reciprocal rank of the first relevant chunk (1.0 = always ranked first).",
        "",
        "| Method | Hit rate | MRR | Avg relevant chunks | Avg retrieval time | Avg total response time |",
        "|---|---|---|---|---|---|",
    ]
    for mode in args.modes:
        s = mode_summary(runs, mode)
        lines.append(
            f"| {mode} | {fmt(s['hit_rate'], '{:.0%}')} | {fmt(s['mrr'], '{:.2f}')} | "
            f"{fmt(s['avg_relevant'], '{:.1f}')} / {args.top_k} | {s['avg_retrieval_s']:.2f} s | {s['avg_total_s']:.2f} s |"
        )

    lines += [
        "",
        "## Per question",
        "",
        "| Question | Method | Retrieved chunks | Relevant | First relevant rank | Retrieval time | Total response time | Answer quality |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        t, s = run["timings"], run["retrieval_score"]
        relevant = f"{s['relevant_count']}" if s["scored"] else "n/a"
        first = (str(s["first_rank"]) if s["first_rank"] else "none") if s["scored"] else "n/a"
        lines.append(
            f"| Q{run['question_number']} | {run['mode']} | {len(run['chunks'])} | {relevant} | {first} | "
            f"{t['retrieval_s']:.2f} s | {t['total_s']:.2f} s | _fill in_ |"
        )

    lines += ["", "## Questions", ""]
    seen = set()
    for run in runs:
        if run["question_number"] in seen:
            continue
        seen.add(run["question_number"])
        expect = ", ".join(f"{k.replace('expected_', '')} contains \"{run[k]}\""
                           for k in ("expected_source", "expected_text") if run.get(k))
        note = f" - _{run['note']}_" if run.get("note") else ""
        lines.append(f"{run['question_number']}. {run['question']}" + (f"  \n   Expected: {expect}" if expect else "") + note)

    lines += ["", "## Details", ""]
    for run in runs:
        lines += [f"### Q{run['question_number']} - {run['mode']}", "", f"**Question:** {run['question']}", "",
                  "**Retrieved chunks** (✓ = matches the expected source/text):", ""]
        marks = run["retrieval_score"]["relevant"]
        for i, chunk in enumerate(run["chunks"], 1):
            where = rag.location(chunk)
            score = chunk.get("reranker_score") if chunk.get("reranker_score") is not None else chunk.get("score")
            text = snippet(chunk.get("content"))
            where_text = f" ({where})" if where else ""
            score_text = f" - score {score:.4f}" if isinstance(score, (int, float)) else ""
            mark = (" ✓" if marks[i - 1] else " ✗") if marks else ""
            lines.append(f"{i}.{mark} {chunk.get('source_name')}{where_text}{score_text}  \n   > {text}")
        t = run["timings"]
        # Quote the answer so the model's own Markdown headings don't break the report's structure.
        answer = "\n".join(f"> {line}" if line.strip() else ">" for line in run["answer"].splitlines())
        lines += ["", f"_Retrieval {t['retrieval_s']:.2f} s, generation {t['generation_s']:.2f} s_", "",
                  "**Answer:**", "", answer, "", "**Answer quality:** _fill in_", ""]

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
