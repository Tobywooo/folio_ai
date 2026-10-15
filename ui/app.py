"""Folio AI - Gradio front end. Talks to the Azure Function App API (see backend/function_app.py)."""

from __future__ import annotations

import functools
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from api_client import Backend, BackendError

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Hugging Face ZeroGPU hardware won't start an app without at least one @spaces.GPU function. This app does no GPU
# work (the models run behind the Azure backend), so register a placeholder that is never called. The `spaces`
# package only exists on Spaces; locally the import fails and nothing happens.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder() -> None:
        """Never called."""
except ImportError:
    pass

backend = Backend(os.getenv("BACKEND_URL", "http://localhost:7071/api"), os.getenv("BACKEND_FUNCTION_KEY", ""))

RETRIEVAL_CHOICES = [
    ("Hybrid + semantic re-ranking (recommended)", "hybrid_semantic"),
    ("Hybrid (keyword + vector)", "hybrid"),
    ("Vector only", "vector"),
    ("Keyword only", "keyword"),
]
DEFAULT_MODE = os.getenv("RETRIEVAL_MODE", "hybrid_semantic")
if DEFAULT_MODE not in {value for _, value in RETRIEVAL_CHOICES}:
    DEFAULT_MODE = "hybrid_semantic"
SOURCE_HEADERS = ["Name", "Type", "Status", "Chunks", "Added", "Error"]
TYPE_LABELS = {"pdf": "PDF", "pptx": "PowerPoint", "txt": "Text", "url": "Web page"}


def ui_errors(fn):
    """Show backend failures as a Gradio error toast instead of a stack trace."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BackendError as exc:
            raise gr.Error(str(exc)) from exc

    return wrapper


def require_notebook(notebook_id: str | None) -> str:
    if not notebook_id:
        raise gr.Error("Create or select a notebook first.")
    return notebook_id


# ---------------------------------------------------------------- formatting


def format_answer(content: str, citations: list[dict[str, Any]]) -> str:
    if not citations:
        return content
    shown = [c for c in citations if c.get("cited")] or citations
    lines = []
    for c in shown:
        name = c.get("source_name") or "Untitled"
        if c.get("source_url"):
            name = f"[{name}]({c['source_url']})"
        where = f" - {c['location']}" if c.get("location") else ""
        snippet = " ".join((c.get("snippet") or "").split())[:160]
        lines.append(f"- **[{c.get('number')}]** {name}{where}" + (f"  \n  _\"{snippet}...\"_" if snippet else ""))
    return f"{content}\n\n---\n**Sources**\n" + "\n".join(lines)


def history_to_chat(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    chat = []
    for m in messages:
        content = m.get("content", "")
        if m.get("role") == "assistant":
            content = format_answer(content, m.get("citations") or [])
        chat.append({"role": m.get("role", "assistant"), "content": content})
    return chat


def sources_table(sources: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            s.get("name", ""),
            TYPE_LABELS.get(s.get("type", ""), s.get("type", "")),
            s.get("status", ""),
            s.get("chunk_count", ""),
            (s.get("created_at") or "")[:16].replace("T", " "),
            s.get("error", ""),
        ]
        for s in sources
    ]


def source_choices(sources: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(f"{s.get('name', '')} ({TYPE_LABELS.get(s.get('type', ''), '')})", s["id"]) for s in sources]


# ----------------------------------------------------------------- notebooks


@ui_errors
def refresh_notebooks(selected: str | None = None):
    notebooks = backend.list_notebooks()
    ids = [n["id"] for n in notebooks]
    value = selected if selected in ids else (ids[0] if ids else None)
    return gr.update(choices=[(n["name"], n["id"]) for n in notebooks], value=value)


@ui_errors
def create_notebook(name: str):
    notebook = backend.create_notebook((name or "").strip())
    gr.Info(f"Created notebook '{notebook['name']}'.")
    return refresh_notebooks(notebook["id"]), ""


@ui_errors
def rename_notebook(notebook_id: str | None, name: str):
    notebook = backend.rename_notebook(require_notebook(notebook_id), (name or "").strip())
    gr.Info(f"Renamed to '{notebook['name']}'.")
    return refresh_notebooks(notebook_id), ""


@ui_errors
def delete_notebook(notebook_id: str | None, confirmed: bool):
    require_notebook(notebook_id)
    if not confirmed:
        raise gr.Error("Tick 'Confirm delete' first. This removes the notebook's sources, chat, and artifacts.")
    backend.delete_notebook(notebook_id)
    gr.Info("Notebook deleted.")
    return refresh_notebooks(None), False


@ui_errors
def load_notebook(notebook_id: str | None):
    if not notebook_id:
        empty = gr.update(choices=[], value=None)
        return [], empty, [], empty, "", None
    sources = backend.list_sources(notebook_id)
    messages = backend.list_messages(notebook_id)
    artifacts = backend.list_artifacts(notebook_id)
    return (
        sources_table(sources),
        gr.update(choices=source_choices(sources), value=None),
        history_to_chat(messages),
        gr.update(choices=[(a["title"], a["id"]) for a in artifacts], value=None),
        "",
        None,
    )


# ------------------------------------------------------------------- sources


def _sources_outputs(notebook_id: str):
    sources = backend.list_sources(notebook_id)
    return sources_table(sources), gr.update(choices=source_choices(sources), value=None)


@ui_errors
def upload_files(notebook_id: str | None, paths: list[str] | None):
    require_notebook(notebook_id)
    if not paths:
        raise gr.Error("Choose one or more .pdf, .pptx, or .txt files first.")
    ok, failed = [], []
    for path in paths:
        name = Path(path).name
        try:
            backend.upload_file(notebook_id, path)
            ok.append(name)
        except BackendError as exc:
            failed.append(f"{name}: {exc}")
    if ok:
        gr.Info(f"Added {len(ok)} source(s): {', '.join(ok)}")
    for message in failed:
        gr.Warning(message)
    return (*_sources_outputs(notebook_id), None)


@ui_errors
def add_url(notebook_id: str | None, url: str):
    require_notebook(notebook_id)
    if not (url or "").strip():
        raise gr.Error("Enter a URL first.")
    source = backend.add_url(notebook_id, url.strip())
    gr.Info(f"Added '{source.get('name', url)}'.")
    return (*_sources_outputs(notebook_id), "")


@ui_errors
def delete_source(notebook_id: str | None, source_id: str | None):
    require_notebook(notebook_id)
    if not source_id:
        raise gr.Error("Pick a source to remove.")
    backend.delete_source(notebook_id, source_id)
    gr.Info("Source removed.")
    return _sources_outputs(notebook_id)


# ---------------------------------------------------------------------- chat


def add_user_message(notebook_id: str | None, question: str, chat: list[dict[str, str]]):
    require_notebook(notebook_id)
    if not (question or "").strip():
        raise gr.Error("Type a question first.")
    return chat + [{"role": "user", "content": question.strip()}], "", question.strip()


def answer_question(notebook_id: str, question: str, mode: str, chat: list[dict[str, str]]):
    try:
        result = backend.chat(notebook_id, question, mode)
    except BackendError as exc:
        # Nothing was saved: drop the optimistic user message and put the question back in the box.
        gr.Warning(str(exc))
        return chat[:-1], f"**Error:** {exc}", question
    timings = result.get("timings", {})
    reply = format_answer(result["answer"], result.get("citations", []))
    status = (
        f"Mode: `{result.get('mode')}` - retrieval {timings.get('retrieval_s', 0):.2f}s, "
        f"generation {timings.get('generation_s', 0):.2f}s"
    )
    return chat + [{"role": "assistant", "content": reply}], status, ""


# ----------------------------------------------------------------- artifacts


@ui_errors
def generate_artifact(notebook_id: str | None, kind: str, instructions: str, question_count: float):
    require_notebook(notebook_id)
    artifact = backend.create_artifact(notebook_id, kind, instructions or "", int(question_count))
    artifacts = backend.list_artifacts(notebook_id)
    gr.Info(f"Generated {artifact['title']}.")
    return gr.update(choices=[(a["title"], a["id"]) for a in artifacts], value=artifact["id"])


@ui_errors
def show_artifact(notebook_id: str | None, artifact_id: str | None):
    if not notebook_id or not artifact_id:
        return "", None
    artifact = backend.get_artifact(notebook_id, artifact_id)
    filename = re.sub(r"[^\w\- ]+", "_", artifact["title"]).strip().replace(" ", "_") + ".md"
    path = Path(tempfile.mkdtemp(prefix="folio-")) / filename
    path.write_text(artifact["content"], encoding="utf-8")
    return artifact["content"], str(path)


# ------------------------------------------------------------------------ UI


with gr.Blocks(title="Folio AI") as demo:
    gr.Markdown("# Folio AI\nCreate notebooks, add sources, ask questions with citations, and generate study materials.")

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            notebook_dd = gr.Dropdown(label="Notebook", choices=[], interactive=True)
            notebook_id_md = gr.Markdown()
            refresh_btn = gr.Button("Refresh", size="sm")
            name_box = gr.Textbox(label="Notebook name", placeholder="e.g. Biology 101")
            with gr.Row():
                create_btn = gr.Button("Create", variant="primary", size="sm")
                rename_btn = gr.Button("Rename", size="sm")
            with gr.Accordion("Delete notebook", open=False):
                confirm_delete = gr.Checkbox(label="Confirm delete")
                delete_btn = gr.Button("Delete notebook", variant="stop", size="sm")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Sources"):
                    with gr.Row():
                        with gr.Column():
                            file_input = gr.File(
                                label="Upload PDF, PPTX, or TXT",
                                file_count="multiple",
                                file_types=[".pdf", ".pptx", ".txt"],
                                type="filepath",
                            )
                            upload_btn = gr.Button("Add files", variant="primary")
                        with gr.Column():
                            url_box = gr.Textbox(label="Web URL", placeholder="https://...")
                            url_btn = gr.Button("Add URL", variant="primary")
                    sources_df = gr.Dataframe(headers=SOURCE_HEADERS, interactive=False, wrap=True)
                    with gr.Row():
                        source_dd = gr.Dropdown(label="Remove a source", choices=[], scale=3)
                        remove_source_btn = gr.Button("Remove", variant="stop", scale=1)

                with gr.Tab("Chat"):
                    chatbot = gr.Chatbot(type="messages", height=520, label="Notebook chat")
                    mode_radio = gr.Radio(RETRIEVAL_CHOICES, value=DEFAULT_MODE, label="Retrieval method")
                    with gr.Row():
                        question_box = gr.Textbox(
                            label="Question", placeholder="Ask about this notebook's sources...", scale=5, lines=1
                        )
                        ask_btn = gr.Button("Ask", variant="primary", scale=1)
                    chat_status = gr.Markdown()
                    pending_question = gr.State("")

                with gr.Tab("Artifacts"):
                    with gr.Row():
                        kind_radio = gr.Radio([("Report", "report"), ("Quiz", "quiz")], value="report", label="Type")
                        question_count = gr.Slider(3, 30, value=10, step=1, label="Quiz questions", visible=False)
                    instructions_box = gr.Textbox(
                        label="Extra instructions (optional)", placeholder="e.g. Focus on chapter 3", lines=2
                    )
                    generate_btn = gr.Button("Generate", variant="primary")
                    artifact_dd = gr.Dropdown(label="Saved artifacts", choices=[], interactive=True)
                    artifact_file = gr.File(label="Download (.md)", interactive=False)
                    artifact_view = gr.Markdown()

    notebook_outputs = [sources_df, source_dd, chatbot, artifact_dd, artifact_view, artifact_file]

    demo.load(refresh_notebooks, outputs=notebook_dd)
    refresh_btn.click(refresh_notebooks, inputs=notebook_dd, outputs=notebook_dd)
    notebook_dd.change(load_notebook, inputs=notebook_dd, outputs=notebook_outputs)
    notebook_dd.change(lambda nb: f"<small>ID: `{nb}`</small>" if nb else "", inputs=notebook_dd, outputs=notebook_id_md)
    create_btn.click(create_notebook, inputs=name_box, outputs=[notebook_dd, name_box])
    rename_btn.click(rename_notebook, inputs=[notebook_dd, name_box], outputs=[notebook_dd, name_box])
    delete_btn.click(delete_notebook, inputs=[notebook_dd, confirm_delete], outputs=[notebook_dd, confirm_delete])

    upload_btn.click(upload_files, inputs=[notebook_dd, file_input], outputs=[sources_df, source_dd, file_input])
    url_btn.click(add_url, inputs=[notebook_dd, url_box], outputs=[sources_df, source_dd, url_box])
    url_box.submit(add_url, inputs=[notebook_dd, url_box], outputs=[sources_df, source_dd, url_box])
    remove_source_btn.click(delete_source, inputs=[notebook_dd, source_dd], outputs=[sources_df, source_dd])

    for trigger in (ask_btn.click, question_box.submit):
        trigger(
            add_user_message,
            inputs=[notebook_dd, question_box, chatbot],
            outputs=[chatbot, question_box, pending_question],
        ).success(
            answer_question,
            inputs=[notebook_dd, pending_question, mode_radio, chatbot],
            outputs=[chatbot, chat_status, question_box],
        )

    kind_radio.change(lambda kind: gr.update(visible=kind == "quiz"), inputs=kind_radio, outputs=question_count)
    generate_btn.click(
        generate_artifact, inputs=[notebook_dd, kind_radio, instructions_box, question_count], outputs=artifact_dd
    )
    artifact_dd.change(show_artifact, inputs=[notebook_dd, artifact_dd], outputs=[artifact_view, artifact_file])


if __name__ == "__main__":
    username, password = os.getenv("UI_USERNAME", ""), os.getenv("UI_PASSWORD", "")
    launch_options: dict[str, Any] = {
        "auth": (username, password) if username and password else None,
        # Gradio's server-side rendering crashes on Spaces ("svelte-i18n ... initial locale"); plain rendering is fine.
        "ssr_mode": False,
    }
    if not os.getenv("SPACE_ID"):
        # Local / other hosts. On Hugging Face Spaces (which sets SPACE_ID) Gradio picks the host and port itself.
        launch_options["server_name"] = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
        launch_options["server_port"] = int(os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or 7860)
    demo.queue(default_concurrency_limit=4).launch(**launch_options)
