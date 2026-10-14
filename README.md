# folio_ai

A NotebookLM-style study app. Create notebooks, add PDF / PPTX / TXT files and web pages, ask questions answered with
Retrieval-Augmented Generation (RAG) and citations, and generate reports and quizzes as Markdown files.

## Architecture

```
Gradio UI (ui/)  ──HTTPS + function key──▶  Azure Function App (backend/, Python)
                                              │
                    ┌─────────────────────────┼──────────────────────────┐
                    ▼                         ▼                          ▼
          Azure Storage account      Azure AI Search             Llama API
          • blobs: raw + enriched    • index of chunks           • embeddings model
            sources, artifacts         (text + vectors,          • chat LLM
          • tables: notebooks,          notebook_id filter)
            sources, messages,
            artifacts
```

**Ingestion** (`backend/folio/ingest.py`): store the original file → extract text (pypdf, python-pptx, plain text,
BeautifulSoup for URLs) → split into overlapping chunks that keep page/slide numbers → embed with the Llama
embeddings API → upload chunks + vectors to Azure AI Search → save the extracted text and chunks as JSON in the
matching `-enriched` container.

**RAG chat** (`backend/folio/rag.py`): embed the question → retrieve the top-k chunks for that notebook only →
send them, numbered, to the Llama LLM with recent chat history → return the answer with `[n]` citations → save
both turns to the notebook's history.

**Artifacts** (`backend/folio/artifacts.py`): sample chunks from every source in the notebook (round-robin up to a
character budget) → ask the LLM for a report or a quiz with an answer key → save as `.md` in the `artifacts`
container.

**Retrieval methods** (selectable in the Chat tab, compared in `eval/`): `keyword` (BM25), `vector` (HNSW),
`hybrid` (keyword + vector with Reciprocal Rank Fusion), `hybrid_semantic` (hybrid + Azure semantic re-ranker).

## Project structure

```
backend/                 Azure Function App (deployed on its own)
  function_app.py        HTTP routes
  folio/
    config.py            environment variables
    storage.py           Blob + Table storage
    extract.py           PDF / PPTX / TXT / web-page text extraction
    chunking.py          recursive character chunking
    model_api.py         Llama chat + embeddings clients
    search_index.py      Azure AI Search schema and retrieval modes
    ingest.py            ingestion pipeline
    rag.py               question answering with citations
    artifacts.py         report / quiz generation
    notebooks.py         notebook create / rename / delete
ui/                      Gradio front end
  app.py, api_client.py
eval/                    retrieval comparison script
.env.example             configuration template
```

## How data is stored

Every notebook has a unique ID, and everything is keyed by it, so data survives restarts and redeploys.

| Data | Where |
|---|---|
| Notebooks | Table `{TABLE_PREFIX}notebooks` |
| Source records (name, type, status, chunk count, errors) | Table `{TABLE_PREFIX}sources`, partition = notebook ID |
| Chat history with citations | Table `{TABLE_PREFIX}messages`, partition = notebook ID |
| Artifact records | Table `{TABLE_PREFIX}artifacts`, partition = notebook ID |
| Original files | `pdf-documents`, `pptx-files`, `txt-files`, `web-scraped` → `<notebook_id>/<source_id>/<file>` |
| Extracted text + chunks | `*-enriched` containers → `<notebook_id>/<source_id>.json` |
| Generated reports / quizzes | `artifacts` → `<notebook_id>/<artifact_id>.md` |
| Chunk text + vectors | Azure AI Search index (`AZURE_SEARCH_INDEX_NAME`), filtered by `notebook_id` |

Deleting a notebook removes its blobs, index chunks, and table rows.

**Limitations:** scanned/image-only PDFs aren't OCR'd; web pages that need JavaScript to render may extract little
text; ingestion runs inside the upload request, so very large files can hit the ~230 s HTTP timeout; reports and
quizzes use at most `ARTIFACT_CONTEXT_CHARS` characters of source text.

## Configuration

Copy `.env.example` to `.env` and fill it in. All variables are documented in that file:

| Group | Variables |
|---|---|
| Storage | `AZURE_STORAGE_CONNECTION_STRING`, `CONTAINER_*`, `TABLE_PREFIX` |
| Search | `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY` (admin key), `AZURE_SEARCH_INDEX_NAME` |
| LLM | `LLM_API_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` |
| Embeddings | `EMBEDDING_API_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` |
| UI | `BACKEND_URL`, `BACKEND_FUNCTION_KEY`, optional `UI_USERNAME` / `UI_PASSWORD` |

The LLM and embeddings endpoints must accept the OpenAI-compatible formats (`/chat/completions`, `/embeddings`).
The containers, tables, and search index are created automatically on first use.

## Run locally

Requires Python 3.11+ and [Azure Functions Core Tools v4](https://learn.microsoft.com/azure/azure-functions/functions-run-local).

Backend (terminal 1):

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp local.settings.example.json local.settings.json
func start                      # serves http://localhost:7071/api
```

The backend reads `../.env` locally. Check it with `http://localhost:7071/api/health`.
In `local.settings.json` (gitignored), set `AzureWebJobsStorage` to your storage connection string; otherwise the
Functions host logs a harmless "Unable to create client for AzureWebJobsStorage" health warning.

UI (terminal 2):

```bash
cd ui
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python app.py                   # opens http://127.0.0.1:7860
```

## Using the app

1. **Create** a notebook in the left panel (or pick one from the dropdown). Rename or delete it from the same panel.
2. **Sources** tab: upload PDF/PPTX/TXT files or paste a URL. Each source shows its status and chunk count;
   failures show the reason.
3. **Chat** tab: ask a question. Answers cite sources as `[1]`, `[2]`, with the matching file, page/slide, and
   snippet listed underneath. Switch the retrieval method to compare results. History is saved per notebook.
4. **Artifacts** tab: generate a report or a quiz (with answer key), view it, and download the `.md` file.

## RAG evaluation

```bash
cp eval/questions.example.json eval/questions.json   # then write your own questions
python eval/run_eval.py --notebook-id <id> --modes vector hybrid_semantic
```

Writes `eval/results/eval-<timestamp>.md` (summary table, retrieved chunks with scores, answers, timings) and a
JSON copy. Fill in the answer-quality column and conclusions.

## Deployment

The Function App and the Gradio UI deploy separately; API keys are stored as Function App environment variables and
GitHub Secrets, never in the repository.

### Backend → Azure Functions (`.github/workflows/deploy-backend.yml`)

Runs on every push to `main` that touches `backend/` (or manually from the Actions tab):
**build** installs `backend/requirements.txt` into the package and checks that all functions register →
**deploy** logs in to Azure with OpenID Connect, copies the app settings from GitHub secrets/variables into the
Function App's environment variables, and publishes the package → **smoke test** calls `/api/health` with the
app's function key and fails the run if the app isn't healthy.

One-time setup:

| Where | What |
|---|---|
| Azure → Managed Identities | A user-assigned identity with a federated credential for this repo's `main` branch and the **Website Contributor** role on the Function App |
| GitHub → Settings → Secrets → Actions | Deploy identity: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`. App: `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_SEARCH_API_KEY`, `LLM_API_KEY`, `EMBEDDING_API_KEY` (and optionally the endpoints/models below) |
| GitHub → Settings → Variables → Actions | `AZURE_FUNCTIONAPP_NAME`, `AZURE_RESOURCE_GROUP`, plus `AZURE_SEARCH_ENDPOINT`, `LLM_API_BASE_URL`, `LLM_MODEL`, `EMBEDDING_API_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, and any optional tuning values from `.env.example` |

Each app setting is read from a secret first, then a variable with the same name. Settings not defined in GitHub
keep their current value in Azure (removing one from GitHub doesn't delete it from the Function App).

The workflow's `PYTHON_VERSION` must match the Function App's Python version.
