"""Publish ui/ to a Hugging Face Space and sync its secrets. Run by .github/workflows/deploy-ui.yml.

Environment:
  HF_TOKEN, HF_SPACE_ID                 required (token needs write access to the Space)
  BACKEND_URL (or AZURE_FUNCTIONAPP_HOST), BACKEND_FUNCTION_KEY   required, become Space secrets
  UI_USERNAME, UI_PASSWORD, RETRIEVAL_MODE   optional, become Space secrets when set
  GRADIO_VERSION                        Gradio version the Space runs (sdk_version)
  GIT_SHA                               used in the Space commit message
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[2]
UI = REPO / "ui"
APP_FILES = ["app.py", "api_client.py"]
REQUIRED_SECRETS = ["BACKEND_URL", "BACKEND_FUNCTION_KEY"]
OPTIONAL_SECRETS = ["UI_USERNAME", "UI_PASSWORD", "RETRIEVAL_MODE"]
FAILED_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}


def env(name: str) -> str:
    return os.environ.get(name, "").strip()


def fail(message: str) -> None:
    print(f"::error::{message}")
    sys.exit(1)


def build_package(target: Path, gradio_version: str) -> None:
    for name in APP_FILES:
        shutil.copy2(UI / name, target / name)
    # The Space installs Gradio itself from sdk_version, so leave it out of requirements.txt.
    requirements = [
        line for line in (UI / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().lower().startswith("gradio")
    ]
    (target / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    readme = (UI / "SPACE_README.md").read_text(encoding="utf-8").replace("{{GRADIO_VERSION}}", gradio_version)
    (target / "README.md").write_text(readme, encoding="utf-8")


def wait_until_running(api: HfApi, space: str, timeout_s: int = 900) -> None:
    """After an upload the Space rebuilds. Wait for it to leave the old RUNNING state, then to run again."""
    started = time.time()
    left_running = False
    last = ""
    while time.time() - started < timeout_s:
        runtime = api.get_space_runtime(space)
        stage = str(getattr(runtime.stage, "value", runtime.stage))
        if stage != last:
            print(f"Space stage: {stage}")
            last = stage
        if stage in FAILED_STAGES:
            reason = (runtime.raw or {}).get("errorMessage") or "see the Space's Logs tab"
            fail(f"Space failed with {stage}: {reason}")
        if stage != "RUNNING":
            left_running = True
        elif left_running or time.time() - started > 120:
            # Back to RUNNING after a rebuild, or still RUNNING after 2 minutes because nothing changed.
            return
        time.sleep(10)
    fail(f"Space did not reach RUNNING within {timeout_s // 60} minutes (last stage: {last}).")


def check_page(api: HfApi, space: str, token: str) -> None:
    info = api.space_info(space)
    host = getattr(info, "host", None) or f"https://{space.replace('/', '-').replace('_', '-').lower()}.hf.space"
    for attempt in range(1, 13):
        try:
            resp = requests.get(host, headers={"Authorization": f"Bearer {token}"}, timeout=30)
            if resp.status_code == 200:
                print(f"Space is serving: {host}")
                return
            print(f"Attempt {attempt}: HTTP {resp.status_code}")
        except requests.RequestException as exc:
            print(f"Attempt {attempt}: {exc}")
        time.sleep(10)
    fail(f"{host} did not return 200.")


def backend_url() -> str:
    """BACKEND_URL if given, else https://<AZURE_FUNCTIONAPP_HOST>/api."""
    url = env("BACKEND_URL")
    if not url and env("AZURE_FUNCTIONAPP_HOST"):
        host = env("AZURE_FUNCTIONAPP_HOST").removeprefix("https://").removeprefix("http://").split("/")[0]
        url = f"https://{host}/api"
    return url.rstrip("/")


def main() -> None:
    os.environ["BACKEND_URL"] = backend_url()
    token, space = env("HF_TOKEN"), env("HF_SPACE_ID")
    missing = [n for n in ["HF_TOKEN", "HF_SPACE_ID", *REQUIRED_SECRETS] if not env(n)]
    if missing:
        fail(f"Missing GitHub secrets/variables: {', '.join(missing)}")

    api = HfApi(token=token)
    try:
        api.space_info(space)
    except Exception as exc:
        fail(f"Can't access Space '{space}'. Create it on huggingface.co first and check HF_TOKEN has write access. ({exc})")

    for name in REQUIRED_SECRETS + OPTIONAL_SECRETS:
        if env(name):
            api.add_space_secret(space, name, env(name))
    print(f"Synced Space secrets: {', '.join(n for n in REQUIRED_SECRETS + OPTIONAL_SECRETS if env(n))}")

    with tempfile.TemporaryDirectory() as tmp:
        build_package(Path(tmp), env("GRADIO_VERSION") or "5.50.0")
        commit = api.upload_folder(
            repo_id=space,
            repo_type="space",
            folder_path=tmp,
            commit_message=f"Deploy from GitHub {env('GIT_SHA')[:7]}".strip(),
            delete_patterns=["*.py", "requirements.txt"],  # drop files that no longer exist in ui/
        )
    print(f"Uploaded: {getattr(commit, 'commit_url', commit)}")

    wait_until_running(api, space)
    check_page(api, space, token)


if __name__ == "__main__":
    main()
