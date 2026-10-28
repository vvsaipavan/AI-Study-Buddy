import argparse
import importlib.util
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
).rstrip("/")
DEFAULT_MODEL = "gemma3:1b"


def get_installed_models():
    request = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
    with urllib.request.urlopen(request, timeout=4) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Ollama returned an invalid model list.")
    models = result.get("models", [])
    if not isinstance(models, list):
        raise ValueError("Ollama returned an invalid model list.")
    return [
        model["name"]
        for model in models
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    ]


def check_setup():
    missing_packages = [
        package
        for package in ("streamlit", "pypdf")
        if importlib.util.find_spec(package) is None
    ]
    if missing_packages:
        print(
            "Missing Python dependencies: "
            + ", ".join(missing_packages)
            + "\nInstall them with: python -m pip install -r requirements.txt"
        )
        return False

    print("Python dependencies: ready")
    try:
        models = get_installed_models()
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as error:
        print(f"Ollama: unavailable at {OLLAMA_BASE_URL} ({error})")
        print("Start Ollama and run: ollama pull gemma3:1b")
        return False

    if not models:
        print("Ollama is running, but no local models are installed.")
        print("Download a model before going offline: ollama pull gemma3:1b")
        return False
    print("Ollama: ready")
    print("Installed models: " + ", ".join(models))
    if DEFAULT_MODEL not in models:
        print(
            f"Note: {DEFAULT_MODEL} is not installed; the app will use "
            "another installed model."
        )
    return True


def parse_port(value):
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Port must be a number.") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535.")
    return port


def main():
    parser = argparse.ArgumentParser(
        description="Start the offline AI Study Buddy on this computer."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Network interface to bind to; defaults to this computer only.",
    )
    parser.add_argument("--port", type=parse_port, default=8501)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Python dependencies, Ollama, and local models, then exit.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the app in a browser automatically.",
    )
    args = parser.parse_args()

    if args.check:
        return 0 if check_setup() else 1

    missing_packages = [
        package
        for package in ("streamlit", "pypdf")
        if importlib.util.find_spec(package) is None
    ]
    if missing_packages:
        check_setup()
        return 1

    display_host = (
        "localhost" if args.host in ("0.0.0.0", "::") else args.host
    )
    app_url = f"http://{display_host}:{args.port}"
    try:
        models = get_installed_models()
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as error:
        models = []
        print(f"Warning: Ollama is not reachable at {OLLAMA_BASE_URL}: {error}")
        print("Chat, quiz generation, and flashcard generation need local Ollama.")
    if models:
        print("Using local Ollama models: " + ", ".join(models))
    else:
        print("No Ollama models are available. Install one before using AI features.")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "Warning: this server is reachable from other devices on the "
            "network. Only use this on a trusted private network."
        )
    print(f"Starting AI Study Buddy at {app_url}")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_DIR / "app.py"),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true" if args.no_browser else "false",
    ]
    try:
        return subprocess.call(command, cwd=PROJECT_DIR)
    except KeyboardInterrupt:
        print("\nAI Study Buddy stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
