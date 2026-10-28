import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path


APP_PATH = Path(__file__).with_name("app.py")
APP_URL = "http://localhost:8501"
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
).rstrip("/")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
DEFAULT_MODEL = "gemma3:1b"
MAX_REPAIRS = 3
EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["old", "new"],
                "additionalProperties": False,
            },
            "maxItems": 3,
        }
    },
    "required": ["edits"],
    "additionalProperties": False,
}


def read_model():
    with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=5) as response:
        result = json.loads(response.read().decode("utf-8"))
    models = result.get("models", [])
    names = [
        model["name"]
        for model in models
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    ]
    if not names:
        raise RuntimeError("No Ollama models are installed.")
    if DEFAULT_MODEL in names:
        return DEFAULT_MODEL
    print(f"Default model {DEFAULT_MODEL} is unavailable; using {names[0]}.")
    return names[0]


def traceback_context(error_text, source):
    matches = list(re.finditer(r'File "([^"]*app\.py)", line (\d+)', error_text))

    lines = source.splitlines()
    if not matches:
        return "\n".join(lines[:80])

    match = matches[-1]
    line_number = int(match.group(2))
    start = max(0, line_number - 31)
    end = min(len(lines), line_number + 30)
    return "\n".join(
        f"{index + 1}: {lines[index]}" for index in range(start, end)
    )


def request_edits(error_text, source):
    model = read_model()
    context = traceback_context(error_text, source)
    prompt = (
        "You are repairing a local Streamlit study app after a Python runtime "
        "error. Return only a JSON object matching the schema. Make the "
        "smallest safe edits to fix the root cause. Each old value must be an "
        "exact, unique snippet copied from the source; each new value replaces "
        "it. Do not add dependencies, run commands, edit files, or rewrite "
        "unrelated behavior. If the error cannot be safely fixed from this "
        "context, return an empty edits array.\n\n"
        f"TRACEBACK:\n{error_text[-4000:]}\n\n"
        f"APP SOURCE AROUND THE ERROR:\n{context}"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": EDIT_SCHEMA,
        "options": {"num_ctx": 8192, "num_predict": 1800},
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("error"):
        raise RuntimeError(f"Ollama could not generate a repair: {result['error']}")
    response_text = result.get("response")
    if not isinstance(response_text, str):
        raise ValueError("Ollama returned no repair response.")
    generated = json.loads(response_text)
    edits = generated.get("edits")
    if not isinstance(edits, list) or len(edits) > 3:
        raise ValueError("Ollama returned an invalid repair.")
    for edit in edits:
        if (
            not isinstance(edit, dict)
            or not isinstance(edit.get("old"), str)
            or not isinstance(edit.get("new"), str)
            or not edit["old"]
            or len(edit["old"]) > 3000
            or len(edit["new"]) > 5000
        ):
            raise ValueError("Ollama returned an invalid code edit.")
    return edits


def apply_edits(source, edits):
    updated = source
    for edit in edits:
        old = edit["old"].replace("\r\n", "\n")
        new = edit["new"].replace("\r\n", "\n")
        if updated.count(old) != 1:
            raise ValueError(
                "A suggested edit did not match exactly one place in app.py."
            )
        updated = updated.replace(old, new, 1)
    if updated == source:
        raise ValueError("The suggested repair did not change app.py.")
    compile(updated, str(APP_PATH), "exec")
    return updated


def start_app():
    output = queue.Queue()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_PATH),
            "--server.headless",
            "true",
            "--server.port",
            "8501",
        ],
        cwd=APP_PATH.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def collect_output():
        if process.stdout is not None:
            for line in process.stdout:
                output.put(line.rstrip())

    threading.Thread(target=collect_output, daemon=True).start()
    return process, output


def drain_output(output, history):
    received = False
    while True:
        try:
            line = output.get_nowait()
        except queue.Empty:
            break
        print(line, flush=True)
        history.append(line)
        received = True
    return received


def latest_traceback(history):
    lines = list(history)
    start = None
    for index, line in enumerate(lines):
        if "Traceback (most recent call last):" in line:
            start = index
    if start is None:
        return None
    return "\n".join(lines[start:])[-6000:]


def app_is_healthy():
    try:
        with urllib.request.urlopen(APP_URL, timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def wait_until_ready(process, output, history, timeout=45):
    started = time.monotonic()
    last_output = started
    ready_at = None
    while time.monotonic() - started < timeout:
        if drain_output(output, history):
            last_output = time.monotonic()
        if process.poll() is not None:
            return False, latest_traceback(history)
        now = time.monotonic()
        if latest_traceback(history) and now - last_output >= 1.2:
            return False, latest_traceback(history)
        if app_is_healthy():
            if ready_at is None:
                ready_at = now
            elif now - ready_at >= 2:
                return True, None
        time.sleep(0.25)
    return False, latest_traceback(history)


def stop_app(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def propose_repair(error_text):
    original_bytes = APP_PATH.read_bytes()
    raw_source = original_bytes.decode("utf-8")
    newline = "\r\n" if "\r\n" in raw_source else "\n"
    source = raw_source.replace("\r\n", "\n")
    edits = request_edits(error_text, source)
    if not edits:
        raise RuntimeError("The local AI did not find a safe automatic repair.")
    updated = apply_edits(source, edits)
    APP_PATH.write_bytes(updated.replace("\n", newline).encode("utf-8"))
    print(f"Applied {len(edits)} proposed edit(s) to app.py.")
    return original_bytes


def run():
    process, output = start_app()
    history = deque(maxlen=500)
    repairs = 0
    ready, error_text = wait_until_ready(process, output, history)
    if ready:
        print(f"\nAI Study Buddy is running at {APP_URL}")
        print("Press Ctrl+C to stop the app and its AI error monitor.\n")
    elif error_text is None:
        stop_app(process)
        raise RuntimeError("Streamlit did not become ready within 45 seconds.")

    try:
        idle_since = time.monotonic()
        while True:
            received = drain_output(output, history)
            if received:
                idle_since = time.monotonic()

            error_text = latest_traceback(history)
            if process.poll() is not None and error_text is None:
                raise RuntimeError("Streamlit exited unexpectedly.")
            if error_text and time.monotonic() - idle_since >= 1.2:
                if repairs >= MAX_REPAIRS:
                    raise RuntimeError(
                        f"Reached the limit of {MAX_REPAIRS} automatic repairs."
                    )
                print("\nAn app error was detected. Asking the local AI to repair it...")
                original_bytes = propose_repair(error_text)
                repairs += 1
                stop_app(process)
                process, output = start_app()
                history = deque(maxlen=500)
                ready, startup_error = wait_until_ready(process, output, history)
                if not ready:
                    stop_app(process)
                    APP_PATH.write_bytes(original_bytes)
                    print("The repair failed validation; restored the previous app.py.")
                    process, output = start_app()
                    history = deque(maxlen=500)
                    restored, _ = wait_until_ready(process, output, history)
                    if restored:
                        print(f"Previous version restored at {APP_URL}.")
                    raise RuntimeError(
                        "Automatic repair did not produce a healthy app."
                    )
                print(f"Repair validated. App is running at {APP_URL}.")
                idle_since = time.monotonic()
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nStopping AI Study Buddy...")
    finally:
        stop_app(process)


if __name__ == "__main__":
    try:
        run()
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError,
            urllib.error.URLError, subprocess.SubprocessError) as error:
        print(f"\nAI web runner stopped: {error}", file=sys.stderr)
        sys.exit(1)
