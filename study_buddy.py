import argparse
import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = "gemma3:1b"
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
).rstrip("/")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"


def build_prompt(question):
    return (
        "You are a friendly study assistant for beginners. "
        "Explain clearly using simple words and examples.\n\n"
        "Student question: " + question
    )


def ask_question(question, model=DEFAULT_MODEL, base_url=None):
    if base_url is None:
        base_url = OLLAMA_BASE_URL
    request_url = f"{base_url.rstrip('/')}/api/generate"

    data = {
        "model": model,
        "prompt": build_prompt(question),
        "stream": False,
    }

    request = urllib.request.Request(
        request_url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not connect to Ollama at {base_url}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Could not read the AI response: {error}") from error

    if not isinstance(result, dict) or "response" not in result:
        raise ValueError("Ollama returned an invalid response.")
    return result["response"]


def main():
    parser = argparse.ArgumentParser(
        description="Ask a local Ollama model a study question."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
        help="Ollama model name to use (default: %(default)s)",
    )
    parser.add_argument(
        "--base-url",
        default=OLLAMA_BASE_URL,
        help="Ollama server URL (default: %(default)s)",
    )
    args = parser.parse_args()

    global OLLAMA_URL
    OLLAMA_URL = f"{args.base_url.rstrip('/')}/api/generate"

    print("================================")
    print("      LOCAL AI STUDY BUDDY")
    print("================================")
    print("Ask a question, or type exit to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Goodbye! Keep learning!")
            break

        if not question:
            continue

        try:
            answer = ask_question(question, model=args.model, base_url=args.base_url)
            print("\nStudy Buddy:", answer)
        except (RuntimeError, ValueError) as error:
            print(f"\n{error}")
            print("Check that Ollama is running and the configured model is installed.")


if __name__ == "__main__":
    main()