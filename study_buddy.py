import json
import urllib.request
import urllib.error

MODEL = "gemma3:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"

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

    data = {
        "model": MODEL,
        "prompt": (
            "You are a friendly study assistant for beginners. "
            "Explain clearly using simple words and examples.\n\n"
            "Student question: " + question
        ),
        "stream": False
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

        print("\nStudy Buddy:", result["response"])

    except urllib.error.URLError as error:
        print("\nCould not connect to Ollama.")
        print("Check that Ollama is running.")
        print("Details:", error)

    except (KeyError, json.JSONDecodeError) as error:
        print("\nCould not read the AI response:", error)