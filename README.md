# AI Study Buddy

A local study assistant with PDF-aware chat and interactive quizzes, powered by Streamlit and Ollama.

## Setup for Windows, macOS, and Linux

Install Python 3.10 or newer and [Ollama](https://ollama.com/download) on each computer that will run the app. In the project folder, create a virtual environment and install the dependencies:

**Windows PowerShell**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull gemma3:1b
.\.venv\Scripts\python.exe run_local.py
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull gemma3:1b
python run_local.py
```

`run_local.py` starts the app on this computer at <http://localhost:8501>, checks for local Ollama models, and displays setup guidance if the model service is unavailable. Run `python run_local.py --check` to check setup without starting the web app. In VS Code, select the project's `.venv` Python interpreter and run **Run AI Study Buddy**.

### Offline use

The app sends AI requests only to the configured Ollama server; it does not use a cloud AI service. First install the app's Python packages and Ollama, and download a model while you have internet access. After those one-time downloads, chat, summaries, quizzes, flashcards, and the app can run without internet, provided Ollama is running and the computer has enough memory and storage for the selected model. Quiz scores are saved in a local SQLite database in the project folder. Each computer needs its own Python environment, Ollama installation, and local model. "Any device" is limited by the operating system, Python/Ollama support, and the model's hardware requirements; this is not a browser-only mobile app.

To use a different installed model, select it in the app's sidebar. To connect the app to an Ollama service at a custom address, set `OLLAMA_BASE_URL` before starting it (for example, `http://127.0.0.1:11434`).

The server binds to `127.0.0.1` by default and is private to that computer. To make the web UI available to other devices on a trusted local network, run `python run_local.py --host 0.0.0.0`; the host computer must remain on, and other clients use the host's local IP address with port `8501`. This shares the host computer's Ollama model and app session; it does not run a separate model on each client. Do not expose the unauthenticated development server to the public internet.

To run the app with local AI error monitoring, start Ollama and run **Run AI Study Buddy with AI Repairs** from the VS Code task menu, or use:

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe dev_runner.py
```

**macOS or Linux** (with the virtual environment activated)

```bash
python dev_runner.py
```

The runner monitors Streamlit's logs and asks an installed local Ollama model to propose a small repair when it detects a Python traceback. It only edits `app.py`, checks that the edited file compiles, and verifies the app starts before keeping the change. If startup validation fails, it restores the previous version. Automatic repairs are limited to three per run. This is a local development helper, not a replacement for reviewing changes or testing the app.

## Features

- Choose an installed local Ollama model for the AI tutor, then ask questions in Study Chat.
- Upload text-based PDFs so chat, summaries, quizzes, and flashcards can use the notes.
- Generate five or ten multiple-choice questions at beginner, intermediate, or advanced difficulty, with an optional topic focus, answer explanations, and scoring.
- Summarize uploaded notes into a concise overview, key concepts, and takeaways.
- Track quiz attempts, average scores, and personal best across app restarts.
- Switch between light and dark themes, or use the system appearance, from **Settings → Theme**.
- Turn PDF notes into an eight-card interactive flashcard deck.
- Export conversations as Markdown or clear the current chat.
- Use the redesigned responsive hero, study-themed backgrounds, and styled study-mode controls.
- Enjoy subtle, locally embedded study-themed background illustrations that work offline.
