# AI Study Buddy

A local study assistant with PDF-aware chat and interactive quizzes, powered by Streamlit and Ollama.

## Setup

1. Install Python 3.10 or newer and [Ollama](https://ollama.com/download).
2. Open a terminal in this project folder and install the Python dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Download the model:

   ```powershell
   ollama pull gemma3:1b
   ```

4. Start the app:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

   Or run **Run AI Study Buddy** from the VS Code task menu.

The app opens at <http://localhost:8501>. Keep Ollama running while using chat or quiz generation.

## Features

- Choose an installed local Ollama model for the AI tutor, then ask questions in Study Chat.
- Upload text-based PDFs so chat and quizzes can use the notes.
- Generate five multiple-choice questions and submit answers for a score.
