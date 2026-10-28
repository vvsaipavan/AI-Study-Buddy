import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
import urllib.request
import urllib.error
import streamlit as st
from pypdf import PdfReader

MODEL = "gemma3:1b"
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
).rstrip("/")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
PROGRESS_DB_PATH = Path(__file__).resolve().with_name("study_progress.db")


def quiz_response_schema(question_count):
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 4,
                            "maxItems": 4
                        },
                        "correct_answer": {"type": "string"},
                        "explanation": {"type": "string"}
                    },
                    "required": [
                        "question",
                        "options",
                        "correct_answer",
                        "explanation"
                    ],
                    "additionalProperties": False
                },
                "minItems": question_count,
                "maxItems": question_count
            }
        },
        "required": ["questions"],
        "additionalProperties": False
    }


def open_progress_database():
    connection = sqlite3.connect(PROGRESS_DB_PATH)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY,
                score INTEGER NOT NULL,
                total INTEGER NOT NULL,
                difficulty TEXT NOT NULL,
                topic TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()
    except Exception:
        connection.close()
        raise
    return connection


def get_quiz_progress():
    with closing(open_progress_database()) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*), AVG(score * 100.0 / total), MAX(score * 100.0 / total)
            FROM quiz_attempts
            """
        ).fetchone()
    return {
        "attempts": row[0],
        "average_percentage": row[1] or 0,
        "best_percentage": row[2] or 0
    }


def save_quiz_result(score, total, difficulty, topic, source):
    with closing(open_progress_database()) as connection:
        connection.execute(
            """
            INSERT INTO quiz_attempts (score, total, difficulty, topic, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (score, total, difficulty, topic, source)
        )
        connection.commit()


FLASHCARD_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "front": {"type": "string"},
                    "back": {"type": "string"}
                },
                "required": ["front", "back"],
                "additionalProperties": False
            },
            "minItems": 8,
            "maxItems": 8
        }
    },
    "required": ["cards"],
    "additionalProperties": False
}


@st.cache_data(ttl=30)
def get_available_models():
    with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=5) as response:
        result = json.loads(response.read().decode("utf-8"))
    models = result.get("models", [])
    return [
        model["name"]
        for model in models
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    ]


def create_ollama_request(
    prompt, model, num_predict=100, response_format=None
):
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": num_predict}
    }
    if response_format is not None:
        data["format"] = response_format

    return urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )


def stream_ollama(request):
    completed = False
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if line.strip():
                chunk = json.loads(line.decode("utf-8"))
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                text = chunk.get("response", "")
                if text:
                    yield text
                if chunk.get("done"):
                    completed = True
    if not completed:
        raise RuntimeError("Ollama ended the response before it was complete.")


def reset_quiz():
    question_count = len(st.session_state.get("quiz", []))
    st.session_state.quiz = []
    st.session_state.quiz_checked = False
    st.session_state.quiz_settings = {}
    for i in range(question_count):
        st.session_state.pop(f"question_{i}", None)


st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("AI Study Buddy", icon=":material/school:")
    st.caption("Your personal space to learn and practice.")
    st.subheader("AI tutor", icon=":material/smart_toy:")
    try:
        available_models = get_available_models()
        if available_models:
            default_model_index = (
                available_models.index(MODEL) if MODEL in available_models else 0
            )
            selected_model = st.selectbox(
                "Choose an Ollama model",
                available_models,
                index=default_model_index,
                key="tutor_model"
            )
            st.caption(f"Offline-ready · Local model · {selected_model}")
        else:
            selected_model = None
            st.error("No Ollama models are installed.")
            st.caption(f"Install the default model with: `ollama pull {MODEL}`")
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        selected_model = None
        st.error("Could not connect to Ollama. Start Ollama to use the AI tutor.")
        st.caption(str(error))

    st.subheader("Study materials", icon=":material/menu_book:")
    uploaded_pdf = st.file_uploader(
        "Upload PDF notes",
        type=["pdf"],
        help="Use text-based PDF notes as context for chat and quizzes."
    )
    with st.expander("About this app"):
        st.markdown(
            "Chat and quizzes run locally with **Ollama** and "
            f"**{selected_model or MODEL}**. Keep Ollama running while you study."
        )

with st.container(border=True):
    st.caption("YOUR PERSONAL AI STUDY SPACE")
    st.title("Learn at your pace")
    st.write(
        "Ask questions, explore your notes, practice with quizzes, and "
        "turn important ideas into flashcards."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "quiz_checked" not in st.session_state:
    st.session_state.quiz_checked = False
if "quiz_settings" not in st.session_state:
    st.session_state.quiz_settings = {}
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
if "flashcard_source" not in st.session_state:
    st.session_state.flashcard_source = ""
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""
if "summary_source" not in st.session_state:
    st.session_state.summary_source = ""

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = 0

if uploaded_pdf is None:
    st.session_state.pdf_text = ""
    st.session_state.pdf_pages = 0
else:
    try:
        reader = PdfReader(uploaded_pdf)
        st.session_state.pdf_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
        st.session_state.pdf_pages = len(reader.pages)
        st.sidebar.success(
            f"{uploaded_pdf.name} · {st.session_state.pdf_pages} pages"
        )
        if not st.session_state.pdf_text.strip():
            st.sidebar.warning("No readable text was found in this PDF.")
    except Exception as error:
        st.session_state.pdf_text = ""
        st.session_state.pdf_pages = 0
        st.sidebar.error("Could not read this PDF.")
        st.sidebar.caption(str(error))

current_pdf_hash = (
    hashlib.sha256(st.session_state.pdf_text.encode("utf-8")).hexdigest()
    if st.session_state.pdf_text.strip()
    else ""
)
if st.session_state.flashcard_source != current_pdf_hash:
    st.session_state.flashcards = []
    st.session_state.flashcard_source = current_pdf_hash
    for i in range(8):
        st.session_state.pop(f"flashcard_reveal_{i}", None)
if st.session_state.summary_source != current_pdf_hash:
    st.session_state.summary_text = ""
    st.session_state.summary_source = current_pdf_hash

mode = st.segmented_control(
    "Choose a study mode",
    ["💬 Study Chat", "📖 Note Summary", "📝 Quiz Mode", "🗂️ Flashcards"],
    default="💬 Study Chat",
    key="study_mode",
    width="stretch"
)

if mode == "💬 Study Chat":
    chat_controls = st.container(border=True, key="chat_tools")
    with chat_controls:
        control_columns = st.columns(2)
        with control_columns[0]:
            st.download_button(
                "Export conversation",
                data="\n\n".join(
                    f"## {message['role'].title()}\n\n{message['content']}"
                    for message in st.session_state.messages
                ),
                file_name="ai-study-buddy-conversation.md",
                mime="text/markdown",
                icon=":material/download:",
                width="stretch",
                disabled=not st.session_state.messages,
                key="export_conversation"
            )
        with control_columns[1]:
            if st.button(
                "Clear conversation",
                icon=":material/delete:",
                width="stretch",
                disabled=not st.session_state.messages,
                key="clear_conversation"
            ):
                st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.messages:
        with st.container(border=True):
            st.subheader("What would you like to understand?")
            st.write(
                "Ask a question in the chat box below. You can ask for "
                "a simple explanation, an example, or help with your notes."
            )
            if st.session_state.pdf_text.strip():
                st.caption("Your uploaded PDF will be used as study context.")
            else:
                st.caption("Tip: add a PDF in the sidebar to study from your notes.")

    question = st.chat_input(
        "Ask your study question...",
        disabled=selected_model is None
    )

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Your Study Buddy is thinking..."):
                try:
                    if uploaded_pdf is not None and st.session_state.pdf_text.strip():
                        prompt = (
                            "You are a patient, friendly AI tutor. Teach clearly "
                            "with simple words and examples. Answer the student's "
                            "question using the PDF notes below. "
                            "If the notes do not contain the answer, say so.\n\n"
                            "PDF NOTES:\n"
                            + st.session_state.pdf_text[:12000]
                            + "\n\nSTUDENT QUESTION:\n"
                            + question
                        )
                    else:
                        prompt = (
                            "You are a patient, friendly AI tutor for beginners. "
                            "Explain clearly with simple words and examples, "
                            "and guide the student toward understanding.\n\n"
                            + question
                        )

                    answer = st.write_stream(
                        stream_ollama(
                            create_ollama_request(prompt, selected_model)
                        )
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as error:
                    st.error(
                        "Could not get an answer from Ollama. "
                        "Check that Ollama is running and the selected "
                        "model is available."
                    )
                    st.caption(str(error))
elif mode == "📖 Note Summary":
    st.subheader("Summarize your notes", icon=":material/summarize:")
    if not st.session_state.pdf_text.strip():
        st.info("Upload a readable PDF in the sidebar to create a summary.")
    else:
        st.caption(
            f"Create a clear summary from {uploaded_pdf.name}, "
            f"containing {st.session_state.pdf_pages} pages."
        )
        if st.button(
            "Generate summary",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
            disabled=selected_model is None,
            key="generate_summary"
        ):
            summary_prompt = (
                "Summarize the following study notes in clear, student-friendly "
                "language. Organize the response with a concise overview, key "
                "concepts, important details, and a short list of takeaways. "
                "Stay faithful to the notes and do not invent facts.\n\n"
                + st.session_state.pdf_text[:12000]
            )
            summary_status = None
            try:
                with st.status(
                    "Summarizing your notes with Ollama...",
                    expanded=False
                ) as summary_status:
                    summary_text = st.write_stream(
                        stream_ollama(
                            create_ollama_request(
                                summary_prompt,
                                selected_model,
                                num_predict=1200
                            )
                        )
                    )
                if not summary_text.strip():
                    raise ValueError("The generated summary was empty.")
                st.session_state.summary_text = summary_text
                st.session_state.summary_source = current_pdf_hash
                summary_status.update(
                    label="Summary ready.",
                    state="complete"
                )
            except Exception as error:
                if summary_status is not None:
                    summary_status.update(
                        label="Summary generation failed.",
                        state="error"
                    )
                st.error(f"Could not summarize your notes: {error}")

        if st.session_state.summary_text:
            st.markdown(st.session_state.summary_text)
elif mode == "📝 Quiz Mode":
    try:
        progress = get_quiz_progress()
        progress_columns = st.columns(3)
        progress_columns[0].metric("Quizzes completed", progress["attempts"])
        progress_columns[1].metric(
            "Average score", f"{progress['average_percentage']:.0f}%"
        )
        progress_columns[2].metric(
            "Best score", f"{progress['best_percentage']:.0f}%"
        )
        st.caption("Your quiz history is saved locally on this computer.")
    except sqlite3.Error as error:
        st.error(f"Could not load your saved quiz progress: {error}")

    difficulty = st.selectbox(
        "Question difficulty",
        ["Beginner", "Intermediate", "Advanced"],
        key="quiz_difficulty",
        disabled=bool(st.session_state.quiz)
    )
    question_count = st.selectbox(
        "Number of questions",
        [5, 10],
        key="quiz_question_count",
        disabled=bool(st.session_state.quiz)
    )
    topic = st.text_input(
        "Focus topic (optional)",
        placeholder="e.g. functions, photosynthesis, or chapter 3",
        disabled=bool(st.session_state.quiz)
    )
    if not st.session_state.quiz:
        with st.container(border=True):
            st.subheader("Ready to test your knowledge?")
            if st.session_state.pdf_text.strip():
                st.write(
                    f"Create {question_count} {difficulty.lower()} "
                    "multiple-choice questions "
                    "from your notes."
                )
            else:
                st.write(
                    f"Create {question_count} {difficulty.lower()} questions "
                    f"about {topic.strip() or 'Python'}."
                )

    if st.button(
        f"Generate {question_count} quiz questions",
        type="primary",
        icon=":material/auto_awesome:",
        width="stretch",
        disabled=selected_model is None
    ):
        reset_quiz()
        focus_instruction = (
            f"Focus especially on {topic.strip()}. "
            if topic.strip()
            else ""
        )
        if uploaded_pdf is not None and st.session_state.pdf_text.strip():
            quiz_prompt = (
                f"Create exactly {question_count} {difficulty.lower()} "
                "multiple-choice questions based only on these notes. "
                + focus_instruction
                + "Each question must have four options, one correct answer, "
                "and a short explanation of why the answer is correct.\n\n"
                + st.session_state.pdf_text[:12000]
            )
        else:
            quiz_prompt = (
                f"Create exactly {question_count} {difficulty.lower()} "
                "multiple-choice questions about "
                f"{topic.strip() or 'Python'}. Each question must have four "
                "options, one correct answer, and a short explanation of why "
                "the answer is correct."
            )

        try:
            with st.status(
                "Generating your quiz with Ollama. This may take a minute...",
                expanded=False
            ) as quiz_status:
                quiz_json = "".join(
                    stream_ollama(
                        create_ollama_request(
                            quiz_prompt,
                            selected_model,
                            num_predict=320 * question_count,
                            response_format=quiz_response_schema(question_count)
                        )
                    )
                )
            quiz_status.update(label="Checking generated questions...")
            quiz_data = json.loads(quiz_json)
            questions = quiz_data.get("questions")
            if (
                not isinstance(questions, list)
                or len(questions) != question_count
            ):
                raise ValueError(
                    f"The generated quiz did not contain {question_count} "
                    "questions."
                )
            for item in questions:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("question"), str)
                    or not isinstance(item.get("options"), list)
                    or len(item["options"]) != 4
                    or not all(isinstance(option, str) for option in item["options"])
                    or not isinstance(item.get("correct_answer"), str)
                    or item["correct_answer"] not in item["options"]
                    or not isinstance(item.get("explanation"), str)
                ):
                    raise ValueError(
                        "The generated quiz has an invalid question or answer."
                    )
            st.session_state.quiz = questions
            st.session_state.quiz_settings = {
                "difficulty": difficulty,
                "topic": topic.strip() or "All topics",
                "source": "Uploaded PDF" if st.session_state.pdf_text.strip()
                else "General study"
            }
            quiz_status.update(
                label="Quiz generated successfully.",
                state="complete"
            )
        except (json.JSONDecodeError, ValueError) as error:
            quiz_status.update(label="Quiz generation failed.", state="error")
            st.error(f"Could not read the generated quiz: {error}")
        except Exception as error:
            quiz_status.update(label="Quiz generation failed.", state="error")
            st.error(f"Could not generate quiz: {error}")

    if st.session_state.quiz:
        settings = st.session_state.quiz_settings
        st.caption(
            f"{settings.get('difficulty', difficulty)} · "
            f"{len(st.session_state.quiz)} questions · "
            f"{settings.get('topic', 'All topics')}"
        )
        st.subheader("Your quiz", icon=":material/quiz:")
        for i, item in enumerate(st.session_state.quiz):
            with st.container(border=True):
                st.radio(
                    f"{i + 1}. {item['question']}",
                    item["options"],
                    key=f"question_{i}",
                    index=None
                )

        if st.button(
            "Submit quiz",
            type="primary",
            icon=":material/check_circle:",
            width="stretch",
            disabled=st.session_state.quiz_checked
        ):
            if any(
                st.session_state.get(f"question_{i}") is None
                for i in range(len(st.session_state.quiz))
            ):
                st.warning("Choose an answer for every question before submitting.")
            else:
                st.session_state.quiz_checked = True
                score = sum(
                    st.session_state.get(f"question_{i}")
                    == item["correct_answer"]
                    for i, item in enumerate(st.session_state.quiz)
                )
                settings = st.session_state.quiz_settings
                try:
                    save_quiz_result(
                        score,
                        len(st.session_state.quiz),
                        settings.get("difficulty", difficulty),
                        settings.get("topic", "All topics"),
                        settings.get("source", "General study")
                    )
                except sqlite3.Error as error:
                    st.error(f"Could not save your quiz result: {error}")
                else:
                    st.success("Quiz result saved to your local study progress.")

        if st.session_state.quiz_checked:
            score = 0
            for i, item in enumerate(st.session_state.quiz):
                selected_answer = st.session_state.get(f"question_{i}")
                if selected_answer == item["correct_answer"]:
                    score += 1
                    st.success(f"Question {i + 1}: Correct!")
                else:
                    st.error(
                        f"Question {i + 1}: The correct answer is "
                        f"{item['correct_answer']}."
                    )
                if item.get("explanation"):
                    st.caption(item["explanation"])
            total_questions = len(st.session_state.quiz)
            st.success(f"Your score: {score}/{total_questions}")
            st.write(f"Percentage: {score / total_questions * 100:.0f}%")
            st.button(
                "Create another quiz",
                icon=":material/refresh:",
                width="stretch",
                key="new_quiz",
                on_click=reset_quiz
            )
else:
    st.subheader("Flashcards from your notes", icon=":material/style:")
    if not st.session_state.pdf_text.strip():
        st.info("Upload a readable PDF in the sidebar to create flashcards.")
    else:
        st.caption(
            "Generate eight quick-review cards from your uploaded PDF. "
            "Your deck refreshes when you change the notes."
        )
        if st.button(
            "Generate flashcards",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
            disabled=selected_model is None,
            key="generate_flashcards"
        ):
            st.session_state.flashcards = []
            for i in range(8):
                st.session_state.pop(f"flashcard_reveal_{i}", None)
            flashcard_prompt = (
                "Create exactly eight concise study flashcards based only on "
                "the notes below. Each front should ask about one important "
                "term, fact, or concept. Each back should give a clear, "
                "accurate answer in one or two sentences.\n\n"
                + st.session_state.pdf_text[:12000]
            )
            try:
                with st.status(
                    "Creating flashcards from your notes...",
                    expanded=False
                ) as flashcard_status:
                    flashcard_json = "".join(
                        stream_ollama(
                            create_ollama_request(
                                flashcard_prompt,
                                selected_model,
                                num_predict=900,
                                response_format=FLASHCARD_RESPONSE_SCHEMA
                            )
                        )
                    )
                    flashcard_status.update(label="Checking your flashcards...")
                flashcard_data = json.loads(flashcard_json)
                cards = flashcard_data.get("cards")
                if not isinstance(cards, list) or len(cards) != 8:
                    raise ValueError(
                        "The generated deck did not contain 8 flashcards."
                    )
                if any(
                    not isinstance(card, dict)
                    or not isinstance(card.get("front"), str)
                    or not card["front"].strip()
                    or not isinstance(card.get("back"), str)
                    or not card["back"].strip()
                    for card in cards
                ):
                    raise ValueError(
                        "The generated deck contains an invalid flashcard."
                    )
                st.session_state.flashcards = cards
                st.session_state.flashcard_source = current_pdf_hash
                flashcard_status.update(
                    label="Flashcards are ready.",
                    state="complete"
                )
            except (json.JSONDecodeError, ValueError) as error:
                flashcard_status.update(
                    label="Flashcard generation failed.",
                    state="error"
                )
                st.error(f"Could not read the generated flashcards: {error}")
            except Exception as error:
                flashcard_status.update(
                    label="Flashcard generation failed.",
                    state="error"
                )
                st.error(f"Could not generate flashcards: {error}")

        if st.session_state.flashcards:
            st.caption(f"{len(st.session_state.flashcards)} cards in this deck")
            for i, card in enumerate(st.session_state.flashcards):
                with st.container(border=True):
                    st.markdown(f"**Card {i + 1}**")
                    st.write(card["front"])
                    revealed = st.toggle(
                        "Reveal answer",
                        key=f"flashcard_reveal_{i}"
                    )
                    if revealed:
                        st.info(card["back"], icon=":material/lightbulb:")