import json
import urllib.request
import urllib.error
import streamlit as st
from pypdf import PdfReader

MODEL = "gemma3:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"
QUIZ_RESPONSE_SCHEMA = {
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
                    "correct_answer": {"type": "string"}
                },
                "required": ["question", "options", "correct_answer"],
                "additionalProperties": False
            },
            "minItems": 5,
            "maxItems": 5
        }
    },
    "required": ["questions"],
    "additionalProperties": False
}


def create_ollama_request(prompt, num_predict=100, response_format=None):
    data = {
        "model": MODEL,
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


st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="centered"
)

st.title("📚 AI Study Buddy")
st.subheader("Your free, local AI learning assistant")
st.write("Ask questions, understand concepts, and practice learning.")
st.caption("Powered by Ollama + Gemma 3 1B — no paid API tokens.")

uploaded_pdf = st.file_uploader(
    "Upload your study notes (PDF)",
    type=["pdf"]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

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
        st.success("PDF uploaded successfully!")
        st.write("Pages:", st.session_state.pdf_pages)
        if not st.session_state.pdf_text.strip():
            st.warning("No readable text was found in this PDF.")
    except Exception as error:
        st.session_state.pdf_text = ""
        st.session_state.pdf_pages = 0
        st.error("Could not read this PDF.")
        st.caption(str(error))

mode = st.radio(
    "Choose your study mode:",
    ["💬 Study Chat", "📝 Quiz Mode"],
    horizontal=True
)

if mode == "💬 Study Chat":
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask your study question...")

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
                            "You are a friendly study assistant. "
                            "Answer the student's question using the PDF notes below. "
                            "If the notes do not contain the answer, say so.\n\n"
                            "PDF NOTES:\n"
                            + st.session_state.pdf_text[:12000]
                            + "\n\nSTUDENT QUESTION:\n"
                            + question
                        )
                    else:
                        prompt = (
                            "You are a friendly study assistant for beginners. "
                            "Explain clearly with simple words and examples.\n\n"
                            + question
                        )

                    answer = st.write_stream(
                        stream_ollama(create_ollama_request(prompt))
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as error:
                    st.error(
                        "Could not get an answer from Ollama. "
                        "Check that Ollama is running and Gemma 3 1B "
                        "is available."
                    )
                    st.caption(str(error))
else:
    if "quiz" not in st.session_state:
        st.session_state.quiz = []
    if "quiz_checked" not in st.session_state:
        st.session_state.quiz_checked = False

    if st.button("Generate 5 quiz questions"):
        st.session_state.quiz = []
        st.session_state.quiz_checked = False
        for i in range(5):
            st.session_state.pop(f"question_{i}", None)
        if uploaded_pdf is not None and st.session_state.pdf_text.strip():
            quiz_prompt = (
                "Create five beginner-friendly multiple-choice questions "
                "based on these notes. Include four options and the correct "
                "answer for each question.\n\n"
                + st.session_state.pdf_text[:12000]
            )
        else:
            quiz_prompt = (
                "Create five beginner-friendly multiple-choice questions "
                "about Python. Include four options and the correct answer "
                "for each question."
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
                            num_predict=1200,
                            response_format=QUIZ_RESPONSE_SCHEMA
                        )
                    )
                )
            quiz_status.update(label="Checking generated questions...")
            quiz_data = json.loads(quiz_json)
            questions = quiz_data.get("questions")
            if not isinstance(questions, list) or len(questions) != 5:
                raise ValueError("The generated quiz did not contain 5 questions.")
            for item in questions:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("question"), str)
                    or not isinstance(item.get("options"), list)
                    or len(item["options"]) != 4
                    or not all(isinstance(option, str) for option in item["options"])
                    or not isinstance(item.get("correct_answer"), str)
                    or item["correct_answer"] not in item["options"]
                ):
                    raise ValueError(
                        "The generated quiz has an invalid question or answer."
                    )
            st.session_state.quiz = questions
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
        st.subheader("Your quiz")
        for i, item in enumerate(st.session_state.quiz):
            st.radio(
                item["question"],
                item["options"],
                key=f"question_{i}"
            )

        if st.button("Submit Quiz"):
            st.session_state.quiz_checked = True

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
            total_questions = len(st.session_state.quiz)
            st.success(f"Your score: {score}/{total_questions}")
            st.write(f"Percentage: {score / total_questions * 100:.0f}%")