
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(
    page_title="AI Document Assistant Pro",
    page_icon="📚",
    layout="wide"
)

# ---------- DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: #f4f7fb;
}
.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}
.hero {
    background: linear-gradient(120deg, #173b66, #3978b8);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    margin-bottom: 8px;
}
.hero p {
    color: #edf5ff;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📚 AI Document Assistant Pro</h1>
    <p>Upload, explore, summarize, and ask questions about your PDF.</p>
</div>
""", unsafe_allow_html=True)

# ---------- FUNCTIONS ----------
def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return pages


def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if len(sentence.strip()) > 2
    ]


def find_answer(question, sentences):
    if not sentences:
        return None, 0.0

    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(
            sentences + [question]
        )

        scores = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        )[0]

        best_index = scores.argmax()

        return sentences[best_index], float(scores[best_index])

    except ValueError:
        return None, 0.0


def summarize_text(text, number_of_sentences=3):
    sentences = split_sentences(text)

    if not sentences:
        return "No readable text was found."

    if len(sentences) <= number_of_sentences:
        return " ".join(sentences)

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(sentences)
        importance = matrix.sum(axis=1).A1

        best_indices = sorted(
            range(len(sentences)),
            key=lambda i: importance[i],
            reverse=True
        )[:number_of_sentences]

        best_indices.sort()

        return " ".join(
            sentences[i] for i in best_indices
        )

    except ValueError:
        return " ".join(sentences[:number_of_sentences])


# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Upload a text-based PDF to begin.")
    st.caption("Supported format: PDF")
    st.divider()
    st.write("AI Document Assistant Pro")
    st.caption("PDF analysis and text similarity")


# ---------- PDF UPLOAD ----------
uploaded_file = st.file_uploader(
    "📂 Upload your PDF document",
    type=["pdf"]
)

if uploaded_file is not None:

    try:
        pages = extract_pdf(uploaded_file)
        text = "\n".join(pages)

        words = re.findall(r"\b[\w'-]+\b", text)

        if not text.strip():
            st.error(
                "No readable text found. "
                "This PDF may contain scanned images."
            )
            st.stop()

        st.success(
            f"Document loaded successfully: {uploaded_file.name}"
        )

        # ---------- DOCUMENT STATISTICS ----------
        col1, col2, col3 = st.columns(3)

        col1.metric("📄 Pages", len(pages))
        col2.metric("🔤 Words", len(words))
        col3.metric("📝 Characters", len(text))

        # Keep history for the current Streamlit session
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if "summary_by_document" not in st.session_state:
            st.session_state.summary_by_document = {}

        # Reset history when a different PDF is uploaded
        if "current_document" not in st.session_state:
            st.session_state.current_document = None

        if st.session_state.current_document != uploaded_file.name:
            st.session_state.chat_history = []
            st.session_state.current_document = uploaded_file.name

        # ---------- TABS ----------
        tab1, tab2, tab3 = st.tabs([
            "💬 Ask Questions",
            "🧠 Summarize PDF",
            "📖 Explore Document"
        ])

        # ==========================================
        # TAB 1: QUESTION ANSWERING
        # ==========================================
        with tab1:

            st.subheader("💬 Ask your document")

            question = st.text_input(
                "Enter your question:",
                placeholder="Example: What is Python?",
                key="document_question"
            )

            if st.button("🔍 Get Answer", type="primary"):

                if not question.strip():
                    st.warning("Please enter a question.")

                else:
                    answer, score = find_answer(
                        question,
                        split_sentences(text)
                    )

                    if answer is None or score == 0:
                        st.warning(
                            "No relevant information was found "
                            "in this document."
                        )

                    else:
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer,
                            "score": score
                        })

            if st.session_state.chat_history:

                st.subheader("📥 Export conversation")

                chat_text = ""

                for item in st.session_state.chat_history:
                    chat_text += f"Question: {item['question']}\n"
                    chat_text += f"Answer: {item['answer']}\n"
                    chat_text += (
                        f"Text similarity: {item['score']:.2%}\n"
                    )
                    chat_text += "-" * 40 + "\n"

                st.download_button(
                    "📥 Export Chat History",
                    data=chat_text,
                    file_name="chat_history.txt",
                    mime="text/plain"
                )

                st.subheader("🗨️ Conversation History")

                for item in reversed(st.session_state.chat_history):

                    with st.container(border=True):
                        st.markdown("**Your question:**")
                        st.write(item["question"])

                        st.markdown("**Answer from the PDF:**")
                        st.info(item["answer"])

                        st.caption(
                            f"Text similarity: {item['score']:.2%}. "
                            "This is not a measure of factual correctness."
                        )

                if st.button("🗑️ Clear Conversation"):
                    st.session_state.chat_history = []
                    st.rerun()

            else:
                st.info(
                    "Ask a question to start your conversation."
                )

        # ==========================================
        # TAB 2: PDF SUMMARY
        # ==========================================
        with tab2:

            st.subheader("🧠 Automatic PDF Summary")

            st.write(
                "This feature selects important sentences from "
                "the document. It does not generate new information."
            )

            number = st.slider(
                "Number of sentences in the summary",
                min_value=1,
                max_value=10,
                value=3
            )

            if st.button("✨ Generate Summary"):

                summary = summarize_text(text, number)

                st.session_state.summary_by_document[
                    uploaded_file.name
                ] = summary

            summary = st.session_state.summary_by_document.get(
                uploaded_file.name
            )

            if summary:
                st.markdown("### 📝 Summary")
                st.write(summary)

                st.download_button(
                    "📥 Download Summary",
                    data=summary,
                    file_name="document_summary.txt",
                    mime="text/plain"
                )

        # ==========================================
        # TAB 3: DOCUMENT EXPLORER
        # ==========================================
        with tab3:

            st.subheader("📖 Explore Document")

            st.text_area(
                "Extracted text",
                value=text[:10000],
                height=350,
                disabled=True
            )

            with st.expander("View text page by page"):

                for index, page_text in enumerate(
                    pages,
                    start=1
                ):
                    st.markdown(f"**Page {index}**")
                    st.write(
                        page_text if page_text.strip()
                        else "No text extracted from this page."
                    )

    except Exception as error:
        st.error(f"Could not process this PDF: {error}")

else:
    st.info("👆 Upload a PDF to unlock all features.")
