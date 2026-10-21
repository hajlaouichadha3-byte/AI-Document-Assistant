
# AI Document Assistant Pro 📚

An interactive application that helps users explore PDF documents, find relevant answers, and generate extractive summaries.

## Features

- 📄 Upload PDF documents.
- 🔍 Ask questions and retrieve relevant sentences using TF-IDF and cosine similarity.
- 💬 Keep a conversation history during the current session.
- 📥 Export chat history to a text file.
- 🧠 Generate extractive summaries.
- 📊 Display document statistics, including pages, words, and characters.
- 📖 Preview extracted PDF text.
- 📥 Download summaries.

## Technologies

- Python
- Streamlit
- PyPDF
- Scikit-learn
- TF-IDF
- Cosine Similarity

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the application

Open a terminal in the project directory and run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Project Structure

```text
AI-Document-Assistant/
├── app.py
├── ai_document_assistant.ipynb
├── document_test.pdf
├── requirements.txt
└── README.md
```

## How It Works

The application extracts text from uploaded PDFs and uses TF-IDF vectorization with cosine similarity to retrieve the sentence most relevant to a user's question.

The summary feature selects important sentences from the document. It is an extractive summarizer, not a generative language model.

## Limitations

- Scanned PDFs may require OCR.
- Answers depend on the text available in the uploaded document.
- Similarity scores do not measure factual correctness.
- The current version does not use an external LLM.

## Future Improvements

- Add OCR support for scanned PDFs.
- Improve multilingual document processing.
- Integrate a language model for more flexible answers.
- Add document comparison and richer analytics.

## Author

Chadha Hajlaoui

Data Science Student
