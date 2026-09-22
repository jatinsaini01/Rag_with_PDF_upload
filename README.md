# RAG with PDF

A Streamlit application that lets users upload PDF files and ask questions about their contents. It uses LangChain, Chroma, HuggingFace embeddings, and a Groq language model.

## Features

- Upload one or more PDF documents.
- Split document text into chunks and store embeddings in Chroma.
- Ask questions using retrieval-augmented generation (RAG).
- Keep separate chat history for each session ID.

## Setup

1. Create and activate a Python virtual environment.
2. Install the existing dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`.
4. Add your Groq API key to `.env`.
5. Start the application:

   ```bash
   streamlit run app.py
   ```

## Security

Never upload `.env`, `venv`, or temporary PDF files. `.gitignore` excludes them from GitHub.
