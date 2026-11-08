# Document RAG System

A local Retrieval-Augmented Generation project built with Python and Chroma. The corpus is a fictional 12-document Atlas Lantern Library, about 15-20 pages of original Markdown content. It demonstrates ingestion, chunking, embedding, retrieval, grounded generation, and the difference between RAG and a plain LLM call.

## How RAG works

1. Markdown is loaded from [documents](documents/).
2. Paragraphs are joined into chunks of about 500 words with 80 words of overlap.
3. Each chunk is converted into a 384-dimensional semantic embedding with `all-MiniLM-L6-v2`.
4. Chunks and embeddings are stored in a persistent local Chroma collection.
5. The question receives the same embedding and Chroma returns the nearest chunks.
6. Those chunks are placed in a prompt and sent to an optional OpenAI-compatible or Ollama model.
7. Without a configured model, the CLI uses an offline extractive answer from the retrieved text.

## Setup

Run these commands in PowerShell from the project folder:

```

## Streamlit document chatbot

Run the browser interface from the project folder:

```powershell
python -m pip install -r requirements.txt
python ingest.py
python -m streamlit run app.py
```

Place your PDF, Markdown, or text file in the `documents` folder and run `python ingest.py`.
The app will answer questions from Chroma and display the source document names.powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python rag_cli.py ingest

# Replace the sample corpus with one of your own files or folders
python rag_cli.py ingest "C:\Users\you\Documents\my-notes.md"
python rag_cli.py ingest "C:\Users\you\Documents\research-folder"
```

Python 3.13 is also a good option if a package is unavailable on 3.14:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Terminal commands

```powershell
# Build or rebuild Chroma and ingest all documents
python rag_cli.py ingest

# Inspect the chunks retrieved for a question
python rag_cli.py retrieve "What checksum phrase opens the Atlas Lantern archive?" --top-k 3

# Answer with retrieved context
python rag_cli.py ask "What checksum phrase opens the Atlas Lantern archive?"

# Compare grounded and plain answers
python rag_cli.py ask "What checksum phrase opens the Atlas Lantern archive?" --compare

# Run retrieval smoke tests
python test_rag.py
```

Optional model setup:

```powershell
$env:LLM_API_KEY="your-key"
$env:LLM_MODEL="gpt-4o-mini"
```

Or use an Ollama server running on `http://localhost:11434`:

```powershell
$env:OLLAMA_MODEL="llama3.2"
```

An offline answer does not require either service.

## With RAG vs. without RAG

Question: **"What checksum phrase opens the Atlas Lantern archive?"**

This fact appears in the project document `documents/02_glass_tide.md`, not in general knowledge:

- With RAG: `ember-violet seven-cast`.
- Plain LLM without retrieval: it should say that the information is not available rather than inventing a phrase.

Retrieved evidence:

```text
Source: 02_glass_tide.md
Access to the restricted Atlas Lantern digital companion is controlled by the archive checksum phrase.
The required phrase is "ember-violet seven-cast".
```

The comparison only demonstrates genuinely different behavior with an API or Ollama model if that provider is configured. In offline mode the project deliberately prints the fallback text `Not enough information: this fact is not from general knowledge.` for the plain call.

## Repository structure

```text
documents/       Source documents
rag/             Chunking, embeddings, retrieval, generation
rag_cli.py       Terminal entry point
test_rag.py      Retrieval smoke tests
chroma_db/       Local persisted vector database
```
