"""
ingest_handbook.py

Ingests .md files from gitlab_handbook_cleaned/,
splits into text chunks, embeds using LiteLLM or HuggingFace,
and saves to FAISS or Chroma.

Usage:
    python ingest_handbook.py --input_dir gitlab_handbook_cleaned --db faiss
    python ingest_handbook.py --input_dir gitlab_handbook_cleaned --db chroma
"""

import argparse
import os
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Optional: LiteLLM embeddings (OpenAI-compatible)
# from langchain.embeddings import OpenAIEmbeddings
# import litellm

# -------------------------------
# Configuration
# -------------------------------

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

# Preferred: local HuggingFace model (fast & free)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -------------------------------
# Helper functions
# -------------------------------

def load_markdown_files(input_dir):
    """Yield Document objects from .md files."""
    input_dir = Path(input_dir)
    md_files = list(input_dir.rglob("*.md"))

    print(f"Found {len(md_files)} markdown files")

    for md_path in md_files:
        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Failed to read {md_path}: {e}")
            continue

        # Metadata includes filepath + title for debugging
        meta = {
            "source": str(md_path),
            "title": text.split("\n")[0].replace("#", "").strip()[:200],
        }

        yield Document(page_content=text, metadata=meta)


def create_text_chunks(docs, chunk_size, chunk_overlap):
    """Split long markdown pages into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)


def create_embedding_fn():
    """
    Return an embedding function.
    Option A: HuggingFace local model (default)
    Option B: LiteLLM/OpenAI embeddings (uncomment if required)
    """
    print(f"Using HuggingFace Embeddings: {EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Example using LiteLLM:
    # return OpenAIEmbeddings(model="text-embedding-3-large")


# -------------------------------
# Main Ingestion Process
# -------------------------------

def ingest_markdown(input_dir, db_choice="faiss"):
    docs = list(load_markdown_files(input_dir))

    print("Splitting into chunks...")
    chunks = create_text_chunks(
        docs,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    print(f"Created {len(chunks)} text chunks")

    embeddings = create_embedding_fn()

    os.makedirs("vectorstores1", exist_ok=True)

    if db_choice == "faiss":
        print("Building FAISS index...")
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local("vectorstores1/gitlab_faiss1")
        print("Saved FAISS index at vectorstores1/gitlab_faiss1")

    elif db_choice == "chroma":
        print("Building Chroma DB...")
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="vectorstores/gitlab_chroma"
        )
        vectordb.persist()
        print("Saved Chroma DB at vectorstores/gitlab_chroma")

    else:
        raise ValueError("db_choice must be 'faiss' or 'chroma'")

    return chunks


# -------------------------------
# CLI Entrypoint
# -------------------------------

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--input_dir", required=True, help="Directory with cleaned .md files")
    # parser.add_argument("--db", default="faiss", choices=["faiss", "chroma"], help="Vector DB type")

    # args = parser.parse_args()


    # ingest_markdown(args.input_dir, args.db)
    ingest_markdown('gitlab_handbook_cleaned', 'faiss')
