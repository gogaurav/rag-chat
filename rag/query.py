from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import litellm

# --------------------------------------------------
# Configuration
# --------------------------------------------------

VECTORSTORE_PATH = "../vectorstores/gitlab_faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# LLM_MODEL = "gpt-4o-mini"
LLM_MODEL = "gpt-5"
TEMPERATURE = 0.0
TOP_K = 4

# --------------------------------------------------
# Load embeddings and vectorstore (ONCE)
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

db = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": TOP_K})

# --------------------------------------------------
# Prompt template (SINGLE SOURCE)
# --------------------------------------------------

RAG_PROMPT = """
You are an AI assistant answering questions using ONLY the provided context.

Rules:
- If the answer is not in the context, say:
  "I could not find this in the GitLab handbook."
- Be concise and factual.
- Cite sources like: (Source: filename.md)

Question:
{question}

Context:
{context}

Answer:
"""

prompt = PromptTemplate.from_template(RAG_PROMPT)

# --------------------------------------------------
# LiteLLM call (REUSED)
# --------------------------------------------------

def llm_call(prompt_text: str) -> str:
    response = litellm.completion(
        model=LLM_MODEL,
        # temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return response["choices"][0]["message"]["content"]

# --------------------------------------------------
# Document formatting (REUSED)
# --------------------------------------------------

def format_docs(docs) -> Dict[str, object]:
    """
    Returns:
    {
        "context": str,
        "sources": List[str]
    }
    """
    context_chunks = []
    sources = []

    for d in docs:
        context_chunks.append(
            f"{d.page_content}\n(Source: {d.metadata.get('source', 'unknown')})"
        )
        if "source" in d.metadata:
            sources.append(d.metadata["source"])

    return {
        "context": "\n\n".join(context_chunks),
        "sources": sorted(set(sources))
    }

# --------------------------------------------------
# LCEL RAG pipeline (SINGLE PIPELINE)
# --------------------------------------------------

def build_rag_chain():
    """
    Returns a runnable that outputs:
    {
        "answer": str,
        "sources": List[str]
    }
    """

    def build_prompt(inputs: dict):
        return {
            "prompt": prompt.format(
                question=inputs["question"],
                context=inputs["context"]
            ),
            "sources": inputs["sources"]
        }

    
    def generate_answer(inputs: dict):
        answer = llm_call(inputs["prompt"])
        return {
            "answer": answer,
            "sources": inputs["sources"],
            "prompt": inputs["prompt"]   # 👈 expose full prompt
        }


    return (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(
            lambda x: {
                "question": x["question"],
                **format_docs(x["docs"])
            }
        )
        | RunnableLambda(build_prompt)
        | RunnableLambda(generate_answer)
    )

# Build once
rag_chain = build_rag_chain()

# --------------------------------------------------
# Public API function (REUSED EVERYWHERE)
# --------------------------------------------------

def ask_rag(question: str) -> Dict[str, object]:
    """
    Returns:
    {
        "answer": str,
        "sources": List[str]
    }
    """
    return rag_chain.invoke(question)
