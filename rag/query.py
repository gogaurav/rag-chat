from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import litellm

# ---------------------------
# Config
# ---------------------------

VECTORSTORE_PATH = "../vectorstores/gitlab_faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 4


# ---------------------------
# Load Embeddings & Retriever
# ---------------------------

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

db = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": TOP_K})


# ---------------------------
# Prompt
# ---------------------------

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


# ---------------------------
# LiteLLM Runnable
# ---------------------------

def llm_call(inputs: dict) -> str:
    response = litellm.completion(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": inputs["prompt"]}],
    )
    return response["choices"][0]["message"]["content"]


llm_runnable = RunnableLambda(llm_call)


# ---------------------------
# Build RAG Pipeline
# ---------------------------

def build_rag_chain():

    def format_docs(docs):
        return "\n\n".join(
            f"{d.page_content}\n(Source: {d.metadata['source']})"
            for d in docs
        )

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(lambda x: {
            "prompt": prompt.format(
                question=x["question"],
                context=x["context"]
            )
        })
        | llm_runnable
        | StrOutputParser()
    )


rag_chain = build_rag_chain()


# ---------------------------
# Public function
# ---------------------------

def ask_rag(question: str) -> str:
    return rag_chain.invoke(question)
