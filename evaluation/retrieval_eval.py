import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

import litellm

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------------------------------------
# Configuration
# --------------------------------------------------

VECTORSTORE_PATH = "vectorstores/gitlab_faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVAL_K = 5

JUDGE_MODEL = "gpt-4o"
JUDGE_TEMPERATURE = 0.0

OUTPUT_JSON = "evaluation/retrieval_eval_results.json"
OUTPUT_CSV = "evaluation/retrieval_eval_results.csv"

# --------------------------------------------------
# Evaluation Queries
# --------------------------------------------------

EVAL_QUERIES = pd.read_csv("evaluation/questions.csv")['question']

# --------------------------------------------------
# Load Vectorstore
# --------------------------------------------------

print("Loading vector store...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

db = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": RETRIEVAL_K})

# --------------------------------------------------
# LLM Judge Prompt
# --------------------------------------------------

JUDGE_PROMPT = """
You are evaluating document retrieval quality for a RAG system.

Query:
"{query}"

Retrieved Document Chunk:
"{chunk}"

Question:
Is this chunk useful for answering the query?

Respond with ONLY one of the following labels:
- Highly Relevant
- Relevant
- Not Relevant
"""

# --------------------------------------------------
# Judge Function
# --------------------------------------------------

def judge_relevance(query: str, chunk: str) -> str:
    prompt = JUDGE_PROMPT.format(query=query, chunk=chunk[:2000])

    response = litellm.completion(
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}]
    )

    verdict = response["choices"][0]["message"]["content"].strip()

    if verdict not in {"Highly Relevant", "Relevant", "Not Relevant"}:
        return "Not Relevant"

    return verdict

# --------------------------------------------------
# Metric Helpers
# --------------------------------------------------

def compute_metrics(judgements: List[str]) -> Dict[str, float]:
    """
    judgements: list like ["Relevant", "Not Relevant", ...]
    """
    relevant_indices = [
        i for i, j in enumerate(judgements)
        if j in ("Relevant", "Highly Relevant")
    ]

    recall = 1.0 if relevant_indices else 0.0
    precision = len(relevant_indices) / len(judgements)

    mrr = 0.0
    if relevant_indices:
        mrr = 1.0 / (relevant_indices[0] + 1)

    return {
        "recall@k": recall,
        "precision@k": precision,
        "mrr": mrr
    }

# --------------------------------------------------
# Evaluation Loop
# --------------------------------------------------

all_results = []

print("Starting retrieval evaluation...\n")

for query in tqdm(EVAL_QUERIES, desc="Evaluating Queries"):
    docs = retriever.get_relevant_documents(query)

    judgements = []
    chunk_results = []

    for rank, doc in enumerate(docs, start=1):
        verdict = judge_relevance(query, doc.page_content)

        judgements.append(verdict)

        chunk_results.append({
            "rank": rank,
            "source": doc.metadata.get("source", "unknown"),
            "verdict": verdict,
            "chunk_preview": doc.page_content[:300]
        })

    metrics = compute_metrics(judgements)

    result = {
        "query": query,
        "metrics": metrics,
        "chunks": chunk_results
    }

    all_results.append(result)

# --------------------------------------------------
# Save Results
# --------------------------------------------------

print("\nSaving results...")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

rows = []
for r in all_results:
    for c in r["chunks"]:
        rows.append({
            "query": r["query"],
            "rank": c["rank"],
            "source": c["source"],
            "verdict": c["verdict"],
            "recall@k": r["metrics"]["recall@k"],
            "precision@k": r["metrics"]["precision@k"],
            "mrr": r["metrics"]["mrr"]
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

# --------------------------------------------------
# Print Summary
# --------------------------------------------------

avg_recall = sum(r["metrics"]["recall@k"] for r in all_results) / len(all_results)
avg_precision = sum(r["metrics"]["precision@k"] for r in all_results) / len(all_results)
avg_mrr = sum(r["metrics"]["mrr"] for r in all_results) / len(all_results)

print("\n====== RETRIEVAL EVALUATION SUMMARY ======")
print(f"Avg Recall@{RETRIEVAL_K}:     {avg_recall:.3f}")
print(f"Avg Precision@{RETRIEVAL_K}:  {avg_precision:.3f}")
print(f"Avg MRR:                     {avg_mrr:.3f}")
print("=========================================\n")
