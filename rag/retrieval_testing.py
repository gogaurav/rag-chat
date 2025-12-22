from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local("vectorstores/gitlab_faiss", emb, allow_dangerous_deserialization=True)
query = "What are GitLab's rules about expenses?"
query = "What are the guidelines to be followed in working remote?"
docs = db.similarity_search(query, k=3)

for d in docs:
    print("\n---\n", d.metadata["source"])
    print(d.page_content[:500])
