from fastapi import FastAPI
from pydantic import BaseModel
from query import ask_rag

app = FastAPI(
    title="GitLab Handbook RAG API",
    description="FastAPI-based RAG service using FAISS + LangChain + LiteLLM",
    version="1.0.0"
)


# ---------------------------
# Request / Response Models
# ---------------------------

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    prompt: str


# ---------------------------
# Routes
# ---------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask",
        #    response_model=AskResponse
           )
def ask_question(req: AskRequest):
    answer = ask_rag(req.question)
    return answer
    # return AskResponse(answer=answer)

