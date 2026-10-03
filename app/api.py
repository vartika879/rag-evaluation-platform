import logging

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.schemas import AskRequest, AskResponse
from app.services.rag_service import answer_question


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Evaluation & Debugging Platform",
    version="1.0.0",
    description="Document-grounded question answering API",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )
    logger.info("Received RAG request; top_k=%s", request.top_k)

    try:
        return await run_in_threadpool(
            answer_question,
            request.question,
            request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("RAG request failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the question.",
        )