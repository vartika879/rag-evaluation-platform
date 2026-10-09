from dataclasses import dataclass
from typing import Any,Dict,List,Optional


STAGES =("vector","bm25","hybrid","rerank")

@dataclass
class RetrievedChunk:
    text:str

    #identity
    chunk_id:Optional[int] = None
    chunk_uid:Optional[str] =None
    point_id:Optional[str] =None
    doc_id:Optional[str] =None
    source:Optional[str] =None
    page:Optional[int] =None
    page_number:Optional[int] =None
    section:Optional[str] =None
    chunk_strategy:Optional[str] =None


       # which retriever first produced this chunk
    method:str = "vector"

    # per-stage scores (None = that stage did not score this chunk)

    vector_score:Optional[float] = None
    bm25_score:Optional[float] = None
    hybrid_score:Optional[float] = None
    rerank_score:Optional[float] = None


# per-stage ranks, 1 = best (None = not ranked by that stage)
    vector_score:Optional[float] = None
    bm25_score:Optional[float] = None
    hybrid_score:Optional[float] = None
    rerank_score:Optional[float] = None
    
 # ------------------------------------------------------------------
    # Final score / rank = value from the most advanced stage that ran
    # ------------------------------------------------------------------

    @property
    def score(self)-> Optional[float]:
        for stage in reversed(STAGES):
            value = getattr(self,f"{stage}_score")

            if value is not None:
                return value

        return None


    @property
    def rank(self) -> Optional[int]:
        for stage in reversed(STAGES):
            value = getattr(self,f"{stage}_rank")

            if value is not None:
                return value

        return None


      # ------------------------------------------------------------------
    # Compatibility with older modules that expect Qdrant point objects
    # (crag.py, context_compressor.py, rag_service.py use .payload/.id)
    # ------------------------------------------------------------------

    @property
    def id(self)->Optional[str]:
        return self.point_id

    @property
    def payload(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "page_number": self.page_number,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "chunk_uid": self.chunk_uid,
            "doc_id": self.doc_id,
            "chunk_strategy": self.chunk_strategy,
        }


    @classmethod
    def from_payload(
        cls,
        payload:Optional[Dict[str,Any]],
        point_id:Any=None,
        method:str="vector",

    ) -> "RetrievedChunk":
        payload = payload or {}



        page = payload.get("page")
        page_number = payload.get("page_number")

        if page_number is None and page is not None:
            page_number = page + 1

        return cls(
            text=payload.get("text", "") or "",
            chunk_id=payload.get("chunk_id"),
            chunk_uid=payload.get("chunk_uid"),
            point_id=str(point_id) if point_id is not None else None,
            doc_id=payload.get("doc_id"),
            source=payload.get("source"),
            page=page,
            page_number=page_number,
            section=payload.get("section"),
            chunk_strategy=payload.get("chunk_strategy"),
            method=method,
        
        )

    if page_number is None and page is not None:
        page_number = page + 1


    @classmethod
    def from_qdrant_point(cls, point, method: str = "vector") -> "RetrievedChunk":
        chunk = cls.from_payload(point.payload, point_id=point.id, method=method)
        chunk.vector_score = float(point.score)
        return chunk


    @classmethod
    def from_langchain_chunk(cls, chunk, method: str = "bm25") -> "RetrievedChunk":
        """Build from a LangChain Document (used by BM25, which indexes
        chunks in memory instead of reading them from Qdrant)."""
        metadata = chunk.metadata or {}

        payload = {
            "text": chunk.page_content,
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "page_number": metadata.get("page_number"),
            "section": metadata.get("section"),
            "chunk_id": metadata.get("chunk_id"),
            "chunk_uid": metadata.get("chunk_uid"),
            "doc_id": metadata.get("doc_id"),
            "chunk_strategy": metadata.get("chunk_strategy"),
        }

        return cls.from_payload(payload, method=method)

    @property
    def key(self) -> Any:
        """Identity used to match the same chunk across stages."""
        return self.chunk_uid if self.chunk_uid is not None else self.chunk_id


    def preview(self, chars: int = 200) -> str:
        text = " ".join(self.text.split())
        return text if len(text) <= chars else text[:chars].rstrip() + "..."


    def rank_change(self, from_stage: str = "vector", to_stage: str = "rerank"):
        """Positive = moved up (e.g. rank 4 -> rank 1 gives +3). None if
        either stage did not rank this chunk."""
        before = getattr(self, f"{from_stage}_rank")
        after = getattr(self, f"{to_stage}_rank")

        if before is None or after is None:
            return None

        return before - after


    def to_dict(self, include_text: bool = True, preview_chars: Optional[int] = None):
        data = {
            "chunk_id": self.chunk_id,
            "chunk_uid": self.chunk_uid,
            "doc_id": self.doc_id,
            "source": self.source,
            "page": self.page,
            "page_number": self.page_number,
            "section": self.section,
            "method": self.method,
            "score": self.score,
            "rank": self.rank,
        }

        for stage in STAGES:
            data[f"{stage}_score"] = getattr(self, f"{stage}_score")
            data[f"{stage}_rank"] = getattr(self, f"{stage}_rank")

        if include_text:
            data["text"] = self.text
        elif preview_chars:
            data["preview"] = self.preview(preview_chars)

        return data

def assign_ranks(chunks: List[RetrievedChunk], stage: str) -> List[RetrievedChunk]:
    """Write 1-based ranks for `stage`, following the current list order.
    Call this right after a stage has sorted its results."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Choose from: {STAGES}")

    for position, chunk in enumerate(chunks, start=1):
        setattr(chunk, f"{stage}_rank", position)

    return chunks

    