import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_experimental.text_splitter import SemanticChunker
from app.embeddings.service import create_embedding_model

HEADING_PATTERN = re.compile(
    r"^\d{1,2}(?:\.\d{1,2}){0,3}\s+[A-Z][^\n]{2,90}$"
)

def _validate(chunk_size,chunk_overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size."

        )
def _finalize(chunks,strategy):
    chunks=[
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]   
    for index,chunk in enumerate(chunks):
        doc_id=chunk.metadata.get("doc_id","doc")

        chunk.metadata["chunk_id"] = index
        chunk.metadata["chunk_uid"] = f"{doc_id}:{strategy}:{index}"
        chunk.metadata["chunk_strategy"] = strategy

    return chunks


def create_recursive_chunks(
        documents,
        chunk_size=1000,
        chunk_overlap=150
        ):

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index = True,
    )
    return _finalize(splitter.split_documents(documents), "recursive")


    

   


def create_fixed_chunks(documents,chunk_size=1000,chunk_overlap=150):

    _validate(chunk_size,chunk_overlap)
    step = chunk_size - chunk_overlap
    chunks=[]

    for document in documents:
        text=document.page_content
        start=0

        while start < len(text):
            end = start + chunk_size
            piece =text[start:end]

            if piece.strip():
                metadata = document.metadata.copy()
                metadata["start_index"] = start

                chunks.append(
                    Document(
                        page_content = piece,
                        metadata=metadata,
                    )
                )
            if end >= len(text):
                break

            start += step
    return _finalize(chunks,"fixed")




def create_semantic_chunks(
        documents,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95):
    embedding_model = create_embedding_model()

    splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
    )

    return _finalize(splitter.split_documents(documents), "semantic")



def _is_heading(line):
    return bool(HEADING_PATTERN.match(line)) and not line.endswith(
        (".", ",", ";", ":")
    )

def create_structure_aware_chunks(
    documents,
    chunk_size=1500,
    chunk_overlap=150,
    min_section_chars=80
):
    _validate(chunk_size,chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    current_section = None

   
    def flush(text,base_metadata,section):
        text = text.strip()

        if not text:
            return

        metadata = base_metadata.copy()
        metadata["section"] = section

        piece = Document(page_content=text, metadata=metadata)

        if len(text) <= chunk_size:
            chunks.append(
                piece
            )
        else:
            chunks.extend(splitter.split_documents([piece]))

    for document in documents:
        current_text= ""

        lines = document.page_content.splitlines()

        for raw_line in document.page_content.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            if _is_heading(line):
                if len(current_text.strip()) >= min_section_chars:
                    flush(current_text, document.metadata, current_section)
                    current_text =""

                current_section = line
                current_text += line + "\n"

            else:
                current_text += line + "\n"
        flush(current_text, document.metadata, current_section)

    return _finalize(chunks, "structure_aware")



CHUNKING_STRATEGIES = {
    "recursive": create_recursive_chunks,
    "fixed": create_fixed_chunks,
    "semantic": create_semantic_chunks,
    "structure_aware": create_structure_aware_chunks,
}

def create_chunks(documents, strategy="recursive", **params):
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Choose from: {list(CHUNKING_STRATEGIES)}"
        )

    return CHUNKING_STRATEGIES[strategy](documents, **params)