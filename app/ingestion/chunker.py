from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_experimental.text_splitter import SemanticChunker
from app.embeddings.service import create_embedding_model

def create_recursive_chunks(documents):

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
       
    )

    

    return splitter.split_documents(documents=documents)


def create_fixed_chunks(documents,chunk_size=1000,chunk_overlap=150):
    chunks=[]

    for document in documents:
        text=document.page_content
        start=0

        while start < len(text):
            end=start + chunk_size
            chunk_text=text[start:end]

            if chunk_text.strip():
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=document.metadata.copy(),
                    )
                )

            start += chunk_size - chunk_overlap

    return chunks


def create_semantic_chunks(documents):
    embedding_model=create_embedding_model()

    splitter=SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    return splitter.split_documents(documents)







import re

from langchain_core.documents import Document


def create_structure_aware_chunks(
    documents,
    chunk_size=1500,
    chunk_overlap=150,
):
    chunks = []

    # Matches headings such as:
    # 1 Introduction
    # 3 Retriever
    # 3.1 Building the Retriever
    # 3.1.1 Chunking Corpus
    heading_pattern = re.compile(
        r"^\d+(?:\.\d+)*\s+[A-Z][^\n]*$"
    )

    current_section = None
    current_text = ""

    def save_chunk(text, metadata):
        text = text.strip()

        if not text:
            return

        if len(text) <= chunk_size:
            chunks.append(
                Document(
                    page_content=text,
                    metadata=metadata.copy(),
                )
            )
            return

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=metadata.copy(),
                    )
                )

            start += chunk_size - chunk_overlap

    for document in documents:
        lines = document.page_content.splitlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if heading_pattern.match(line):
                if current_text:
                    metadata = document.metadata.copy()
                    metadata["section"] = current_section

                    save_chunk(current_text, metadata)

                current_section = line
                current_text = line + "\n"
            else:
                current_text += line + "\n"

        if current_text:
            metadata = document.metadata.copy()
            metadata["section"] = current_section

            save_chunk(current_text, metadata)

            current_text = ""

    return chunks