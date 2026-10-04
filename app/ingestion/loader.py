from pathlib import Path
import hashlib
import re
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

logger=logging.getLogger(__name__)

def compute_doc_id(path:Path) -> str:
    hasher= hashlib.sha1()
    with open(path,"rb") as file :
        for block in iter(lambda:file.read(8192),b""):
            hasher.update(block)

    return hasher.hexdigest()[:12]

def clean_text(text:str,dehyphenate: bool=False) -> str:
    text =text.replace("\x00", "")

    if dehyphenate:
        text = re.sub(r"(\w)-\n([a-z])",r"\1\2",text)

    text=re.sub(r"[ \t]+", " ",text)
    text=re.sub(r"\n{3,}","\n\n",text)
    return text.strip()



def load_pdf(file_path:str, dehyphenate: bool = False):
    path=Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f" PDF not found : {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path.name}")

    documents=PyPDFLoader(str(path)).load()
    doc_id = compute_doc_id(path)

    kept=[]
    empty_pages=[]

    for document in documents:
        page=document.metadata.get("page")
        text=clean_text(document.page_content, dehyphenate)

        if not text:
            empty_pages.append(page + 1 if page is not None else None)
            continue

        document.page_content = text
        document.metadata["source"] = path.name
        document.metadata["file_path"] = str(path)
        document.metadata["doc_id"] = doc_id
        document.metadata["page_number"] = (page +1 if page is not None else None)

        kept.append(document)

    if empty_pages:
          logger.warning(
            "Skipped %d empty pages in %s: %s",
            len(empty_pages),
            path.name,
            empty_pages,

        )
    if not kept :
            raise ValueError(
                f"No extractable text in {path.name}."
                "If may be scanned PDF and need OCR. "

            )
    logger.info("Loaded %d pages from %s",len(kept),path.name)
    return kept

if __name__ == "__main__":
    d = load_pdf("data/documents/rag_survey.pdf")
    print(len(d))
