import json
import os
import time
from pathlib import Path

from app.embeddings.service import EMBEDDING_DIM,EMBEDDING_MODEL
from app.vectorstore.qdrant_store import PROJECT_ROOT

REGISTRY_PATH= Path(
    os.getenv("REGISTRY_PATH", str(PROJECT_ROOT / "data" / "registry.json"))
)

def _load():
    if not REGISTRY_PATH.exists():
        return {"collection": {},"aliases":{}}

    with open(REGISTRY_PATH,"r",encoding="utf-8") as file:
        return json.load(file)

def _save(data):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = REGISTRY_PATH.with_suffix(".tmp")

    with open(temp_path, "w", encoding ="utf-8") as file:
        json.dump(data, file, indent=2)

    temp_path.replace(REGISTRY_PATH)



def register_collection(
    name,
    strategy,
    chunk_size,
    chunk_overlap,
    doc_id,
    source,
    chunk_count,
):
    data = _load()
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    info = data["collections"].setdefault(
        name,
        {
            "strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "created_at": now,
            "documents": {},
        },
    )

    info["documents"][doc_id] = {
        "source": source,
        "chunk_count": chunk_count,
    }
    info["chunk_count"] = sum(
        document["chunk_count"] for document in info["documents"].values()
    )
    info["updated_at"] = now

    data["aliases"].setdefault("default", name)

    _save(data)

    return info

def list_registered():
    return _load()["collections"]

def get_colection_info(name_or_alias):
    data = _load()
    name = data["aliases"].get(name_or_alias, name_or_alias)

    return data["collections"].get(name)


def resolve_collection(name_or_alias):
    return _load()["aliases"].get(name_or_alias, name_or_alias)



def set_alias(alias, name):
    data = _load()

    if name not in data["collections"]:
        raise ValueError(f"Collection '{name}' is not registered.")

    data["aliases"][alias] = name
    _save(data)

def unregister_collection(name):
    data = _load()

    data["collections"].pop(name, None)
    data["aliases"] = {
        alias: target
        for alias, target in data["aliases"].items()
        if target != name
    }

    _save(data)