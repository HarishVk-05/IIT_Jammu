from __future__ import annotations
import logging

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from iit_jammu.config import CHROMA_DIR, EMBED_MODEL

from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    try:
        return HuggingFaceEmbeddings(model_name = EMBED_MODEL)
    except Exception as exc:
        logger.error("Failed to load embedding model '%s':%s", EMBED_MODEL, exc)
        raise RuntimeError(
            f"Could not load embedding model '{EMBED_MODEL}'. "
            "Check your environment and internet connection."
        ) from exc
 
@lru_cache(maxsize=1)
def _get_vectordb() -> Chroma:
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_DIR}. "
            "Run ingest/build_vector_store.py first."
        )
    
    try:
        return Chroma(
            collection_name="Argo_profiles",
            embedding_function=_get_embeddings(),
            persist_directory=str(CHROMA_DIR)
        )
    except Exception as exc:
        logger.error("Failed to load Chroma vector store: %s", exc)
        raise RuntimeError(
            "Could not connect to the vector store. "
            "Try rebuilding it with build_vector_store.py."
        ) from exc
    

def get_retriever(k: int = 4):
    vectordb = _get_vectordb()
    return vectordb.as_retriever(search_kwargs = {"k": k})

def safe_retrieve(question: str, k: int = 4) -> tuple[list[Document], str]:
    try:
        retriever = get_retriever(k = k)
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        return docs, context
    except FileNotFoundError:
        logger.warning("Vector store missing - running without RAG context.")
        return [], ""
    except Exception as exc:
        logger.warning("Retrieval failed (%s) - running without RAG context.", exc)
        return [], ""