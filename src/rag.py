import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
DEFAULT_COLLECTION_NAME = "pdf_chunks"
DEFAULT_PDF_PATH = BASE_DIR / "document.pdf"
OUT_OF_CONTEXT_ANSWER = "Não tenho informações necessárias para responder sua pergunta."


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    return value.strip().strip("\"'")


def get_provider() -> str:
    provider = (_env("LLM_PROVIDER") or "").lower()
    if provider in {"openai", "gemini"}:
        return provider

    if _env("OPENAI_API_KEY"):
        return "openai"
    if _env("GOOGLE_API_KEY"):
        return "gemini"

    raise ValueError(
        "Defina LLM_PROVIDER como 'openai' ou 'gemini', ou informe uma API key valida."
    )


def get_pdf_path() -> Path:
    configured_path = _env("PDF_PATH")
    pdf_path = Path(configured_path) if configured_path else DEFAULT_PDF_PATH
    if not pdf_path.is_absolute():
        pdf_path = BASE_DIR / pdf_path
    return pdf_path


def get_database_url() -> str:
    return _env("DATABASE_URL", DEFAULT_DATABASE_URL) or DEFAULT_DATABASE_URL


def get_collection_name() -> str:
    return _env("PG_VECTOR_COLLECTION_NAME", DEFAULT_COLLECTION_NAME) or DEFAULT_COLLECTION_NAME


def get_embeddings():
    provider = get_provider()

    if provider == "openai":
        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY nao configurada.")
        return OpenAIEmbeddings(
            api_key=api_key,
            model=_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )

    api_key = _env("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY nao configurada.")
    return GoogleGenerativeAIEmbeddings(
        google_api_key=api_key,
        model=_env("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"),
    )


def get_llm():
    provider = get_provider()

    if provider == "openai":
        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY nao configurada.")
        return ChatOpenAI(
            api_key=api_key,
            model=_env("OPENAI_LLM_MODEL", "gpt-5-nano"),
            temperature=0,
        )

    api_key = _env("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY nao configurada.")
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=_env("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"),
        temperature=0,
    )


def get_vector_store(*, pre_delete_collection: bool = False) -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        connection=get_database_url(),
        collection_name=get_collection_name(),
        use_jsonb=True,
        create_extension=True,
        pre_delete_collection=pre_delete_collection,
    )
