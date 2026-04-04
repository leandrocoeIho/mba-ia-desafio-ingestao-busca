import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def _exit_with_error(message: str) -> None:
    print(message)
    raise SystemExit(1)


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        _exit_with_error(f"Variavel de ambiente obrigatoria ausente: {name}")
    return value


def _validate_database_url(database_url: str) -> None:
    if not database_url.startswith("postgresql+psycopg://"):
        _exit_with_error("DATABASE_URL invalida: use o prefixo postgresql+psycopg://")


def _pages_have_text(pages: list) -> bool:
    return any((page.page_content or "").strip() for page in pages)


def _ensure_source_metadata(documents: list, pdf_path: str) -> None:
    # Preserve source traceability even when the loader/splitter returns sparse metadata.
    for document in documents:
        metadata = getattr(document, "metadata", None)
        if metadata is None:
            document.metadata = {"source": pdf_path}
            continue
        metadata.setdefault("source", pdf_path)


def ingest_pdf():
    pdf_path = _get_required_env("PDF_PATH")
    database_url = _get_required_env("DATABASE_URL")
    _get_required_env("OPENAI_API_KEY")
    embedding_model = _get_required_env("OPENAI_EMBEDDING_MODEL")
    collection_name = _get_required_env("PG_VECTOR_COLLECTION_NAME")

    _validate_database_url(database_url)

    if not Path(pdf_path).is_file():
        _exit_with_error(f"PDF nao encontrado: {pdf_path}")

    try:
        pages = PyPDFLoader(pdf_path).load()
        if not pages:
            _exit_with_error("PDF sem paginas ou sem conteudo extraivel.")
        if not _pages_have_text(pages):
            _exit_with_error("PDF sem texto extraivel.")

        print(f"Paginas carregadas: {len(pages)}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
        if not chunks:
            _exit_with_error("Nenhum chunk foi gerado a partir do PDF.")

        _ensure_source_metadata(chunks, pdf_path)
        print(f"Chunks gerados: {len(chunks)}")

        embeddings = OpenAIEmbeddings(model=embedding_model)
        PGVector.from_documents(
            chunks,
            embeddings,
            connection=database_url,
            collection_name=collection_name,
            # Re-ingesting the same PDF must replace the collection contents instead of duplicating rows.
            pre_delete_collection=True,
        )

        print("Ingestao concluida com sucesso.")
    except SystemExit:
        raise
    except Exception as exc:
        _exit_with_error(f"Falha na ingestao do PDF: {exc}")


if __name__ == "__main__":
    ingest_pdf()
