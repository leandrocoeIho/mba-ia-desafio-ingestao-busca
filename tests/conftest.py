import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234567890abcdef")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("PG_VECTOR_COLLECTION_NAME", "documents")
    monkeypatch.setenv("PDF_PATH", "document.pdf")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.5")
