import pytest
from unittest.mock import MagicMock

from ingest import ingest_pdf


# --- Helper ---

def _setup(mocker, pages=None, chunks=None):
    """Mocka todas as dependências externas do ingest_pdf para o happy path."""
    if pages is None:
        p = MagicMock()
        p.page_content = "Conteúdo real do PDF."
        p.metadata = {"source": "document.pdf", "page": 0}
        pages = [p]

    if chunks is None:
        c = MagicMock()
        c.page_content = "Conteúdo real do PDF."
        c.metadata = {"source": "document.pdf", "page": 0}
        chunks = [c]

    loader_inst = MagicMock()
    loader_inst.load.return_value = pages

    splitter_inst = MagicMock()
    splitter_inst.split_documents.return_value = chunks

    loader_cls = mocker.patch("ingest.PyPDFLoader", create=True)
    loader_cls.return_value = loader_inst

    splitter_cls = mocker.patch("ingest.RecursiveCharacterTextSplitter", create=True)
    splitter_cls.return_value = splitter_inst

    embeddings_cls = mocker.patch("ingest.OpenAIEmbeddings", create=True)
    pgvector_cls = mocker.patch("ingest.PGVector", create=True)

    return {
        "loader_cls": loader_cls,
        "loader_inst": loader_inst,
        "splitter_cls": splitter_cls,
        "splitter_inst": splitter_inst,
        "embeddings_cls": embeddings_cls,
        "pgvector_cls": pgvector_cls,
        "pages": pages,
        "chunks": chunks,
    }


# --- Story 2: Ingestão do PDF ---

def test_ingest_executes_successfully_with_valid_pdf(mocker):
    m = _setup(mocker)
    ingest_pdf()
    m["pgvector_cls"].from_documents.assert_called_once()


def test_ingest_loads_pdf_using_pypdf_loader(mocker):
    m = _setup(mocker)
    ingest_pdf()
    m["loader_cls"].assert_called_once_with("document.pdf")


def test_ingest_uses_chunk_size_1000(mocker):
    m = _setup(mocker)
    ingest_pdf()
    m["splitter_cls"].assert_called_once()
    assert m["splitter_cls"].call_args.kwargs.get("chunk_size") == 1000


def test_ingest_uses_chunk_overlap_150(mocker):
    m = _setup(mocker)
    ingest_pdf()
    m["splitter_cls"].assert_called_once()
    assert m["splitter_cls"].call_args.kwargs.get("chunk_overlap") == 150


def test_ingest_uses_embedding_model_from_env(mocker):
    m = _setup(mocker)
    ingest_pdf()
    m["embeddings_cls"].assert_called_once()
    assert m["embeddings_cls"].call_args.kwargs.get("model") == "text-embedding-3-small"


def test_ingest_calls_pgvector_from_documents(mocker):
    m = _setup(mocker)
    ingest_pdf()
    m["pgvector_cls"].from_documents.assert_called_once()


def test_ingest_uses_pre_delete_collection_true(mocker):
    m = _setup(mocker)
    ingest_pdf()
    m["pgvector_cls"].from_documents.assert_called_once()
    assert m["pgvector_cls"].from_documents.call_args.kwargs.get("pre_delete_collection") is True


def test_ingest_uses_correct_collection_name(mocker):
    m = _setup(mocker)
    ingest_pdf()
    m["pgvector_cls"].from_documents.assert_called_once()
    assert m["pgvector_cls"].from_documents.call_args.kwargs.get("collection_name") == "documents"


def test_ingest_uses_database_url_from_env(mocker):
    m = _setup(mocker)
    ingest_pdf()
    m["pgvector_cls"].from_documents.assert_called_once()
    assert m["pgvector_cls"].from_documents.call_args.kwargs.get("connection") == \
        "postgresql+psycopg://postgres:postgres@localhost:5432/rag"


def test_ingest_logs_page_count(mocker, capsys):
    pages = [MagicMock(page_content=f"página {i}", metadata={}) for i in range(5)]
    _setup(mocker, pages=pages)
    ingest_pdf()
    assert "5" in capsys.readouterr().out


def test_ingest_logs_chunk_count(mocker, capsys):
    chunks = [MagicMock(page_content=f"chunk {i}", metadata={}) for i in range(7)]
    _setup(mocker, chunks=chunks)
    ingest_pdf()
    assert "7" in capsys.readouterr().out


def test_ingest_logs_completion_status(mocker, capsys):
    _setup(mocker)
    ingest_pdf()
    assert capsys.readouterr().out.strip()


def test_ingest_missing_pdf_exits_with_nonzero_code(mocker, monkeypatch):
    monkeypatch.setenv("PDF_PATH", "/caminho/inexistente/arquivo.pdf")
    mocker.patch("ingest.PyPDFLoader", create=True)
    mocker.patch("ingest.RecursiveCharacterTextSplitter", create=True)
    mocker.patch("ingest.OpenAIEmbeddings", create=True)
    mocker.patch("ingest.PGVector", create=True)
    with pytest.raises(SystemExit) as exc_info:
        ingest_pdf()
    assert exc_info.value.code != 0


def test_ingest_missing_pdf_prints_clear_error_message(mocker, monkeypatch, capsys):
    monkeypatch.setenv("PDF_PATH", "/caminho/inexistente/arquivo.pdf")
    mocker.patch("ingest.PyPDFLoader", create=True)
    mocker.patch("ingest.RecursiveCharacterTextSplitter", create=True)
    mocker.patch("ingest.OpenAIEmbeddings", create=True)
    mocker.patch("ingest.PGVector", create=True)
    with pytest.raises(SystemExit):
        ingest_pdf()
    assert "/caminho/inexistente/arquivo.pdf" in capsys.readouterr().out


def test_ingest_image_pdf_prints_explicit_warning(mocker, capsys):
    page = MagicMock()
    page.page_content = ""
    page.metadata = {"source": "document.pdf", "page": 0}
    _setup(mocker, pages=[page])
    with pytest.raises(SystemExit):
        ingest_pdf()
    assert capsys.readouterr().out.strip()


def test_ingest_missing_pdf_path_env_var_exits_with_error(monkeypatch):
    monkeypatch.delenv("PDF_PATH")
    with pytest.raises(SystemExit) as exc_info:
        ingest_pdf()
    assert exc_info.value.code != 0


def test_ingest_missing_database_url_exits_with_error(monkeypatch):
    monkeypatch.delenv("DATABASE_URL")
    with pytest.raises(SystemExit) as exc_info:
        ingest_pdf()
    assert exc_info.value.code != 0


def test_ingest_missing_openai_api_key_exits_with_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(SystemExit) as exc_info:
        ingest_pdf()
    assert exc_info.value.code != 0


def test_ingest_rerun_does_not_duplicate_records(mocker):
    m = _setup(mocker)
    ingest_pdf()
    ingest_pdf()
    assert m["pgvector_cls"].from_documents.call_count == 2
    for c in m["pgvector_cls"].from_documents.call_args_list:
        assert c.kwargs.get("pre_delete_collection") is True


def test_ingest_database_connection_failure_exits_with_error(mocker):
    from sqlalchemy.exc import OperationalError
    m = _setup(mocker)
    m["pgvector_cls"].from_documents.side_effect = OperationalError("conn", {}, Exception())
    with pytest.raises(SystemExit) as exc_info:
        ingest_pdf()
    assert exc_info.value.code != 0


def test_ingest_pdf_with_zero_pages_does_not_crash(mocker):
    _setup(mocker, pages=[], chunks=[])
    with pytest.raises(SystemExit):
        ingest_pdf()


def test_ingest_chunks_contain_source_metadata(mocker):
    chunk = MagicMock()
    chunk.page_content = "Conteúdo do chunk."
    chunk.metadata = {"source": "document.pdf", "page": 0}
    m = _setup(mocker, chunks=[chunk])
    ingest_pdf()
    m["pgvector_cls"].from_documents.assert_called_once()
    docs = m["pgvector_cls"].from_documents.call_args.args[0]
    assert all("source" in d.metadata for d in docs)


def test_ingest_invalid_database_url_prefix_exits_with_error(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")
    with pytest.raises(SystemExit) as exc_info:
        ingest_pdf()
    assert exc_info.value.code != 0


def test_ingest_chunks_have_faturamento_max_metadata(mocker):
    """Chunks com R$ devem ter faturamento_max no metadata."""
    chunk = MagicMock()
    chunk.page_content = "Alfa Energia S.A. R$ 722.875.391,46 1972\nAlfa IA Indústria R$ 548.789.613,65 2020"
    chunk.metadata = {"source": "document.pdf", "page": 0}
    m = _setup(mocker, chunks=[chunk])
    ingest_pdf()
    doc = m["pgvector_cls"].from_documents.call_args.args[0][0]
    assert "faturamento_max" in doc.metadata
    assert doc.metadata["faturamento_max"] == 722875391.46


def test_ingest_chunks_have_faturamento_min_metadata(mocker):
    """Chunks com R$ devem ter faturamento_min no metadata."""
    chunk = MagicMock()
    chunk.page_content = "Alfa Energia S.A. R$ 722.875.391,46 1972\nAlfa IA Indústria R$ 548.789.613,65 2020"
    chunk.metadata = {"source": "document.pdf", "page": 0}
    m = _setup(mocker, chunks=[chunk])
    ingest_pdf()
    doc = m["pgvector_cls"].from_documents.call_args.args[0][0]
    assert "faturamento_min" in doc.metadata
    assert doc.metadata["faturamento_min"] == 548789613.65


def test_ingest_chunks_without_values_have_no_faturamento_metadata(mocker):
    """Chunks sem R$ não devem receber metadata de faturamento."""
    chunk = MagicMock()
    chunk.page_content = "Nome da empresa Faturamento Ano de fundação"
    chunk.metadata = {"source": "document.pdf", "page": 0}
    m = _setup(mocker, chunks=[chunk])
    ingest_pdf()
    doc = m["pgvector_cls"].from_documents.call_args.args[0][0]
    assert "faturamento_max" not in doc.metadata
    assert "faturamento_min" not in doc.metadata

# --- Summary documents ---

def test_ingest_creates_summary_documents_with_global_stats(mocker):
    """Ingest deve gerar docs-resumo com max/min/top-10 globais."""
    chunks = [
        MagicMock(
            page_content="Empresa A R$ 1.000.000,00 2001\nEmpresa B R$ 500.000,00 2002",
            metadata={"source": "document.pdf", "page": 0},
        ),
        MagicMock(
            page_content="Empresa C R$ 2.000.000,00 1999\nEmpresa D R$ 100.000,00 2010",
            metadata={"source": "document.pdf", "page": 1},
        ),
    ]
    m = _setup(mocker, chunks=chunks)
    ingest_pdf()
    all_docs = m["pgvector_cls"].from_documents.call_args.args[0]
    summary_docs = [d for d in all_docs if getattr(d, "metadata", {}).get("type") == "summary"]
    assert len(summary_docs) >= 1
    combined_text = " ".join(d.page_content for d in summary_docs)
    assert "Empresa C" in combined_text
    assert "2.000.000" in combined_text
    assert "Empresa D" in combined_text
    assert "100.000" in combined_text


def test_ingest_summary_contains_top_10(mocker):
    """O doc-resumo de ranking deve conter top 10 por faturamento."""
    lines = [f"Empresa{i} R$ {(i+1)*100}.000,00 2000" for i in range(15)]
    chunk = MagicMock(
        page_content="\n".join(lines),
        metadata={"source": "document.pdf", "page": 0},
    )
    m = _setup(mocker, chunks=[chunk])
    ingest_pdf()
    all_docs = m["pgvector_cls"].from_documents.call_args.args[0]
    summary_docs = [d for d in all_docs if getattr(d, "metadata", {}).get("type") == "summary"]
    combined_text = " ".join(d.page_content for d in summary_docs)
    assert "Empresa14" in combined_text


def test_ingest_summary_not_created_when_no_values(mocker):
    """Se nenhum chunk tem R$, não deve criar doc de resumo."""
    chunk = MagicMock(
        page_content="Texto simples sem valores monetarios",
        metadata={"source": "document.pdf", "page": 0},
    )
    m = _setup(mocker, chunks=[chunk])
    ingest_pdf()
    all_docs = m["pgvector_cls"].from_documents.call_args.args[0]
    summary_docs = [d for d in all_docs if getattr(d, "metadata", {}).get("type") == "summary"]
    assert len(summary_docs) == 0


def test_ingest_extract_company_data_handles_glued_text():
    """Parser deve tratar nomes colados ao R$ (ex: ParticipaçõesR$)."""
    from ingest import _extract_company_data
    text = "Aurora E-commerce ParticipaçõesR$ 15.935.845,08 1953"
    entries = _extract_company_data(text)
    assert len(entries) >= 1
    assert any(abs(v - 15935845.08) < 0.01 for _, v, _ in entries)