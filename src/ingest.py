import os
import re
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


def _extract_faturamento_values(text: str) -> list[float]:
    """Extrai todos os valores R$ de um chunk como floats."""
    matches = re.findall(r'R\$\s*([\d\.]+,\d{2})', text)
    values = []
    for m in matches:
        try:
            values.append(float(m.replace('.', '').replace(',', '.')))
        except ValueError:
            continue
    return values


def _extract_company_data(text: str) -> list[tuple[str, float, str]]:
    """Extrai (nome, faturamento, ano) de linhas com R$. Trata nomes colados ao R$."""
    results = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.search(r'R\$\s*([\d\.]+,\d{2})', line)
        if not match:
            continue
        value_str = match.group(1).replace('.', '').replace(',', '.')
        try:
            value = float(value_str)
        except ValueError:
            continue
        # Separar nome (tudo antes de R$, tratando casos colados como "ParticipaçõesR$")
        name_part = re.split(r'R\$', line[:match.end()])[0].strip()
        # Extrair ano (4 digitos após o valor)
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', line[match.end():])
        year = year_match.group(1) if year_match else ""
        if name_part:
            results.append((name_part, value, year))
    return results


def _build_summary_documents(chunks: list) -> list:
    """Gera documentos-resumo com estatísticas globais extraídas de todos os chunks."""
    from langchain_core.documents import Document

    all_entries = []
    for chunk in chunks:
        all_entries.extend(_extract_company_data(chunk.page_content))

    if not all_entries:
        return []

    # Deduplicar por nome (chunk overlap gera duplicatas)
    seen = {}
    for name, value, year in all_entries:
        key = name.strip()
        if key not in seen:
            seen[key] = (name, value, year)
    unique_entries = list(seen.values())

    sorted_desc = sorted(unique_entries, key=lambda x: x[1], reverse=True)
    sorted_asc = sorted(unique_entries, key=lambda x: x[1])

    top = sorted_desc[0]
    bottom = sorted_asc[0]
    total = len(unique_entries)

    def fmt(v):
        return f"R$ {v:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

    # Doc 1: max, min, contagem
    stats_text = (
        f"Resumo estatístico do documento:\n"
        f"Total de empresas: {total}\n"
        f"Empresa com maior faturamento: {top[0]} com {fmt(top[1])}"
        f"{' fundada em ' + top[2] if top[2] else ''}\n"
        f"Empresa com menor faturamento: {bottom[0]} com {fmt(bottom[1])}"
        f"{' fundada em ' + bottom[2] if bottom[2] else ''}"
    )
    docs = [Document(page_content=stats_text, metadata={"source": "summary", "type": "summary"})]

    # Doc 2: Top 10
    top_10 = sorted_desc[:10]
    ranking_lines = ["Ranking das 10 empresas com maior faturamento:"]
    for i, (name, value, year) in enumerate(top_10):
        y = f" fundada em {year}" if year else ""
        ranking_lines.append(f"{i+1}. {name} - {fmt(value)}{y}")
    docs.append(Document(
        page_content="\n".join(ranking_lines),
        metadata={"source": "summary", "type": "summary"},
    ))

    # Doc 3: Bottom 10
    bottom_10 = sorted_asc[:10]
    ranking_lines = ["Ranking das 10 empresas com menor faturamento:"]
    for i, (name, value, year) in enumerate(bottom_10):
        y = f" fundada em {year}" if year else ""
        ranking_lines.append(f"{i+1}. {name} - {fmt(value)}{y}")
    docs.append(Document(
        page_content="\n".join(ranking_lines),
        metadata={"source": "summary", "type": "summary"},
    ))

    return docs


def _enrich_metadata(documents: list) -> None:
    """Adiciona faturamento_max/min ao metadata de chunks que contêm valores R$."""
    for doc in documents:
        values = _extract_faturamento_values(doc.page_content)
        if values:
            doc.metadata["faturamento_max"] = max(values)
            doc.metadata["faturamento_min"] = min(values)
            doc.metadata["faturamento_count"] = len(values)


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
        _enrich_metadata(chunks)
        print(f"Chunks gerados: {len(chunks)}")

        summary_docs = _build_summary_documents(chunks)
        all_docs = chunks + summary_docs
        if summary_docs:
            print(f"Documentos-resumo gerados: {len(summary_docs)}")

        embeddings = OpenAIEmbeddings(model=embedding_model)
        PGVector.from_documents(
            all_docs,
            embeddings,
            connection=database_url,
            collection_name=collection_name,
            pre_delete_collection=True,
        )

        print("Ingestao concluida com sucesso.")
    except SystemExit:
        raise
    except Exception as exc:
        _exit_with_error(f"Falha na ingestao do PDF: {exc}")


if __name__ == "__main__":
    ingest_pdf()
