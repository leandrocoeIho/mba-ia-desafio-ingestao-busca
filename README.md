# Desafio MBA Engenharia de Software com IA - Full Cycle

## Visão geral

Este projeto implementa um fluxo de ingestão e busca semântica sobre o arquivo `document.pdf` usando LangChain, PostgreSQL e pgVector.

O processo funciona em duas etapas:

1. `src/ingest.py` carrega o PDF, divide o conteúdo em chunks, gera embeddings e persiste os vetores no PostgreSQL.
2. `src/chat.py` inicia um chat em linha de comando que responde perguntas em linguagem natural usando apenas o conteúdo recuperado do PDF.

Se a informação não estiver no PDF, a resposta padrão do sistema é:

`Não tenho informações necessárias para responder sua pergunta.`

## Stack obrigatória

- Python 3.12
- LangChain `0.3.27`
- `langchain-openai` `0.3.30`
- `langchain-postgres` `0.0.15`
- OpenAI Embeddings `text-embedding-3-small`
- OpenAI Chat Model `gpt-4o-mini` com `temperature=0`
- PostgreSQL 17
- pgVector
- psycopg3 via URL `postgresql+psycopg://`
- Docker e Docker Compose
- `python-dotenv`

## Pré-requisitos

- Python 3.12 instalado
- Docker Desktop com Docker Compose habilitado
- Chave válida da OpenAI
- Porta `5432` livre para o PostgreSQL local
- Arquivo `document.pdf` presente na raiz do projeto

## Configuração do ambiente

Os comandos abaixo foram validados em PowerShell no Windows, no diretório raiz do projeto.

### 1. Criar e ativar o ambiente virtual

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Instalar as dependências

```powershell
pip install -r requirements.txt
```

### 3. Criar o arquivo `.env`

```powershell
Copy-Item .env.example .env
```

Preencha o `.env` com os valores abaixo:

```env
OPENAI_API_KEY=sua_chave_openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=documents
PDF_PATH=document.pdf
SIMILARITY_THRESHOLD=0.5
```

## Execução

### 1. Subir o banco com pgVector

```powershell
docker compose up -d
```

O `docker-compose.yml` sobe:

- um PostgreSQL 17 local em `localhost:5432`
- um bootstrap que executa `CREATE EXTENSION IF NOT EXISTS vector;`

### 2. Ingerir o PDF

```powershell
python src/ingest.py
```

Com o `document.pdf` atual do repositório, a ingestão processa `34` páginas e gera `67` chunks antes de persistir os vetores.

Saída esperada:

```text
Paginas carregadas: 34
Chunks gerados: 67
Documentos-resumo gerados: 3
Ingestao concluida com sucesso.
```

### 3. Iniciar o chat

```powershell
python src/chat.py
```

Fluxo da CLI:

- o banner inicial exibido uma vez é `Faça sua pergunta:`
- a pergunta é digitada em `PERGUNTA: `
- a resposta é impressa em `RESPOSTA: `
- para encerrar, use `sair`, `exit`, `quit` ou `Ctrl+C`

## Exemplo validado em execução real

Os exemplos abaixo seguem o formato real da CLI e foram validados contra o `document.pdf` atual do repositório.

### Pergunta dentro do contexto

```text
Faça sua pergunta:
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: A empresa SuperTechIABrazil tem um faturamento de 10 milhões de reais.
```

### Pergunta fora do contexto

```text
Faça sua pergunta:
PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Troubleshooting

### `PDF não encontrado`

Verifique se `PDF_PATH=document.pdf` aponta para um arquivo existente na raiz do projeto.

### `PDF sem texto extraível`

O arquivo precisa conter texto selecionável. PDF escaneado como imagem não será ingerido.

### `DATABASE_URL inválida: use o prefixo postgresql+psycopg://`

Use exatamente o prefixo `postgresql+psycopg://` no `.env`.

### `A coleção está vazia. Execute a ingestão do PDF antes de iniciar o chat.`

Execute primeiro:

```powershell
python src/ingest.py
```

### Erro ao criar a extensão `vector`

Confirme que os containers subiram e consulte o bootstrap:

```powershell
docker compose ps
docker compose logs bootstrap_vector_ext
```

### Erro de autenticação ou conexão com a OpenAI

Valide a `OPENAI_API_KEY` no `.env` e confirme acesso de rede à API da OpenAI.

## Limitações conhecidas

O PDF contém dados de 1001 empresas distribuídos em 67 chunks de 1000 caracteres. Com `k=10`, cada busca recupera no máximo 10 chunks — uma fração do total. Isso significa que:

- Perguntas sobre empresas específicas podem falhar se o chunk correspondente não pontuar entre os 10 mais similares.
- Respostas comparativas por setor (ex: "maior faturamento de Telecom") são corretas apenas dentro dos chunks recuperados, não do dataset completo.
- Perguntas globais (maior/menor faturamento geral, contagem total) são precisas porque usam documentos-resumo pré-computados na ingestão.

Essas limitações decorrem da configuração exigida pelo enunciado (`chunk_size=1000`, `chunk_overlap=150`, `k=10`) e da natureza da busca por similaridade semântica sobre dados tabulares.

## Observações de segurança

- Não commite o arquivo `.env`.
- A credencial `postgres/postgres` do `docker-compose.yml` é apenas para desenvolvimento local.
- Não altere os modelos configurados: `text-embedding-3-small` e `gpt-4o-mini`.
- O sistema foi projetado para responder apenas com base no `document.pdf`, sem uso de conhecimento externo.
