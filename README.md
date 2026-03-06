# Desafio MBA Engenharia de Software com IA - Full Cycle

Projeto de ingestao de PDF e busca semantica com LangChain, PostgreSQL e pgVector.

## Requisitos

- Python 3.11+
- Docker e Docker Compose
- Uma chave de API da OpenAI ou do Google Gemini

## Configuracao

1. Crie e ative um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instale as dependencias:

```bash
pip install -r requirements.txt
```

3. Crie o arquivo `.env` a partir do exemplo:

```bash
cp .env.example .env
```

4. Ajuste o `.env`:

- `LLM_PROVIDER=openai` para usar OpenAI, ou `LLM_PROVIDER=gemini` para usar Google.
- Preencha `OPENAI_API_KEY` ou `GOOGLE_API_KEY`.
- Se quiser, ajuste `DATABASE_URL`, `PG_VECTOR_COLLECTION_NAME` e `PDF_PATH`.

## Execucao

1. Suba o banco com pgVector:

```bash
docker compose up -d
```

2. Execute a ingestao do PDF:

```bash
python src/ingest.py
```

3. Rode o chat no terminal:

```bash
python src/chat.py
```

Digite perguntas sobre o conteudo do PDF. Para sair, use `sair`, `exit`, `quit` ou `Ctrl+C`.

## Estrutura

```text
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py
│   ├── search.py
│   ├── chat.py
│   └── rag.py
├── document.pdf
└── README.md
```
