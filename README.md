# Autonomous Research RAG Agent

A **CLI research agent** over a local PDF corpus. The agent learns **when to retrieve, what to retrieve, whether evidence is enough, and when to stop**.

Control flow lives in **LangGraph** (conditional edges). Retrieval lives in **explicit tools**. Gemini does not freely pick tools for the whole run.

```text
START → planner → retriever → evidence_analyzer
                      ↑              │
                      └── query_rewriter (gaps, under cap)
                                     │ enough or cap
                                     ↓
                              synthesizer → critic
                                     ↑         │
                              researcher ──────┘ fail + extra retrieve unused
                                               │ pass or cap
                                               ↓
                                          finalize → END
```

## What is in agent

| Piece | Choice |
|---|---|
| LLM | `gemini-3.6-flash` via `langchain-google-genai` (override with `GEMINI_MODEL`) |
| Embeddings | `gemini-embedding-001`, output dim configurable (default **768**, legal range 128–3072) |
| Vectors | Qdrant in Docker (`6333`, dashboard `/dashboard`) |
| Graph | LangGraph `StateGraph` + conditional edges |
| PDF | PyMuPDF |
| Config | `python-dotenv` + pydantic-settings |

## Layout

```text
autonomous-rag-agent/
├── data/pdfs/              # drop PDFs here
├── data/processed/         # ingest manifests
├── logs/                   # JSONL traces (gitignored)
├── app/
│   ├── config.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── loaders.py
│   ├── tools.py
│   ├── observability.py
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── edges.py
│   │   └── workflow.py
│   └── main.py
├── scripts/ingest.py
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

Pipeline: extract (page numbers) → light clean → ~500–800 token chunks with overlap → embed → upsert. Payloads include `text`, `source`, `page`, `title`, `section`, `chunk_id`, `document_id`. Upserts are idempotent on `chunk_id`. Re-ingest skips PDFs that already have a manifest under `data/processed/`.


## Tools vs edges

[`app/tools.py`](app/tools.py) is Qdrant only:

- `retrieve_chunks` / `search_research_papers` / `search_again`
- `get_document` / `get_document_metadata`

**Retriever** and **researcher** nodes call these. Planner, analyzer, rewriter, critic, and synthesizer only call Gemini. Edges encode autonomy.

## Logs and traces

JSON lines: `logs/agent.jsonl` (same `trace_id` for a run). Components:

| `component` | Use when debugging |
|---|---|
| `ingestion` | PDF empty, chunk counts, upsert |
| `retrieval` | hits, scores, empty search, Qdrant down |
| `generation` | planner queries, gaps, critic, 429/503 |
| `graph` | rewrite vs synthesize, researcher vs finalize |

## License

Use and modify for research and learning. Add a license file if you redistribute.