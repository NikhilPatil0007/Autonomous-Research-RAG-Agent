# FastAPI, streaming, and caching for this agent

Learning note. The research loop stays in LangGraph. A FastAPI service would only start a run and watch it. The CLI already walks `graph.stream` node by node in `app/main.py` and prints the cited answer at the end.

A question can pass through planner, retriever, a rewrite loop, synthesizer, and critic before anything is printed.

## Streaming

Over Server-Sent Events, a client sees each step as it finishes:

- search queries from the planner
- how many chunks the retriever returned
- whether the evidence analyzer asked for another pass
- the critic result
- the cited answer from finalize

The wait is the same length. The difference is you can see which step is slow or failing instead of a blank terminal until finalize.

The events already exist. `iter_research` in `app/main.py` consumes `graph.stream(..., stream_mode="updates")`. The CLI prints each event. `POST /research` in `app/api.py` forwards the same events as Server-Sent Events.

```text
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Each SSE line is `data: {json}`. The payload names the node and only the fields worth showing while the run is still going: `search_queries`, `n_chunks`, `needs_more_research`, `critic_pass`, `critic_notes`, and `final_answer`. A final `done` event repeats the path and the cited answer. Chunk text stays in Qdrant and in the graph state.

## Async service

FastAPI holds several questions in one process. The graph, Gemini calls, and Qdrant client are synchronous, so each run executes in a worker thread. `MAX_CONCURRENT_RESEARCH` (default 2) is an `asyncio.Semaphore`. A request past that cap gets HTTP 503 instead of another Gemini call.

That cap matters because Gemini already returns 429 and 503 when embed or chat traffic spikes. More parallel requests would make the service less reliable.

What stays the same:

- Qdrant stays local
- citations still come only from chunk payloads
- Gemini does not pick tools
- the iteration cap and the one critic-triggered lookup stay in the graph

FastAPI does not improve precision or recall. It lets more than one client use the same process and shows the loop while it runs.

## Caching

Three caches match this project.

### Query cache

The same sub-query (embedding plus Qdrant hits) is reused across rewrite loops and follow-up turns. That cuts `gemini-embedding-001` calls, which is the call that already exhausts quota during ingest and retrieval.

A useful key is the query text, retrieval strategy, top-k, and collection name. Store the hit list (title, page, chunk id, score, text), not a second copy of the corpus.

### Answer cache

An identical question on the same thread, with the same last-6 memory and the same ingested corpus, returns the stored cited answer. Useful for demos and eval reruns.

Skip the cache when the PDF set changes. Key the entry by the ingest manifests under `data/processed/`, so a new or updated PDF does not return an old citation list.

### Session memory

`data/sessions/{thread}.json` is conversation context, not a retrieval cache. `load_thread_messages` and `save_thread_messages` in `app/graph/workflow.py` keep the last 6 user, tool, and assistant messages. The CLI passes `--thread`. `POST /research` passes `thread_id`. Both load that file before the planner runs and write it back after finalize. A fresh question still starts with empty chunks. Only the message window carries over.

## What this does not change

- Control flow stays in the graph: planner, retriever, evidence analyzer, query rewriter, synthesizer, critic, researcher, finalize.
- Tools stay capabilities called by nodes.
- Retrieved chunk text stays out of the 6-message window. The tool message stays a short list of titles, pages, and chunk ids.
