# Multi-Agent Research Analyst

A LangGraph-orchestrated research system that plans a question, retrieves evidence with a hybrid lexical retriever, writes a grounded answer, critiques it, fact-checks it, and sends it through an independent judge before final editing.

## Architecture

```text
question
  ↓
planner
  ↓
parallel multi-query retrieval
(BM25 + word TF-IDF + char n-grams + RRF)
  ↓
evidence consolidation
  ↓
writer
  ↓
skeptical critic
  ↓
fact checker + deterministic citation checks
  ↓
judge
  ├─ accept → final editor → approved
  ├─ revise → writer (bounded loop)
  └─ reject → rejected
```

The revision loop is capped at two passes. If the system still cannot clear the quality gates, the result is marked `unresolved` instead of being silently presented as verified.

## What changed from the original version

- **Hybrid retrieval:** BM25, word-level TF-IDF, and character n-gram TF-IDF are independently ranked and fused with Reciprocal Rank Fusion.
- **Query planning:** a planner decomposes broad questions into focused retrieval subquestions.
- **Multi-query research:** evidence is retrieved per subquestion, deduplicated, scored, and consolidated before writing.
- **Deterministic citation verification:** invalid citations and sentence-level citation coverage are measured without relying on an LLM.
- **Independent fact-checker:** a second reviewer checks claims against retrieved evidence.
- **Independent judge:** critique, fact-check, and citation results are evaluated before a draft can be accepted.
- **Final-editor gate:** editing only happens after approval, reducing the risk that fluent rewriting hides verification failures.
- **Richer traces:** plans, per-retriever scores, critique/fact-check/judge histories, citation analysis, and run metrics are logged.

## Why the retriever is still local

This repository intentionally keeps retrieval runnable without a second hosted API or vector database. The `HybridRetriever.retrieve(query, k)` interface is narrow enough to replace later with OpenAI/Voyage embeddings, sentence-transformers, FAISS, Pinecone, or another vector store without rewriting the graph.

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python main.py "What regulatory catalysts are pushing embodied carbon data into commercial real estate underwriting?"
```

PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="your_key_here"
python main.py "What regulatory catalysts are pushing embodied carbon data into commercial real estate underwriting?"
```

## Test

```bash
python -m pytest -q
```

The tests mock LLM calls, so the graph routing tests do not spend API credits or require an Anthropic key.

## Project layout

```text
src/
  agents.py      # planner, writer, critic, fact-checker, judge, editor
  retrieval.py   # BM25 + TF-IDF hybrid retrieval
  research.py    # multi-query retrieval + deduplication
  evidence.py    # deterministic citation analysis
  graph.py       # LangGraph orchestration and revision routing
  eval.py        # metrics and JSON traces
```
