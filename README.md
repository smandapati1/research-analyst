# Multi-Agent Research Analyst

A LangGraph-orchestrated research system that decomposes a question, retrieves evidence with a hybrid lexical retriever, drafts a source-grounded answer, challenges it with independent review stages, and only finalizes it after a quality gate.

The project is intentionally built around **separation of responsibilities** rather than a single prompt. Planning, retrieval, writing, critique, fact-checking, judging, and final editing are distinct stages, which makes failures easier to inspect and gives the graph explicit places to revise, reject, or stop an answer.

## Why use multiple specialized stages?

A single retrieval-and-writing prompt can produce an answer that sounds confident even when the supporting evidence is weak. This pipeline forces the draft through several independent checks before it is accepted:

- a **planner** turns the original question into focused retrieval subquestions;
- a **writer** is restricted to the retrieved corpus and must cite factual claims;
- a **skeptical critic** looks for overreach, unsupported claims, incorrect citations, contradictions, and missing uncertainty;
- a **fact-checker** independently verifies the draft against the same evidence;
- a deterministic **citation checker** detects invalid source names and measures sentence-level citation coverage;
- a **judge** decides whether the answer should be accepted, revised, or rejected;
- a **final editor** may improve clarity only after the answer passes the quality gate, and is explicitly instructed not to introduce new factual claims or alter citations.

This design makes the system behave more like a small research workflow than a single chatbot response.

## Architecture

```text
User question
     |
     v
Planner
     |
     v
Multi-query retrieval
     |
     v
Hybrid Retriever
  - BM25
  - word TF-IDF
  - character n-gram TF-IDF
  - Reciprocal Rank Fusion
     |
     v
Evidence consolidation + deduplication
     |
     v
Writer / Synthesis
     |
     v
Skeptical Critique
     |
     v
Deterministic Citation Check
     |
     v
Independent Fact Check
     |
     v
Judge
   /   |    \
accept revise reject
  |      |      |
  v      +------> Writer (bounded loop)
Final Editor     |
  |              |
  +--------------+
     |
     v
Final answer + metrics + JSON trace
```

The revision loop is bounded by `MAX_REVISIONS = 2`. If the judge continues requesting revision after the limit is reached, the graph finalizes the run as `unresolved` instead of silently treating the answer as approved. A judge decision of `reject` produces a `rejected` run without calling the final editor.

## Components

### Query planning — `src/agents.py::plan_query`

The planning agent decomposes the user question into 2–5 focused retrieval subquestions. The graph also preserves the original user question as a retrieval angle and caps the final list at six queries.

This improves evidence coverage for questions that span several concepts instead of relying on one broad search string.

### Hybrid retrieval — `src/retrieval.py`

The retriever is fully local and combines three lexical signals:

1. **BM25** — useful for exact terminology, regulations, acronyms, and rare entities.
2. **Word-level TF-IDF** with unigrams and bigrams — captures topical and phrase-level similarity.
3. **Character n-gram TF-IDF** — adds fuzzy lexical matching for morphology and small wording differences.

The three rankings are combined with **Reciprocal Rank Fusion (RRF)**. Every returned chunk contains both a fused score and a score breakdown:

```python
chunk.score
chunk.score_breakdown["bm25"]
chunk.score_breakdown["word_tfidf"]
chunk.score_breakdown["char_tfidf"]
```

The public interface remains:

```python
retriever.retrieve(query, k)
```

so the retrieval implementation can later be replaced or extended with embeddings or a vector database without changing the graph contract.

### Multi-query evidence collection — `src/research.py`

Each planned subquestion is retrieved independently. Results are deduplicated by chunk ID, the strongest version of each chunk is retained, and the final evidence set is ranked before being passed to the writer.

The current graph retrieves up to three chunks per planned query and consolidates them into at most eight unique evidence chunks.

### Writer — `src/agents.py::synthesize`

The writer is instructed to:

- use only the retrieved evidence;
- attach inline citations in the form `[source_filename]` to factual claims;
- avoid citing sources that do not support the claim;
- explicitly acknowledge evidence gaps instead of filling them with outside knowledge;
- incorporate concrete critique and fact-check feedback on revision rounds.

### Skeptical critic — `src/agents.py::critique`

The critic reviews the draft against the retrieved evidence and returns structured JSON containing:

```json
{
  "approved": false,
  "issues": ["specific issue"],
  "feedback": "specific revision instructions"
}
```

It checks for unsupported claims, overreach, wrong citations, missed contradictions, and failure to acknowledge insufficient evidence.

### Deterministic citation analysis — `src/evidence.py`

Citation checking does not rely entirely on another LLM. The deterministic checker:

- extracts markdown-source citations from the answer;
- verifies that cited filenames exist in the retrieved evidence set;
- reports invalid citations;
- counts cited sources;
- estimates sentence-level citation coverage for factual sentences.

An invalid citation prevents an `accept` decision from being routed directly to final editing.

### Independent fact checker — `src/agents.py::fact_check`

The fact-checking agent receives the original question, evidence, draft, and deterministic citation analysis. It independently returns whether the draft passed, which claims are unsupported, which citations are problematic, and what should be fixed.

### Judge — `src/agents.py::judge`

The judge receives the critic result, fact-check result, and citation analysis and returns one of three decisions:

```text
accept
revise
reject
```

- **accept** → final editor
- **revise** → bounded revision loop back to the writer
- **reject** → terminate without polishing the answer

### Final editor — `src/agents.py::final_edit`

The final editor runs only after approval. Its prompt explicitly prohibits adding new factual claims or adding, deleting, moving, or changing citations. Its purpose is presentation, not another research step.

### LangGraph orchestration — `src/graph.py`

`StateGraph` carries the complete run state, including:

- original query;
- planned subquestions;
- retrieved chunks;
- current draft;
- revision count;
- critique history;
- fact-check history;
- judge history;
- citation analysis;
- final answer;
- final status (`approved`, `rejected`, or `unresolved`).

Because these artifacts remain in graph state, the system can expose not just the final answer but also how that answer was produced and reviewed.

## Evaluation and tracing

`src/eval.py` provides a lightweight self-built evaluation layer. Each completed run records:

- **faithfulness proxy** — fraction of factual output sentences containing a valid retrieved-source citation;
- citation count;
- invalid citation count;
- source diversity;
- number of retrieved chunks;
- revision count;
- total elapsed time;
- query plan;
- retrieval scores and per-retriever score breakdowns;
- critique history;
- fact-check history;
- judge history;
- deterministic citation analysis;
- final status and answer.

A full JSON trace is written to `traces/` for inspection after each CLI run.

The faithfulness metric is deliberately a **citation-coverage proxy**, not a claim that every cited sentence is objectively true. The separate critic, fact checker, and deterministic citation validation provide additional quality gates around that metric.

## Project structure

```text
research-analyst/
├── corpus/
│   ├── doc1_gresb.md
│   ├── doc2_ab2446.md
│   ├── doc3_scope3.md
│   ├── doc4_cpace.md
│   └── doc5_underwriting.md
├── src/
│   ├── agents.py       # planner, writer, critic, fact checker, judge, editor
│   ├── evidence.py     # deterministic citation analysis
│   ├── eval.py         # metrics and JSON tracing
│   ├── graph.py        # LangGraph orchestration and revision routing
│   ├── research.py     # multi-query retrieval + deduplication
│   └── retrieval.py    # BM25 + TF-IDF + char n-grams + RRF
├── tests/
│   └── test_graph.py
├── traces/
│   └── .gitkeep
├── main.py
└── requirements.txt
```

## Running it

Install dependencies:

```bash
pip install -r requirements.txt
```

Set an Anthropic API key.

macOS/Linux:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="your_key_here"
```

Run a query:

```bash
python main.py "What regulatory catalysts are pushing embodied carbon data into commercial real estate underwriting?"
```

The default model can be overridden with `RESEARCH_ANALYST_MODEL`.

CLI output includes:

- run status;
- revision count;
- elapsed time;
- faithfulness score;
- source diversity;
- planned retrieval questions;
- retrieved evidence and fused scores;
- final answer;
- path to the JSON trace.

## Tests

Run:

```bash
python -m pytest
```

The current suite contains **6 tests** covering:

- hybrid retrieval returns relevant chunks and exposes BM25 / word-TFIDF / character-TFIDF scores;
- invalid citations are detected deterministically;
- a failed first draft is revised and subsequently accepted;
- repeated revision requests terminate as `unresolved` after the configured limit;
- a judge rejection skips the final editor;
- the citation-coverage faithfulness metric penalizes uncited factual claims.

The current implementation passes all six tests.

## Honest scope note

This project is designed as a compact, inspectable research-agent system rather than a production search platform.

It currently uses a **small local markdown corpus and lexical hybrid retrieval**, not a hosted embedding service, vector database, web search engine, or continuously updated external knowledge source. Query planning improves retrieval breadth, but the system does not yet perform open-web research or iterative tool-driven multi-hop search.

The evaluation layer is also intentionally self-built and lightweight rather than a replacement for a full evaluation platform such as LangSmith or a dedicated RAG evaluation suite.

Those choices keep the project runnable with a single Anthropic API key while still demonstrating the parts that matter architecturally: query decomposition, hybrid retrieval, evidence consolidation, explicit agent roles, deterministic verification, bounded revision routing, rejection handling, evaluation, and traceability.

## Possible next steps

Natural extensions include:

- adding embedding-based semantic retrieval and reranking alongside the existing lexical retriever;
- supporting web or API research tools;
- adding iterative retrieval when reviewers identify a specific evidence gap;
- storing historical runs and evaluation metrics for comparison;
- building a frontend that visualizes the plan, retrieved evidence, reviewer decisions, and revision history.
