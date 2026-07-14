# Multi-Agent Research Analyst

A LangGraph-orchestrated research pipeline with three specialized agents
(retrieval, synthesis, critique) instead of a single monolithic prompt, plus
a self-built evaluation and tracing layer.

## Why split into three agents instead of one prompt?

A single agent doing retrieval and synthesis in one pass tends to confidently
write conclusions from weak or tangential sources, because it never has to
defend its answer to anything. Separating out a dedicated **critique agent**
forces an explicit check before an answer is finalized: it re-reads the draft
against the original source chunks, looking specifically for unsupported
claims and citation errors, and can send the draft back for revision with
concrete feedback. This is closer to how a human research team catches its
own mistakes than a single writer editing their own work.

## Architecture

```
retrieve -> synthesize -> critique --(approved)--> finalize
                ^              |
                |     (rejected, revisions < 2)
                +--------------+
```

- **Retrieval** (`src/retrieval.py`): TF-IDF vector search over a local
  markdown corpus. Chosen over an embedding API for this build to keep the
  project runnable without extra API keys or vector DB infrastructure — the
  retriever interface (`.retrieve(query, k)`) is designed so it can be
  swapped for a proper embedding-based retriever (OpenAI, Voyage,
  sentence-transformers + a vector store) without touching the graph.

- **Synthesis agent** (`src/agents.py::synthesize`): writes an answer
  grounded only in retrieved chunks, with inline citations. On a revision
  pass, it receives the critique agent's specific feedback appended to the
  query.

- **Critique agent** (`src/agents.py::critique`): reviews the draft against
  the source chunks for fabricated claims, wrong citations, and
  unacknowledged gaps in the source material. Returns structured JSON
  (`approved`, `issues`, `feedback`) so the graph can route on it
  programmatically.

- **Orchestration** (`src/graph.py`): a `LangGraph` `StateGraph` wiring the
  above into a bounded revision loop (max 2 revisions) so a stubborn
  disagreement between synthesis and critique can't loop forever. If the
  loop exhausts revisions without approval, the run is finalized as
  `"unresolved"` rather than silently shipped as if it passed review.

- **Evaluation & tracing** (`src/eval.py`): scores each run for
  faithfulness (fraction of output sentences with a valid source citation)
  and logs a full JSON trace of the run — query, sources retrieved, every
  critique round, and timing — to `traces/`. This is what actually changed
  my prompts during development: it surfaced a specific failure mode where
  the synthesis agent would cite correctly in the first sentence and then
  drop citations on later sentences, which spot-checking outputs never
  caught.

## Honest scope note

This is a rebuilt, scoped-down version of a project I originally built and
demoed. It uses a local TF-IDF retriever and a self-built evaluation layer
instead of a hosted embedding API and Ragas/LangSmith, so it runs end-to-end
with just an Anthropic API key and no other account setup. The architecture
and the design reasoning are the same; the infrastructure choices are
lighter-weight for portability.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python main.py "What regulatory catalysts are pushing embodied carbon data into commercial real estate underwriting?"
```

Output includes the final answer, faithfulness score, number of revision
rounds, sources used, and a path to the full JSON trace of the run.

## Tested components

Both the retrieval layer and the full graph orchestration (retrieve →
synthesize → critique → revise → finalize) were tested independently:
retrieval was verified to surface the correct source chunks for a sample
query, and the graph was tested with mocked agent responses to confirm the
critique-driven revision loop correctly routes a rejected draft back to
synthesis with feedback, then approves and finalizes the revised draft. See
`tests/test_graph.py`.
