"""
Agent definitions for the Multi-Agent Research Analyst.

Three roles, deliberately separated instead of collapsed into one prompt:

1. Retrieval agent: pulls relevant chunks from the corpus for a query.
2. Synthesis agent: writes an answer grounded ONLY in retrieved chunks.
3. Critique agent: checks the synthesis for unsupported claims and either
   approves it or sends it back with specific feedback for revision.

The retrieval/synthesis split from a single-agent design was motivated by a
concrete failure mode: a single agent doing retrieval and synthesis in one
pass tends to confidently write conclusions from weak or tangential sources,
because it never has to defend its synthesis to anything. Separating out a
critique agent forces an explicit check before an answer is finalized.
"""

import os
import json
from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def synthesize(query: str, chunks: list) -> str:
    context = "\n\n---\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
    prompt = f"""You are a research analyst. Answer the question using ONLY the
context provided below. Every claim must be traceable to a specific source.
Cite sources inline using the format [source_filename].

If the context does not contain enough information to answer fully, say so
explicitly rather than filling gaps with outside knowledge.

Context:
{context}

Question: {query}

Answer:"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def critique(query: str, chunks: list, draft_answer: str) -> dict:
    context = "\n\n---\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
    prompt = f"""You are a critical reviewer checking a draft research answer
for faithfulness to its sources. You are skeptical by default: your job is to
find problems, not to rubber-stamp the draft.

Context the draft was supposed to be grounded in:
{context}

Original question: {query}

Draft answer to review:
{draft_answer}

Check for:
1. Any claim in the draft NOT supported by the context (fabrication or overreach)
2. Any citation that misattributes a claim to the wrong source
3. Whether the draft acknowledges gaps where the context is insufficient

Respond ONLY with valid JSON in this exact format, no other text:
{{"approved": true or false, "issues": ["issue 1", "issue 2"], "feedback": "specific instructions for revision, or empty string if approved"}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown code fences if the model wraps the JSON in them
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fail safe: if critique parsing breaks, don't silently approve
        return {"approved": False, "issues": ["critique_parse_error"], "feedback": raw}
