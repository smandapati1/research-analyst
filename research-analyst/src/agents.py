"""Specialized LLM agents for planning, writing, review, judging, and editing."""

from __future__ import annotations

import json
import os
from anthropic import Anthropic

MODEL = os.environ.get("RESEARCH_ANALYST_MODEL", "claude-sonnet-4-6")
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _text(prompt: str, max_tokens: int = 900) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _json(prompt: str, fallback: dict, max_tokens: int = 700) -> dict:
    raw = _text(prompt, max_tokens=max_tokens)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else fallback
    except json.JSONDecodeError:
        return {**fallback, "parse_error": raw}


def plan_query(query: str) -> dict:
    return _json(
        f"""You are the planning agent for a research system.
Break the user's question into 2-5 retrieval-focused subquestions. Keep them
specific, non-overlapping, and answerable from documents. Include the original
question if it already represents an important retrieval angle.

Question: {query}

Return ONLY JSON:
{{"subquestions": ["..."], "reasoning_summary": "one concise sentence"}}""",
        {"subquestions": [query], "reasoning_summary": "Fallback to the original question."},
        max_tokens=450,
    )


def synthesize(query: str, chunks: list, feedback: str = "") -> str:
    context = "\n\n---\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
    revision = f"\nReviewer instructions from the previous round:\n{feedback}\n" if feedback else ""
    return _text(
        f"""You are the writer agent in a research-analysis pipeline.
Answer using ONLY the evidence below. Every factual claim must have an inline
citation in the format [source_filename]. Do not cite a source unless that
source actually supports the claim. Explicitly state evidence gaps instead of
using outside knowledge. Prefer synthesis across multiple independent sources
when the evidence supports it.{revision}

Evidence:
{context}

Question: {query}

Answer:""",
        max_tokens=1000,
    )


def critique(query: str, chunks: list, draft_answer: str) -> dict:
    context = "\n\n---\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
    return _json(
        f"""You are the skeptical reviewer. Review the draft against the evidence.
Check for unsupported claims, overreach, wrong citations, missed contradictions,
and failure to acknowledge insufficient evidence.

Evidence:\n{context}\n\nQuestion: {query}\n\nDraft:\n{draft_answer}

Return ONLY JSON:
{{"approved": true, "issues": [], "feedback": ""}}
or
{{"approved": false, "issues": ["specific issue"], "feedback": "specific revision instructions"}}""",
        {"approved": False, "issues": ["critique_parse_error"], "feedback": "Critique could not be parsed; revise conservatively."},
    )


def fact_check(query: str, chunks: list, draft_answer: str, citation_analysis: dict) -> dict:
    context = "\n\n---\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
    return _json(
        f"""You are an independent fact-checking agent. Verify the draft line by line
against ONLY the supplied evidence. The deterministic citation checker reported:
{json.dumps(citation_analysis)}

Evidence:\n{context}\n\nQuestion: {query}\n\nDraft:\n{draft_answer}

Return ONLY JSON:
{{"passed": true, "unsupported_claims": [], "citation_errors": [], "feedback": ""}}
or the same object with passed=false and precise problems.""",
        {"passed": False, "unsupported_claims": ["fact_check_parse_error"], "citation_errors": [], "feedback": "Fact check could not be parsed."},
    )


def judge(query: str, critique_result: dict, fact_check_result: dict, citation_analysis: dict) -> dict:
    return _json(
        f"""You are the final quality judge. Decide whether the draft can be accepted,
needs revision, or should be rejected because the evidence is insufficient.
Do not override concrete citation or factual failures.

Question: {query}
Critique: {json.dumps(critique_result)}
Fact check: {json.dumps(fact_check_result)}
Citation analysis: {json.dumps(citation_analysis)}

Return ONLY JSON:
{{"decision": "accept|revise|reject", "reason": "concise reason"}}""",
        {"decision": "revise", "reason": "Judge response could not be parsed."},
        max_tokens=350,
    )


def final_edit(query: str, answer: str) -> str:
    return _text(
        f"""You are the final editor. Improve clarity and organization of the approved
answer without adding, deleting, moving, or changing any source citation and
without introducing any new factual claim. Preserve uncertainty language.

Question: {query}\n\nApproved answer:\n{answer}\n\nEdited answer:""",
        max_tokens=1000,
    )
