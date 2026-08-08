"""Hybrid retrieval for the Multi-Agent Research Analyst.

The retriever combines three complementary lexical signals:
- BM25 for exact terms / rare entities / acronyms
- word-level TF-IDF for phrase and topical similarity
- character n-gram TF-IDF for fuzzy matching and morphology

Rankings are fused with Reciprocal Rank Fusion (RRF).  The public interface
remains ``retrieve(query, k)`` so the orchestration layer can swap in a hosted
embedding/vector-store implementation later without changing graph code.
"""

from __future__ import annotations

import glob
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass
class Chunk:
    doc_id: str
    text: str
    source: str
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _chunk_text(text: str, source: str, doc_id_prefix: str, chunk_size: int = 550) -> list[Chunk]:
    """Split markdown into paragraph-aware chunks of roughly ``chunk_size`` chars."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    buf = ""
    idx = 0
    for para in paragraphs:
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) > chunk_size and buf:
            chunks.append(Chunk(doc_id=f"{doc_id_prefix}-{idx}", text=buf.strip(), source=source))
            idx += 1
            buf = para
        else:
            buf = candidate
    if buf:
        chunks.append(Chunk(doc_id=f"{doc_id_prefix}-{idx}", text=buf.strip(), source=source))
    return chunks


class HybridRetriever:
    """Offline hybrid retriever using BM25 + TF-IDF + character n-grams."""

    def __init__(self, corpus_dir: str, *, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            source = os.path.basename(path)
            prefix = os.path.splitext(source)[0]
            self.chunks.extend(_chunk_text(text, source=source, doc_id_prefix=prefix))

        if not self.chunks:
            raise ValueError(f"No .md documents found in {corpus_dir}")

        texts = [c.text for c in self.chunks]
        self.word_vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.word_matrix = self.word_vectorizer.fit_transform(texts)

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
        )
        self.char_matrix = self.char_vectorizer.fit_transform(texts)

        self._doc_tokens = [_tokenize(t) for t in texts]
        self._doc_term_freqs = [Counter(tokens) for tokens in self._doc_tokens]
        self._doc_lengths = np.array([len(tokens) for tokens in self._doc_tokens], dtype=float)
        self._avg_doc_len = float(self._doc_lengths.mean()) if len(self._doc_lengths) else 1.0
        self._doc_freq: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            self._doc_freq.update(set(tokens))

    def _bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        q_tokens = _tokenize(query)
        scores = np.zeros(len(self.chunks), dtype=float)
        n_docs = len(self.chunks)
        for term in q_tokens:
            df = self._doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for i, tf_map in enumerate(self._doc_term_freqs):
                tf = tf_map.get(term, 0)
                if not tf:
                    continue
                denom = tf + k1 * (1 - b + b * self._doc_lengths[i] / max(self._avg_doc_len, 1.0))
                scores[i] += idf * (tf * (k1 + 1)) / denom
        return scores

    @staticmethod
    def _ranking(scores: np.ndarray) -> list[int]:
        return [int(i) for i in np.argsort(scores)[::-1] if scores[int(i)] > 0]

    def retrieve(self, query: str, k: int = 4) -> list[Chunk]:
        if not query.strip() or k <= 0:
            return []

        bm25 = self._bm25_scores(query)
        word = cosine_similarity(self.word_vectorizer.transform([query]), self.word_matrix).ravel()
        char = cosine_similarity(self.char_vectorizer.transform([query]), self.char_matrix).ravel()

        rankings = [self._ranking(bm25), self._ranking(word), self._ranking(char)]
        fused = np.zeros(len(self.chunks), dtype=float)
        for ranking in rankings:
            for rank, idx in enumerate(ranking, start=1):
                fused[idx] += 1.0 / (self.rrf_k + rank)

        top_idx = [int(i) for i in np.argsort(fused)[::-1] if fused[int(i)] > 0][:k]
        results: list[Chunk] = []
        for idx in top_idx:
            base = self.chunks[idx]
            results.append(
                Chunk(
                    doc_id=base.doc_id,
                    text=base.text,
                    source=base.source,
                    score=round(float(fused[idx]), 6),
                    score_breakdown={
                        "bm25": round(float(bm25[idx]), 6),
                        "word_tfidf": round(float(word[idx]), 6),
                        "char_tfidf": round(float(char[idx]), 6),
                    },
                )
            )
        return results


# Backward compatibility with the original repository API.
TfidfRetriever = HybridRetriever
