"""
Retrieval layer for the Multi-Agent Research Analyst.

Uses TF-IDF vectorization + cosine similarity for chunk retrieval over a local
corpus. This is intentionally dependency-light (no external embedding API
calls, no vector DB) so the whole pipeline can run offline except for the LLM
calls made by the synthesis and critique agents. Swap in a proper embedding
model (OpenAI, Voyage, sentence-transformers) by replacing `TfidfRetriever`
with an equivalent class that implements `.retrieve(query, k)`.
"""

import os
import re
import glob
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    doc_id: str
    text: str
    source: str


def _chunk_text(text: str, source: str, doc_id_prefix: str, chunk_size: int = 400) -> list[Chunk]:
    """Split a document into paragraph-aware chunks of roughly `chunk_size` chars."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buf = ""
    idx = 0
    for para in paragraphs:
        if len(buf) + len(para) > chunk_size and buf:
            chunks.append(Chunk(doc_id=f"{doc_id_prefix}-{idx}", text=buf.strip(), source=source))
            idx += 1
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(Chunk(doc_id=f"{doc_id_prefix}-{idx}", text=buf.strip(), source=source))
    return chunks


class TfidfRetriever:
    def __init__(self, corpus_dir: str):
        self.chunks: list[Chunk] = []
        for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            doc_id_prefix = os.path.splitext(os.path.basename(path))[0]
            self.chunks.extend(_chunk_text(text, source=os.path.basename(path), doc_id_prefix=doc_id_prefix))

        if not self.chunks:
            raise ValueError(f"No .md documents found in {corpus_dir}")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, k: int = 4) -> list[Chunk]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        return [self.chunks[i] for i in top_idx if sims[i] > 0]
