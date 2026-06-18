"""
retrieve.py
-----------
Semantic retrieval from the FAISS index.  Can be used as a library
(import OSHARetriever) or as a standalone CLI for testing.

No logic changes from the PDF version.  Because XML chunking preserves
clean regulatory paragraph boundaries, retrieval quality should be
noticeably higher — especially for specific paragraph-level citations
like § 1910.147(c)(4).

Usage (CLI):
    python src/retrieve.py "What are the LOTO requirements for energy isolation?"
    python src/retrieve.py --top-k 8 "PPE requirements for eye protection"
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

LOG_PATH = Path(__file__).parent.parent / "logs" / "pipeline.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH),
    ],
)
log = logging.getLogger("retrieve")


@dataclass
class RetrievedChunk:
    chunk_id: str
    section_id: str
    section_title: str
    subpart: str
    subpart_title: str
    chunk_index: int
    score: float
    original_score: float
    text: str


class OSHARetriever:
    """
    Load the FAISS index and metadata once, then answer queries.

    Parameters
    ----------
    config_path : str | Path
        Path to config/retrieval.yaml
    embedding_config_path : str | Path
        Path to config/embedding.yaml
    base_dir : str | Path
        Project root directory
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        embedding_config_path: Optional[str | Path] = None,
        base_dir: Optional[str | Path] = None,
    ):
        self._base = Path(base_dir) if base_dir else Path(__file__).parent.parent

        cfg_path = Path(config_path) if config_path else self._base / "config" / "retrieval.yaml"
        emb_cfg_path = (
            Path(embedding_config_path)
            if embedding_config_path
            else self._base / "config" / "embedding.yaml"
        )

        self._ret_cfg = yaml.safe_load(cfg_path.read_text())["retrieval"]
        emb_cfg = yaml.safe_load(emb_cfg_path.read_text())

        self._model_name: str = emb_cfg["model"]["name"]
        self._normalise: bool = emb_cfg["model"]["normalize"]
        self._device: str = emb_cfg["model"].get("device", "") or ""
        self._index_path = self._base / emb_cfg["output"]["index_file"]
        self._meta_path = self._base / emb_cfg["output"]["metadata_file"]

        self._model = None
        self._index = None
        self._metadata: list[dict] = []

    # ── lazy initialisation ──────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._model is None:
            self._load_model()
        if self._index is None:
            self._load_index()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            log.error("sentence-transformers not installed.")
            sys.exit(1)

        log.info("Loading embedding model: %s", self._model_name)
        kwargs = {}
        if self._device:
            kwargs["device"] = self._device
        self._model = SentenceTransformer(self._model_name, **kwargs)

    def _load_index(self):
        try:
            import faiss  # type: ignore
        except ImportError:
            log.error("faiss-cpu not installed.")
            sys.exit(1)

        if not self._index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self._index_path}\n"
                "Run embed_chunks.py first."
            )

        log.info("Loading FAISS index: %s", self._index_path)
        self._index = faiss.read_index(str(self._index_path))
        log.info("Index has %d vectors", self._index.ntotal)

        self._metadata = []
        with self._meta_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._metadata.append(json.loads(line))
        log.info("Loaded %d metadata records", len(self._metadata))

    # ── public API ───────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top_k most relevant OSHA chunks for query.
        """
        self._ensure_loaded()
        assert self._model is not None
        assert self._index is not None

        k = top_k if top_k is not None else self._ret_cfg["top_k"]
        min_s = min_score if min_score is not None else self._ret_cfg["min_score"]
        context_window: int = self._ret_cfg.get("context_window", 0)

        query_vec = self._model.encode(
            [query],
            normalize_embeddings=self._normalise,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self._index.search(query_vec, k)
        log.info(
            "FAISS top-%d raw scores: %s",
            k,
            list(zip(indices[0].tolist(), [round(float(s), 4) for s in scores[0]])),
        )

        results: list[RetrievedChunk] = []
        seen_indices: set[int] = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < min_s:
                continue

            chunk_indices = self._context_indices(idx, context_window)

            for ci in chunk_indices:
                if ci in seen_indices:
                    continue
                seen_indices.add(ci)

                chunk_score = float(score) if ci == idx else 0.0
                if chunk_score < min_s:
                    continue

                meta = self._metadata[ci]
                chunk_file = Path(meta["file"])
                text = (
                    chunk_file.read_text(encoding="utf-8", errors="replace")
                    if chunk_file.exists()
                    else "[chunk file missing]"
                )

                results.append(
                    RetrievedChunk(
                        chunk_id=meta["chunk_id"],
                        section_id=meta["section_id"],
                        section_title=meta["section_title"],
                        subpart=meta.get("subpart", ""),
                        subpart_title=meta.get("subpart_title", ""),
                        chunk_index=meta["chunk_index"],
                        score=chunk_score,
                        original_score=chunk_score,
                        text=text,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return self.rerank(query, results, k)

    def _context_indices(self, idx: int, window: int) -> list[int]:
        lo = max(0, idx - window)
        hi = min(len(self._metadata) - 1, idx + window)
        return list(range(lo, hi + 1))

    # ── optional re-ranking ──────────────────────────────────────────────────

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Re-rank chunks with a cross-encoder and return the top top_n."""
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError:
            log.warning("CrossEncoder not available; skipping re-rank.")
            return chunks[:top_n]

        model_name = self._ret_cfg.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        log.info("Re-ranking with %s …", model_name)
        ce = CrossEncoder(model_name)
        pairs = [(query, c.text) for c in chunks]
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                section_id=c.section_id,
                section_title=c.section_title,
                subpart=c.subpart,
                subpart_title=c.subpart_title,
                chunk_index=c.chunk_index,
                score=float(s),
                original_score=c.original_score,
                text=c.text,
            )
            for s, c in ranked[:top_n]
        ]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_results(query: str, results: list[RetrievedChunk]) -> None:
    print(f"{'='*72}")
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} chunk(s)")
    print(f"{'='*72}")
    for i, r in enumerate(results, 1):
        print(f"[{i}] § {r.section_id}  —  {r.section_title}")
        print(f"     Subpart {r.subpart}: {r.subpart_title}")
        score_line = f"     Chunk {r.chunk_index}  |  Score: {r.score:.4f}"
        if r.score != r.original_score:
            score_line += f"  (raw: {r.original_score:.4f})"
        print(score_line)
        print(f"     {'-'*64}")
        preview = r.text[:400].replace("\n", " ")
        print(f"     {preview}…")
        print()


def main():
    base = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Retrieve relevant OSHA 1910 chunks.")
    parser.add_argument(
        "query", nargs="*", help="Natural-language query (omit for interactive mode)"
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--config", default=str(base / "config" / "retrieval.yaml"))
    parser.add_argument("--emb-config", default=str(base / "config" / "embedding.yaml"))
    args = parser.parse_args()

    retriever = OSHARetriever(
        config_path=args.config,
        embedding_config_path=args.emb_config,
        base_dir=base,
    )

    if args.query:
        query = " ".join(args.query)
        results = retriever.retrieve(query, top_k=args.top_k, min_score=args.min_score)
        _print_results(query, results)
    else:
        print("OSHA Retrieval — interactive mode  (type 'quit' or Ctrl-C to exit)")
        while True:
            try:
                query = input("Query> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not query:
                continue
            if query.lower() in {"quit", "exit", "q"}:
                break
            results = retriever.retrieve(query, top_k=args.top_k, min_score=args.min_score)
            _print_results(query, results)


if __name__ == "__main__":
    main()