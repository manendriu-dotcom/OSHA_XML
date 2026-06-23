"""
embed_chunks.py
---------------
Encodes every chunk from data/chunks/manifest.jsonl into dense vectors
using a sentence-transformers model, then saves a FAISS index and a
metadata file for use by retrieve.py.

No logic changes from the PDF version — the manifest format is identical.
XML chunking produces cleaner, more uniform chunk lengths, so you may find
you can reduce batch_size if memory is tight, or increase it for speed.

Usage:
    python src/embed_chunks.py
    python src/embed_chunks.py --config config/embedding.yaml \
                                --manifest data/chunks/manifest.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

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
log = logging.getLogger("embed_chunks")


def choose_embedding_model(model_name: str | None = None) -> str:
    """Return the actual embedding model name for the selected option."""
    model_options = {
        "all-minilm-l6-v2": "all-MiniLM-L6-v2",
        "qwen3-embedding": "Qwen/Qwen3-Embedding-0.6B",
        "all-mpnet-base-v2": "all-mpnet-base-v2",
        "granite-embedding-small-english-r2": "ibm-granite/granite-embedding-small-english-r2",
        "mini-gte": "thenlper/gte-small",
    }

    if model_name:
        normalized = model_name.strip().lower()
        if normalized in model_options:
            return model_options[normalized]
        if normalized in model_options.values():
            return normalized
        log.error(
            "Unsupported model '%s'. Valid choices: %s",
            model_name,
            ", ".join(model_options.keys()),
        )
        sys.exit(1)

    print("Select embedding model:")
    for idx, name in enumerate(model_options.keys(), start=1):
        print(f"  {idx}. {name}")

    while True:
        choice = input(
            f"Enter 1-{len(model_options)} or one of {', '.join(model_options.keys())}: "
        ).strip()
        if choice.isdigit() and 1 <= int(choice) <= len(model_options):
            selected_key = list(model_options.keys())[int(choice) - 1]
            return model_options[selected_key]
        if choice.lower() in model_options:
            return model_options[choice.lower()]
        print(
            "Please enter a valid option (for example: 1, all-minilm-l6-v2, or qwen3-embedding)."
        )


def load_chunks(manifest_path: Path) -> tuple[list[str], list[dict]]:
    """Return (texts, metadata_list) from the JSONL manifest."""
    texts: list[str] = []
    metas: list[dict] = []

    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            chunk_file = Path(meta["file"])
            if not chunk_file.exists():
                log.warning("Chunk file missing, skipping: %s", chunk_file)
                continue
            text = chunk_file.read_text(encoding="utf-8", errors="replace")
            texts.append(text)
            metas.append(meta)

    log.info("Loaded %d chunks from manifest", len(texts))
    return texts, metas


def embed(
    texts: list[str],
    model_name: str,
    batch_size: int,
    normalise: bool,
    device: str,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    log.info("Loading model: %s  (device=%s)", model_name, device or "auto")
    kwargs = {}
    if device:
        kwargs["device"] = device
    model = SentenceTransformer(model_name, **kwargs)

    log.info("Encoding %d chunks in batches of %d …", len(texts), batch_size)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalise,
        convert_to_numpy=True,
    )
    log.info("Embedding shape: %s  dtype: %s", vectors.shape, vectors.dtype)
    return vectors.astype(np.float32)


def build_faiss_index(vectors: np.ndarray, index_type: str, metric: str, nlist: int):
    """Build and return a FAISS index."""
    try:
        import faiss  # type: ignore
    except ImportError:
        log.error("faiss-cpu not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    dim = vectors.shape[1]
    log.info("Building FAISS index  type=%s  metric=%s  dim=%d", index_type, metric, dim)

    if metric == "cosine":
        base_index = faiss.IndexFlatIP(dim)
    else:
        base_index = faiss.IndexFlatL2(dim)

    if index_type == "flat":
        index = base_index
    elif index_type == "ivf":
        quantiser = faiss.IndexFlatIP(dim) if metric == "cosine" else faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantiser, dim, nlist)
        log.info("Training IVF index on %d vectors …", vectors.shape[0])
        index.train(vectors)
    else:
        log.warning("Unknown index_type '%s', defaulting to flat.", index_type)
        index = base_index

    index.add(vectors)
    log.info("FAISS index built — %d vectors", index.ntotal)
    return index


def save_index(index, index_path: Path, vectors: np.ndarray, metadata: list[dict], meta_path: Path):
    """Persist the FAISS index and metadata."""
    import faiss  # type: ignore

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    log.info("FAISS index saved → %s", index_path)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as fh:
        for meta in metadata:
            fh.write(json.dumps(meta) + "\n")
    log.info("Metadata saved → %s  (%d entries)", meta_path, len(metadata))

    np_path = index_path.with_suffix(".npy")
    np.save(str(np_path), vectors)
    log.info("Raw vectors saved → %s", np_path)


def main():
    base = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Embed OSHA chunks and build FAISS index.")
    parser.add_argument("--config", default=str(base / "config" / "embedding.yaml"))
    parser.add_argument(
        "--manifest",
        default=str(base / "data" / "chunks" / "manifest.jsonl"),
    )
    parser.add_argument(
        "--model",
        choices=[
            "all-minilm-l6-v2",
            "qwen3-embedding",
            "all-mpnet-base-v2",
            "granite-embedding-small-english-r2",
            "mini-gte",
        ],
        help=(
            "Embedding model choice. Options: "
            + ", ".join(
                [
                    "all-minilm-l6-v2",
                    "qwen3-embedding",
                    "all-mpnet-base-v2",
                    "granite-embedding-small-english-r2",
                    "mini-gte",
                ]
            )
        ),
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    m_cfg = cfg["model"]
    f_cfg = cfg["faiss"]
    o_cfg = cfg["output"]

    model_name = choose_embedding_model(args.model)
    log.info("Using embedding model: %s", model_name)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s — run chunk_osha.py first.", manifest_path)
        sys.exit(1)

    texts, metadata = load_chunks(manifest_path)
    if not texts:
        log.error("No chunks found in manifest.")
        sys.exit(1)

    vectors = embed(
        texts,
        model_name=model_name,
        batch_size=m_cfg["batch_size"],
        normalise=m_cfg["normalize"],
        device=m_cfg.get("device", "") or "",
    )

    index = build_faiss_index(
        vectors,
        index_type=f_cfg["index_type"],
        metric=f_cfg["metric"],
        nlist=f_cfg.get("nlist", 100),
    )

    save_index(
        index,
        index_path=base / o_cfg["index_file"],
        vectors=vectors,
        metadata=metadata,
        meta_path=base / o_cfg["metadata_file"],
    )


if __name__ == "__main__":
    main()