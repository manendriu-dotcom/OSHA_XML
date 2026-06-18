"""
analyze_chunks.py
-----------------
Analyze chunking statistics: count chunks by split method and compute percentages.

Usage:
    python src/analyze_chunks.py
    python src/analyze_chunks.py --manifest data/chunks/manifest.jsonl
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

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
log = logging.getLogger("analyze_chunks")


def analyze_chunks(manifest_path: Path) -> None:
    """
    Read manifest and analyze chunking statistics.
    
    Calculating method:
    -------------------
    The manifest is a JSONL file where each line is a JSON object with chunk metadata.
    Each chunk has a 'split_method' field and 'section_id' field.
    
    1. Sections that fit token limit: unique sections with split_method == 'section'
    2. Sections that didn't fit: total unique sections minus sections that fit
    
    3. Chunks breakdown:
       - 'section': The section fit within max_tokens, no splitting occurred
       - 'paragraph': Section exceeded max_tokens, split at paragraph boundaries
       - 'word_window': A paragraph exceeded max_tokens, split with word window
    """
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    chunks = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("Line %d: %s", line_num, e)

    if not chunks:
        log.warning("No chunks found in manifest")
        return

    # Get unique sections and sections that fit
    all_sections = set(c.get("section_id") for c in chunks)
    sections_that_fit = set(
        c.get("section_id") for c in chunks if c.get("split_method") == "section"
    )
    sections_that_didnt_fit = all_sections - sections_that_fit

    total_sections = len(all_sections)
    fit_count = len(sections_that_fit)
    didnt_fit_count = len(sections_that_didnt_fit)

    # Categorize chunks by split method
    split_methods = Counter(c.get("split_method", "unknown") for c in chunks)
    
    # Compute statistics by split method (explicit title/body/overlap)
    chunk_stats = {}
    for method in split_methods:
        method_chunks = [c for c in chunks if c.get("split_method") == method]
        counts = len(method_chunks)
        pct = 100 * counts / len(chunks) if chunks else 0

        title_tokens = [c.get("title_token_estimate", 0) for c in method_chunks]
        body_tokens = [c.get("body_token_estimate", 0) for c in method_chunks]
        overlap_tokens = [c.get("overlap_token_estimate", 0) for c in method_chunks]

        avg_title = sum(title_tokens) / counts if counts else 0
        avg_body = sum(body_tokens) / counts if counts else 0
        avg_overlap = sum(overlap_tokens) / counts if counts else 0
        avg_total = avg_title + avg_body + avg_overlap

        chunk_stats[method] = {
            "count": counts,
            "percentage": pct,
            "avg_title": avg_title,
            "avg_body": avg_body,
            "avg_overlap": avg_overlap,
            "avg_tokens": avg_total,
            "total_tokens": sum(title_tokens) + sum(body_tokens),
        }

    # Display results
    print("\n" + "=" * 70)
    print("SECTIONS STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal sections: {total_sections}")
    print(f"{'Category':<30} {'Count':<8} {'Percentage':<12}")
    print("-" * 70)
    print(f"{'Sections that fit token limit':<30} {fit_count:<8} {100*fit_count/total_sections:>10.1f}%")
    print(f"{'Sections that exceeded limit':<30} {didnt_fit_count:<8} {100*didnt_fit_count/total_sections:>10.1f}%")
    
    print("\n" + "=" * 70)
    print("CHUNKS STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"{ 'Method':<15} {'Count':<8} {'Pct':<8} {'AvgTitle':<10} {'AvgBody':<10} {'AvgOverlap':<12} {'AvgTotal':<10}")
    print("-" * 90)

    for method in ["section", "paragraph", "word_window"]:
        if method in chunk_stats:
            s = chunk_stats[method]
            print(
                f"{method:<15} {s['count']:<8} {s['percentage']:>6.1f}% {s['avg_title']:>9.1f} {s['avg_body']:>9.1f} {s['avg_overlap']:>11.1f} {s['avg_tokens']:>9.1f}"
            )


def main():
    base = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Analyze chunking statistics.")
    parser.add_argument(
        "--manifest",
        default=str(base / "data" / "chunks" / "manifest.jsonl"),
        help="Path to the chunk manifest JSONL file",
    )
    args = parser.parse_args()

    analyze_chunks(Path(args.manifest))


if __name__ == "__main__":
    main()
