"""
chunk_osha.py
-------------
Splits the cleaned OSHA 1910 text into semantically meaningful chunks.

Identical pipeline to the PDF version with two key XML-specific improvements:

  1. Section regex is tightened to match the normalised output of extract_text.py
     (§ 1910.XXX with a mandatory space after §).
  2. Subpart regex uses the em-dash form enforced by clean_text.py
     ("Subpart J—Title" not "Subpart J -- Title").
  3. Paragraph-level sub-splitting:  when a section exceeds max_tokens, the
     script first tries to split on paragraph boundaries (lines starting with
     a list marker like (a), (b), (1)) before falling back to word-window
     splitting.  This keeps regulatory paragraphs intact wherever possible.

Usage:
    python src/chunk_osha.py
    python src/chunk_osha.py --config config/chunking.yaml \
                              --input  data/cleaned/osha_1910_cleaned.txt
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

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
log = logging.getLogger("chunk_osha")


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    section_id: str
    section_title: str
    subpart: str
    subpart_title: str
    chunk_index: int
    text: str
    token_estimate: int
    is_split: bool = False
    split_method: str = "section"  # 'section', 'paragraph', or 'word_window'
    title_token_estimate: int = 0
    body_token_estimate: int = 0
    overlap_token_estimate: int = 0


# ── tokenisation ──────────────────────────────────────────────────────────────

# Lazy-loaded tokenizer for accurate token counting
_TOKENIZER = None

def _get_tokenizer():
    """Lazy-load the tokenizer on first use."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
    return _TOKENIZER

def count_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))

def split_tokens(text: str, size: int, overlap: int) -> list[tuple[str, int]]:
    """Split `text` into chunks where each chunk's token count <= size.

    Returns list of tuples: (chunk_text, overlap_tokens_for_this_chunk)
    where `overlap_tokens_for_this_chunk` is the number of tokens at the
    start of this chunk that overlap with the previous chunk.
    """
    words = text.split()
    chunks: list[tuple[str, int]] = []
    start = 0
    n = len(words)
    prev_overlap = 0

    while start < n:
        end = start
        while end < n:
            candidate = " ".join(words[start : end + 1])
            if count_tokens(candidate) > size:
                break
            end += 1

        if end == start:
            end = start + 1

        chunk_text = " ".join(words[start:end])
        chunks.append((chunk_text, prev_overlap))

        if end >= n:
            break

        # determine overlap for next chunk (in tokens)
        overlap_end = end
        while overlap_end > start:
            overlap_candidate = " ".join(words[overlap_end - 1 : end])
            if count_tokens(overlap_candidate) <= overlap:
                overlap_end -= 1
            else:
                break

        # overlap tokens for next chunk is tokens in words[overlap_end:end]
        overlap_tokens = count_tokens(" ".join(words[overlap_end:end])) if overlap_end < end else 0
        prev_overlap = overlap_tokens
        start = overlap_end
    return chunks


def split_text_with_header(
    body: str,
    header: str,
    max_tokens: int,
    overlap: int,
    base_method: str,
) -> list[tuple[str, str, int, int, int]]:
    """Ensure the final chunk text with header does not exceed max_tokens."""
    header_tokens = count_tokens(header)
    body_budget = max_tokens - header_tokens
    if body_budget <= 0:
        raise ValueError(
            "max_tokens must exceed header token count to build a valid chunk"
        )

    if count_tokens(body) <= body_budget:
        final_text = header + body
        if count_tokens(final_text) > max_tokens:
            raise AssertionError("Header+body chunk exceeds max_tokens")
        return [(final_text, base_method, header_tokens, count_tokens(body), 0)]

    results: list[tuple[str, str, int, int, int]] = []
    for chunk_text, overlap_tokens in split_tokens(body, body_budget, overlap):
        final_text = header + chunk_text
        if count_tokens(final_text) > max_tokens:
            raise AssertionError("Header+chunk exceeds max_tokens")
        results.append((final_text, "word_window", header_tokens, count_tokens(chunk_text), overlap_tokens))
    return results


# ── paragraph-aware sub-splitting ────────────────────────────────────────────

# Matches lines that start a new regulatory paragraph:
#   (a)  (1)  (iv)  (A)  — list items
#   "Note:" / "Note to paragraph"
_PARA_START = re.compile(
    r"^(\(\s*[a-zA-Z0-9]{1,4}\s*\)"    # (a) (1) (iv) (A)
    r"|Note[\s:]"                        # Note: / Note to
    r"|\d+\.\s+[A-Z]"                   # 1. Scope
    r")"
)


def split_on_paragraphs(text: str, max_tokens: int, overlap: int) -> list[tuple[str, str]]:
    """
    Try to split *text* at paragraph boundaries first.
    Falls back to word-window splitting when paragraphs are themselves too long.
    Returns list of (chunk_text, split_method) tuples where split_method is 'paragraph' or 'word_window'.
    """
    lines = text.splitlines(keepends=True)
    groups: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if _PARA_START.match(line.lstrip()) and current:
            groups.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        groups.append(current)

    chunks: list[tuple[str, str]] = []
    buffer: list[str] = []
    buf_tokens = 0

    for grp in groups:
        grp_text = "".join(grp)
        grp_tokens = count_tokens(grp_text)

        if grp_tokens > max_tokens:
            # Paragraph itself too large — word-window split
            if buffer:
                chunks.append(("".join(buffer), "paragraph"))
                buffer = []
                buf_tokens = 0
            word_window_chunks = split_tokens(grp_text, max_tokens, overlap)
            chunks.extend([(chunk_text, "word_window") for chunk_text, _ in word_window_chunks])
            continue

        if buf_tokens + grp_tokens > max_tokens and buffer:
            chunks.append(("".join(buffer), "paragraph"))
            # Overlap: keep last paragraph of previous buffer
            buffer = buffer[-1:] if overlap > 0 else []
            buf_tokens = count_tokens("".join(buffer))

        buffer.append(grp_text)
        buf_tokens += grp_tokens

    if buffer:
        chunks.append(("".join(buffer), "paragraph"))

    return chunks if chunks else [(text, "paragraph")]


# ── model selection ───────────────────────────────────────────────────────────

_EMBEDDING_MODELS = {
    "all-minilm-l6-v2": ("all-MiniLM-L6-v2", 512),
    "qwen3-embedding-0.6b": ("Qwen3-Embedding-0.6B", 8000),
}


def choose_embedding_model(model_name: str | None = None) -> tuple[str, int]:
    """Return an embedding model and recommended max token limit."""
    normalized = model_name.strip() if model_name else ""
    if normalized:
        key = normalized.lower()
        if key in _EMBEDDING_MODELS:
            return _EMBEDDING_MODELS[key]
        if normalized in ("all-MiniLM-L6-v2", "Qwen3-Embedding-0.6B"):
            return _EMBEDDING_MODELS[normalized.lower()]
        log.error("Unsupported model '%s'. Valid choices: %s", normalized, ", ".join(_EMBEDDING_MODELS.keys()))
        sys.exit(1)

    choices = ["all-MiniLM-L6-v2", "Qwen3-Embedding-0.6B"]
    print("Select embedding model:")
    for idx, name in enumerate(choices, start=1):
        print(f"  {idx}. {name}")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in {"1", "2"}:
            selected = choices[int(choice) - 1]
            return choose_embedding_model(selected)
        print("Please enter 1 or 2.")


# ── section parsing ───────────────────────────────────────────────────────────

# Matches: § 1910.147 Control of hazardous energy (lockout/tagout).
# clean_text.py guarantees "§ " (with space) and the section number form.
_SUBPART_RE = re.compile(r"^Subpart\s+([A-Z]+)—(.+)", re.MULTILINE)
_SECTION_RE = re.compile(r"^§\s+(1910\.\S+)\s+(.*)", re.MULTILINE)


def parse_sections(text: str) -> list[dict]:
    """
    Return a list of section dicts in document order.
    Each dict: section_id, section_title, subpart, subpart_title, body.
    """
    events: list[tuple[int, str, tuple]] = []

    for m in _SUBPART_RE.finditer(text):
        events.append((m.start(), "subpart", (m.group(1).strip(), m.group(2).strip())))

    for m in _SECTION_RE.finditer(text):
        events.append((m.start(), "section", (m.group(1).strip(), m.group(2).strip())))

    events.sort(key=lambda e: e[0])

    sections: list[dict] = []
    current_subpart = ""
    current_subpart_title = ""

    for i, (pos, kind, data) in enumerate(events):
        if kind == "subpart":
            current_subpart, current_subpart_title = data
            continue

        sec_id, sec_title = data
        next_section_events = [e for e in events[i + 1:] if e[1] == "section"]
        end_pos = next_section_events[0][0] if next_section_events else len(text)

        body = text[pos:end_pos].strip()
        sections.append(
            {
                "section_id": sec_id,
                "section_title": sec_title,
                "subpart": current_subpart,
                "subpart_title": current_subpart_title,
                "body": body,
            }
        )

    return sections


# ── chunking ──────────────────────────────────────────────────────────────────

def make_chunks(
    sections: list[dict],
    max_tokens: int,
    overlap_tokens: int,
    filename_template: str,
) -> Iterator[Chunk]:
    for sec in sections:
        sec_id_safe = sec["section_id"].replace(".", "_")
        section_text = sec["body"]
        header = f"§ {sec['section_id']} {sec['section_title']}\n"
        if section_text.startswith(header):
            section_body = section_text[len(header) :]
        else:
            _, _, section_body = section_text.partition("\n")

        tok = count_tokens(section_text)
        title_tokens = count_tokens(header)

        if tok <= max_tokens:
            body_tokens = tok - title_tokens if tok >= title_tokens else tok
            yield Chunk(
                chunk_id=sec_id_safe,
                section_id=sec["section_id"],
                section_title=sec["section_title"],
                subpart=sec["subpart"],
                subpart_title=sec["subpart_title"],
                chunk_index=0,
                text=section_text,
                token_estimate=tok,
                is_split=False,
                split_method="section",
                title_token_estimate=title_tokens,
                body_token_estimate=body_tokens,
                overlap_token_estimate=0,
            )
        else:
            # XML paragraphs are cleanly delineated — prefer paragraph splits
            body_budget = max_tokens - title_tokens
            sub_chunks = split_on_paragraphs(section_body, body_budget, overlap_tokens)
            for idx, (sub, split_method) in enumerate(sub_chunks):
                for subidx, item in enumerate(
                    split_text_with_header(sub, header, max_tokens, overlap_tokens, split_method)
                ):
                    final_text, final_method, title_tokens, body_tokens, overlap_tokens_val = item
                    chunk_id = filename_template.format(
                        section_id=sec_id_safe,
                        chunk_index=f"{idx}_{subidx}",
                    ).replace(".txt", "")
                    yield Chunk(
                        chunk_id=chunk_id,
                        section_id=sec["section_id"],
                        section_title=sec["section_title"],
                        subpart=sec["subpart"],
                        subpart_title=sec["subpart_title"],
                        chunk_index=idx,
                        text=final_text,
                        token_estimate=count_tokens(final_text),
                        is_split=True,
                        split_method=final_method,
                        title_token_estimate=title_tokens,
                        body_token_estimate=body_tokens,
                        overlap_token_estimate=overlap_tokens_val,
                    )


# ── output ────────────────────────────────────────────────────────────────────

def save_chunks(
    chunks: list[Chunk],
    chunks_dir: Path,
    manifest_path: Path,
) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as mf:
        for chunk in chunks:
            fname = chunk.chunk_id + ".txt"
            out = chunks_dir / fname
            out.write_text(chunk.text, encoding="utf-8")

            meta = asdict(chunk)
            del meta["text"]
            meta["file"] = str(out)
            mf.write(json.dumps(meta) + "\n")

    log.info("Wrote %d chunk files → %s", len(chunks), chunks_dir)
    log.info("Manifest → %s", manifest_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    base = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Chunk cleaned OSHA 1910 text.")
    parser.add_argument("--config", default=str(base / "config" / "chunking.yaml"))
    parser.add_argument(
        "--input",
        default=str(base / "data" / "cleaned" / "osha_1910_cleaned.txt"),
    )
    parser.add_argument(
        "--model",
        choices=["all-MiniLM-L6-v2", "Qwen3-Embedding-0.6B"],
        help="Embedding model choice for token limit and later embedding.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not cfg or "section" not in cfg or "output" not in cfg:
        log.error("Config missing 'section' or 'output' keys: %s", cfg_path)
        sys.exit(1)

    sec_cfg = cfg["section"]
    out_cfg = cfg["output"]

    model_name, model_max_tokens = choose_embedding_model(args.model)
    max_tokens: int = model_max_tokens
    overlap: int = sec_cfg["overlap_tokens"]
    log.info("Selected embedding model: %s", model_name)
    log.info("Using max_tokens_per_chunk=%d based on model", max_tokens)
    tmpl: str = out_cfg["filename_template"]
    chunks_dir = base / out_cfg["chunks_dir"]
    manifest = base / out_cfg["manifest_file"]

    in_path = Path(args.input)
    if not in_path.exists():
        log.error("Input file not found: %s", in_path)
        sys.exit(1)

    log.info("Reading cleaned text: %s", in_path)
    text = in_path.read_text(encoding="utf-8", errors="replace")
    log.info("Text length: %d chars", len(text))

    sections = parse_sections(text)
    log.info("Found %d sections", len(sections))
    if not sections:
        log.warning(
            "No sections matched. Check that clean_text.py ran successfully "
            "and that section markers are in the form '§ 1910.XXX Title'."
        )

    chunks = list(make_chunks(sections, max_tokens, overlap, tmpl))
    log.info(
        "Generated %d chunks  (avg %d tokens)",
        len(chunks),
        sum(c.token_estimate for c in chunks) // max(len(chunks), 1),
    )

    save_chunks(chunks, chunks_dir, manifest)


if __name__ == "__main__":
    main()