"""
max_paragraph_chunk_tokens.py
----------------------------
Compute the maximum token count among paragraph-level chunks produced by
paragraph-aware splitting (i.e., grouping paragraphs and buffering them)
BUT without performing the word-window fallback split for overlong paragraphs.

This mirrors `split_on_paragraphs()` grouping logic but treats oversized
paragraphs as single chunks so we can see their raw token sizes.

Usage:
    python src/max_paragraph_chunk_tokens.py
"""
import re
import yaml
from pathlib import Path

_PARA_START = re.compile(
    r"^(\(\s*[a-zA-Z0-9]{1,4}\s*\)"    # (a) (1) (iv) (A)
    r"|Note[\s:]"                        # Note: / Note to
    r"|\d+\.\s+[A-Z]"                   # 1. Scope
    r")"
)

# tokenizer fallback
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))
    TOKENIZER_OK = True
except Exception:
    def count_tokens(text: str) -> int:
        return len(text.split())
    TOKENIZER_OK = False


def paragraph_groups(text: str):
    lines = text.splitlines(keepends=True)
    groups = []
    current = []
    for line in lines:
        if _PARA_START.match(line.lstrip()) and current:
            groups.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        groups.append(current)
    return ["".join(g) for g in groups]


def paragraph_chunks_from_groups(groups, max_tokens, overlap):
    chunks = []
    buffer = []
    buf_tokens = 0

    for grp_text in groups:
        grp_tokens = count_tokens(grp_text)

        if grp_tokens > max_tokens:
            # Normally split_on_paragraphs would word-window split; here we
            # keep the oversized paragraph as a single chunk so we can measure it.
            if buffer:
                chunks.append(("".join(buffer), count_tokens("".join(buffer))))
                buffer = []
                buf_tokens = 0
            chunks.append((grp_text, grp_tokens))
            continue

        if buf_tokens + grp_tokens > max_tokens and buffer:
            chunks.append(("".join(buffer), count_tokens("".join(buffer))))
            buffer = buffer[-1:] if overlap > 0 else []
            buf_tokens = count_tokens("".join(buffer))

        buffer.append(grp_text)
        buf_tokens += grp_tokens

    if buffer:
        chunks.append(("".join(buffer), count_tokens("".join(buffer))))

    return chunks


def main():
    base = Path(__file__).parent.parent
    cfg_path = base / "config" / "chunking.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    models_cfg = cfg.get("models")
    if not models_cfg:
        raise SystemExit(
            f"Config is missing the 'models' block with chunking parameters: {cfg_path}"
        )

    # Use the first configured model so paragraph analysis stays deterministic
    # even when the user selects a different embedding model at runtime.
    default_model = next(iter(models_cfg))
    model_cfg = models_cfg[default_model]
    max_tokens = model_cfg["max_tokens_per_chunk"]
    overlap = model_cfg["overlap_tokens"]

    in_path = base / "data" / "cleaned" / "osha_1910_cleaned.txt"
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    text = in_path.read_text(encoding="utf-8", errors="replace")

    # find sections
    SECTION_RE = re.compile(r"^§\s+(1910\.\S+)\s+(.*)", re.MULTILINE)
    events = [(m.start(), "section", (m.group(1).strip(), m.group(2).strip())) for m in SECTION_RE.finditer(text)]
    events.sort(key=lambda e: e[0])

    sections = []
    for i, (pos, kind, data) in enumerate(events):
        sec_id, sec_title = data
        next_section_events = [e for e in events[i+1:] if e[1] == "section"]
        end_pos = next_section_events[0][0] if next_section_events else len(text)
        body = text[pos:end_pos].strip()
        sections.append((sec_id, sec_title, body))

    max_par_chunk = 0
    max_info = None
    total_par_chunks = 0

    max_section_tokens = 0
    longest_section = None

    for sec_id, sec_title, body in sections:
        # Section-level token count (entire section body)
        sec_tok = count_tokens(body)
        if sec_tok > max_section_tokens:
            max_section_tokens = sec_tok
            longest_section = (sec_id, sec_title)

        groups = paragraph_groups(body)
        par_chunks = paragraph_chunks_from_groups(groups, max_tokens, overlap)
        for chunk_text, tok in par_chunks:
            total_par_chunks += 1
            if tok > max_par_chunk:
                max_par_chunk = tok
                max_info = (sec_id, sec_title)

    print(f"Tokenizer used: {TOKENIZER_OK}")
    print(f"Total sections: {len(sections)}")
    print(f"Total paragraph-level chunks (no word-window fallback): {total_par_chunks}")
    print(f"Max tokens in a paragraph-level chunk: {max_par_chunk}")
    if max_info:
        print(f"Section containing largest paragraph-chunk: {max_info[0]} — {max_info[1]}")

    print(f"Max tokens in a whole section: {max_section_tokens}")
    if longest_section:
        print(f"Longest section: {longest_section[0]} — {longest_section[1]}")

if __name__ == '__main__':
    main()
