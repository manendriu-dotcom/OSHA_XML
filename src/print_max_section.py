"""
One-off: print total sections and the maximum token count among sections.
Falls back to word-count if `transformers` isn't installed.
"""
import re
from pathlib import Path

# Lightweight section parser (same logic as chunk_osha.parse_sections)
_SUBPART_RE = re.compile(r"^Subpart\s+([A-Z]+)—(.+)", re.MULTILINE)
_SECTION_RE = re.compile(r"^§\s+(1910\.\S+)\s+(.*)", re.MULTILINE)


def parse_sections(text: str):
    events = []
    for m in _SUBPART_RE.finditer(text):
        events.append((m.start(), "subpart", (m.group(1).strip(), m.group(2).strip())))
    for m in _SECTION_RE.finditer(text):
        events.append((m.start(), "section", (m.group(1).strip(), m.group(2).strip())))
    events.sort(key=lambda e: e[0])

    sections = []
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


# Try to load a tokenizer; fall back to whitespace token count
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


def main():
    base = Path(__file__).parent.parent
    in_path = base / "data" / "cleaned" / "osha_1910_cleaned.txt"
    if not in_path.exists():
        print("Input file not found:", in_path)
        return

    text = in_path.read_text(encoding="utf-8", errors="replace")
    sections = parse_sections(text)
    toks = [(count_tokens(s["body"]), s["section_id"], s["section_title"]) for s in sections]
    toks.sort(reverse=True)

    total_sections = len(sections)
    max_tok, max_id, max_title = toks[0] if toks else (0, None, None)

    print(f"Total sections: {total_sections}")
    print(f"Tokenizer used: {TOKENIZER_OK}")
    print(f"Max tokens in a section: {max_tok}")
    print(f"Section: {max_id} — {max_title}")
    print("\nTop 5 sections:")
    for t, sid, title in toks[:5]:
        print(f"- {t:6d} tokens — {sid} — {title}")


if __name__ == "__main__":
    main()
