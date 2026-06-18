"""
extract_text.py
---------------
Extracts structured text from the OSHA 1910 XML file (eCFR / GPO format).

Replaces the pdfplumber-based extractor.  XML gives us clean, already-wrapped
text with explicit structural tags — no line-wrap heuristics needed here.

What this script does:
  1. Parses 1910.xml with lxml (falls back to stdlib xml.etree if absent).
  2. Walks every <DIV> element, recognising:
       SUBCHAP / PART / SUBPART  → section-level headers
       SECTION                   → § 1910.XXX entries
       P / FP / NOTE / HD        → body paragraphs
       EXTRACT / GPOTABLE        → tables preserved as plain text
  3. Writes a single UTF-8 .txt file whose structure mirrors what
     clean_text.py and chunk_osha.py already expect:

       Subpart J—General Environmental Controls
       § 1910.141 Sanitation.
       (a) Scope and application. ...
       (b)(1) ...

  No line-wrap artefacts, no page headers/footers — the XML is already clean.

Usage:
    python src/extract_text.py
    python src/extract_text.py --input  data/raw/1910.xml \
                                --output data/extracted/osha_1910_raw_text.txt
"""

import argparse
import logging
import re
import sys
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
log = logging.getLogger("extract_text")


# ── XML library (prefer lxml for speed / namespace stripping) ─────────────────

def _get_etree():
    try:
        from lxml import etree as ET  # type: ignore
        log.info("Using lxml for XML parsing.")
        return ET, True
    except ImportError:
        import xml.etree.ElementTree as ET  # type: ignore
        log.info("lxml not found — using stdlib xml.etree (install lxml for speed).")
        return ET, False


# ── text helpers ──────────────────────────────────────────────────────────────

def _tag(element) -> str:
    """Return the local tag name without namespace prefix."""
    tag = element.tag
    if "}" in tag:
        return tag.split("}", 1)[1].upper()
    return tag.upper()


def _all_text(element) -> str:
    """
    Recursively collect all text content from an element and its children,
    joining with single spaces.  Strips internal whitespace runs.
    """
    parts: list[str] = []

    def _walk(el):
        if el.text:
            parts.append(el.text.strip())
        for child in el:
            _walk(child)
            if child.tail:
                parts.append(child.tail.strip())

    _walk(element)
    text = " ".join(p for p in parts if p)
    # Collapse internal whitespace
    return re.sub(r" {2,}", " ", text).strip()


def _table_to_text(table_el) -> str:
    """
    Convert a GPOTABLE or EXTRACT element to a readable plain-text block.
    Rows are tab-separated; an ASCII rule separates header from body.
    """
    lines: list[str] = []
    for row in table_el.iter():
        t = _tag(row)
        if t in ("TTITLE", "BOXHD"):
            txt = _all_text(row)
            if txt:
                lines.append(txt)
                lines.append("-" * min(len(txt), 80))
        elif t in ("ROW",):
            cells = [_all_text(c) for c in row if _tag(c) in ("ENT", "CHED")]
            if any(cells):
                lines.append("\t".join(cells))
    return "\n".join(lines)


# ── main extractor ────────────────────────────────────────────────────────────

# Tags that contain displayable paragraph text
_PARA_TAGS = {"P", "FP", "FP-1", "FP-2", "PSPACE", "APPRO", "STARS"}
# Tags that are section/subpart header lines
_HEAD_TAGS = {"HD", "HEAD", "SUBJECT"}
# Tags we skip entirely (metadata, XML boilerplate)
_SKIP_TAGS = {
    "AUTH", "SOURCE", "CITA", "EFFDNOT", "EFFD", "FTNT",
    "SECAUTH", "AMDPAR", "DATED", "SIGNER", "SIGNJOB",
    "ACT", "BILCOD",
}


def extract_section(section_el) -> list[str]:
    """
    Extract text from a <SECTION> element.
    Returns a list of logical lines in document order.
    """
    lines: list[str] = []

    for child in section_el:
        t = _tag(child)

        if t in _SKIP_TAGS:
            continue

        if t == "SECTNO":
            # § 1910.147  — emit on its own line; title follows on same line
            sec_no = _all_text(child)
            lines.append("")          # blank line before new section
            lines.append(sec_no)      # e.g. "§ 1910.147"

        elif t == "SUBJECT":
            # Section title — append to the SECTNO line we just added
            subj = _all_text(child)
            if lines and lines[-1].startswith("§"):
                lines[-1] = lines[-1] + " " + subj
            else:
                lines.append(subj)

        elif t in _HEAD_TAGS:
            txt = _all_text(child)
            if txt:
                lines.append(txt)

        elif t in _PARA_TAGS:
            txt = _all_text(child)
            if txt:
                lines.append(txt)

        elif t in ("NOTE", "NOTES"):
            # Regulatory note — prefix so chunk logic can identify it
            inner = _all_text(child)
            if inner:
                lines.append("Note: " + inner)

        elif t in ("GPOTABLE", "EXTRACT"):
            tbl = _table_to_text(child)
            if tbl:
                lines.append("")
                lines.append(tbl)
                lines.append("")

        elif t == "CITA":
            pass   # skip citations/authority lines

        else:
            # Catch-all: grab any text we haven't explicitly handled
            txt = _all_text(child)
            if txt:
                lines.append(txt)

    return lines


def extract_osha_xml(xml_path: Path, output_path: Path) -> None:
    ET, using_lxml = _get_etree()

    log.info("Parsing XML: %s", xml_path)
    try:
        if using_lxml:
            parser = ET.XMLParser(recover=True, encoding="utf-8")
            tree = ET.parse(str(xml_path), parser=parser)
            root = tree.getroot()
        else:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
    except Exception as exc:
        log.error("Failed to parse XML: %s", exc)
        sys.exit(1)

    output_lines: list[str] = []
    sections_found = 0
    subparts_found = 0

    # eCFR XML structure: CFRGRANULE > PART > SUBPART > SECTION
    # We walk the entire tree and emit content in document order.
    def walk(el, depth=0):
        nonlocal sections_found, subparts_found
        t = _tag(el)

        if t in _SKIP_TAGS:
            return

        if t in ("SUBPART", "SUBCHAP"):
            # Emit Subpart header line: "Subpart J—General Environmental Controls"
            hd_el = next((c for c in el if _tag(c) in _HEAD_TAGS), None)
            if hd_el is not None:
                hd_text = _all_text(hd_el).strip()
                # Normalise dash variants to em-dash for downstream pattern matching
                hd_text = re.sub(r"\s*[-–—]\s*", "—", hd_text, count=1)
                output_lines.append("")
                output_lines.append(hd_text)
                subparts_found += 1
            # Recurse into children (sections live inside subparts)
            for child in el:
                if _tag(child) not in _HEAD_TAGS:
                    walk(child, depth + 1)
            return

        if t == "SECTION":
            sec_lines = extract_section(el)
            output_lines.extend(sec_lines)
            sections_found += 1
            return

        # PART, CFRGRANULE, REGTEXT, etc. — just recurse
        for child in el:
            walk(child, depth + 1)

    walk(root)

    # Collapse 3+ blank lines to 2
    text = "\n".join(output_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    log.info("Extracted %d subparts, %d sections", subparts_found, sections_found)
    log.info("Saved raw text → %s  (%d chars)", output_path, len(text))


def main():
    base = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Extract text from OSHA 1910 XML.")
    parser.add_argument(
        "--input",
        default=str(base / "data" / "raw" / "1910.xml"),
    )
    parser.add_argument(
        "--output",
        default=str(base / "data" / "extracted" / "osha_1910_raw_text.txt"),
    )
    args = parser.parse_args()

    extract_osha_xml(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()