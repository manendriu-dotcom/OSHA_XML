"""
clean_text.py
-------------
Cleans the extracted OSHA 1910 text produced by extract_text.py.

Because the XML source is already well-structured (no PDF line wrapping,
no page numbers, no running headers), this script's job is lighter than
the PDF version.  It focuses on:

  1. Normalising line endings and unicode characters.
  2. Removing any residual eCFR / GPO XML artefacts that slipped through
     (entity reference remnants, processing instructions, etc.).
  3. Normalising section / paragraph markers to the canonical forms that
     chunk_osha.py's regex patterns expect.
  4. Collapsing excessive blank lines.
  5. Normalising whitespace inside lines (tabs → spaces, multi-space → single).

No line-wrap joining is performed — the XML extractor already produces
complete logical lines.

Usage:
    python src/clean_text.py
    python src/clean_text.py --input  data/extracted/osha_1910_raw_text.txt \
                              --output data/cleaned/osha_1910_cleaned.txt
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
log = logging.getLogger("clean_text")


# ── individual cleaning passes ────────────────────────────────────────────────

def _normalise_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalise_unicode(text: str) -> str:
    """
    Replace common 'smart' / special characters with plain ASCII equivalents
    so downstream regex patterns (which use ASCII punctuation) work reliably.
    """
    replacements = {
        "\u2019": "'",   # right single quotation mark
        "\u2018": "'",   # left single quotation mark
        "\u201c": '"',   # left double quotation mark
        "\u201d": '"',   # right double quotation mark
        "\u2013": "-",   # en-dash  (paragraph ranges like "(a)(1)-(a)(3)")
        "\u2014": "—",   # em-dash  (keep — used in Subpart headers)
        "\u00a7": "§",   # section sign (already in source but normalise)
        "\u00b0": " degrees ",  # degree symbol in temperature requirements
        "\u00b1": "+/-",        # plus-minus
        "\u2264": "<=",
        "\u2265": ">=",
        "\u00bd": "1/2",
        "\u00bc": "1/4",
        "\u00be": "3/4",
        "\u00d7": "x",   # multiplication sign
        "\u2022": "-",   # bullet → hyphen
        "\u25a0": "-",   # black square → hyphen
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _remove_xml_artefacts(text: str) -> str:
    """
    Strip residual XML / eCFR artefacts that may survive extraction:
      - Entity references like &amp; &lt; &#160;
      - XML processing instructions <?...?>
      - Standalone XML tags that weren't unwrapped
      - eCFR display boilerplate lines
    """
    # XML/HTML entity references
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    # Processing instructions
    text = re.sub(r"<\?[^>]*\?>", "", text)
    # Stray tags (shouldn't be present after extraction but just in case)
    text = re.sub(r"<[^>]{1,80}>", "", text)
    # eCFR / GPO display boilerplate lines
    text = re.sub(r"^29 CFR Part 1910[^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Occupational Safety and Health Standards[^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^page\s+\d+\s+of\s+\d+\s*\n", "", text, flags=re.MULTILINE | re.IGNORECASE)
    # SOURCE / AUTH lines (legal authority citations at part level)
    text = re.sub(r"^SOURCE:[^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^AUTHORITY:[^\n]*\n", "", text, flags=re.MULTILINE)
    return text


def _normalise_section_markers(text: str) -> str:
    """
    Ensure section markers are in the exact form chunk_osha.py expects:
        § 1910.147 Control of hazardous energy...
    Handles cases where the extractor may have emitted:
        §1910.147  (no space)
        Sec. 1910.147
        1910.147  (bare number)
    """
    # §1910  → § 1910
    text = re.sub(r"§\s*(\d)", r"§ \1", text)
    # "Sec. 1910.XXX" → "§ 1910.XXX"
    text = re.sub(r"^Sec\.\s+(1910\.\S+)", r"§ \1", text, flags=re.MULTILINE)
    return text


def _normalise_subpart_headers(text: str) -> str:
    """
    Ensure Subpart headers use an em-dash with no surrounding spaces:
        Subpart J — General  →  Subpart J—General
        Subpart J - General  →  Subpart J—General
    """
    text = re.sub(
        r"^(Subpart\s+[A-Z]+)\s*[-–—]+\s*",
        r"\1—",
        text,
        flags=re.MULTILINE,
    )
    return text


def _normalise_whitespace_in_lines(text: str) -> str:
    """
    Within each line:
      - Replace tabs with a single space.
      - Collapse multiple consecutive spaces to one.
      - Strip trailing whitespace.
    Empty lines are preserved.
    """
    cleaned_lines = []
    for line in text.splitlines():
        line = line.replace("\t", " ")
        line = re.sub(r" {2,}", " ", line)
        line = line.rstrip()
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _collapse_blank_lines(text: str) -> str:
    """Reduce runs of 3+ blank lines to exactly two (one visual gap)."""
    return re.sub(r"\n{3,}", "\n\n", text)


# ── pipeline ──────────────────────────────────────────────────────────────────

def clean_osha_text(input_path, output_path):
    in_path = Path(input_path)
    out_path = Path(output_path)

    log.info("Reading extracted text: %s", in_path)
    text = in_path.read_text(encoding="utf-8", errors="replace")
    log.info("Input length: %d chars", len(text))

    text = _normalise_line_endings(text)
    text = _normalise_unicode(text)
    text = _remove_xml_artefacts(text)
    text = _normalise_section_markers(text)
    text = _normalise_subpart_headers(text)
    text = _normalise_whitespace_in_lines(text)
    text = _collapse_blank_lines(text)
    text = text.strip() + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    log.info("Cleaned text saved → %s  (%d chars)", out_path, len(text))


def main():
    base = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Clean extracted OSHA 1910 text.") #description appears if user asks for --help in cmd line
    parser.add_argument(
        "--input",
        default=str(base / "data" / "extracted" / "osha_1910_raw_text.txt"),
    )
    parser.add_argument(
        "--output",
        default=str(base / "data" / "cleaned" / "osha_1910_cleaned.txt"),
    )
    #"--" double dash indicates an optional arg. If not provided, default Path is used.
    args = parser.parse_args()
    clean_osha_text(args.input, args.output)


if __name__ == "__main__":
    main()