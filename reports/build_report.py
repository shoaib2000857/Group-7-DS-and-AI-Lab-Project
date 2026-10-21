"""Build a milestone report PDF from its Markdown source.

Usage:
    python reports/build_report.py reports/Milestone1_Report.md "reports/DSAI Lab_Group 7_Milestone 1.pdf"

Requires: pip install markdown weasyprint
"""
import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML

HERE = Path(__file__).resolve().parent


def build(src: Path, out: Path) -> None:
    body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables", "md_in_html", "toc"])
    # Italic one-line paragraphs starting with "Table" or "Figure" are captions.
    body = re.sub(r"<p><em>((?:Table|Figure)[^<]*)</em></p>", r'<p class="caption"><em>\1</em></p>', body)
    # A paragraph holding only an image is a figure.
    body = re.sub(r"<p>(<img [^>]+>)</p>", r'<p class="figure">\1</p>', body)
    title = re.search(r'<h1 class="title">(.*?)</h1>', body)
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{title.group(1) if title else src.stem}</title>"
        f"<link rel='stylesheet' href='{(HERE / 'report.css').as_uri()}'></head>"
        f"<body>{body}</body></html>"
    )
    HTML(string=html, base_url=str(src.parent)).write_pdf(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    build(Path(sys.argv[1]), Path(sys.argv[2]))
