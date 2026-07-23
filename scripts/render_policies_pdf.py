"""Render seed_data/policies.md to a PDF for human reference.

seed_data/policies.md is the source of truth (what build_policy_index()
ingests) — this PDF is a convenience artifact only, regenerated from the
markdown, never hand-edited.
"""

from pathlib import Path

import markdown
from xhtml2pdf import pisa

SOURCE_PATH = Path(__file__).resolve().parent.parent / "seed_data" / "policies.md"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "seed_data" / "policies.pdf"

HTML_TEMPLATE = """\
<html>
<head>
<style>
  body {{ font-family: Helvetica, sans-serif; font-size: 11pt; }}
  h1 {{ font-size: 18pt; margin-top: 24pt; page-break-before: always; }}
  h1:first-of-type {{ page-break-before: avoid; }}
  h2 {{ font-size: 13pt; margin-top: 14pt; color: #333333; }}
  p {{ line-height: 1.4; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    markdown_text = SOURCE_PATH.read_text(encoding="utf-8")
    body_html = markdown.markdown(markdown_text)
    full_html = HTML_TEMPLATE.format(body=body_html)

    with OUTPUT_PATH.open("wb") as f:
        result = pisa.CreatePDF(full_html, dest=f)

    if result.err:
        raise RuntimeError(f"Failed to render PDF ({result.err} error(s))")

    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
