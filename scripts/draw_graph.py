"""Render the compiled PlannerState graph as a Mermaid PNG.

draw_mermaid_png() only inspects the compiled graph's structure and never
invokes a node, so build_graph() can be called with dummy services here —
no Google credentials needed. By default it renders via the Mermaid.INK
API (network call to a third party); pass --pyppeteer to render locally
via headless Chromium instead (requires `pip install pyppeteer`).
"""

import sys
from pathlib import Path

from langchain_core.runnables.graph import MermaidDrawMethod

from agentic_secretary.graph import build_graph

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "graph.png"


def main() -> None:
    draw_method = (
        MermaidDrawMethod.PYPPETEER
        if "--pyppeteer" in sys.argv
        else MermaidDrawMethod.API
    )
    graph = build_graph(gmail_service=None, calendar_service=None)
    graph.get_graph().draw_mermaid_png(
        output_file_path=str(DEFAULT_OUTPUT_PATH),
        draw_method=draw_method,
    )
    print(f"Saved graph diagram to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
