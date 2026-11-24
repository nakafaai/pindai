"""High-fidelity exam/worksheet HTML + KaTeX renderer using Playwright."""

from __future__ import annotations

import html
import re
from io import BytesIO
from pathlib import Path

from PIL import Image

from pindai.dataset.augment import apply_augmentation
from pindai.dataset.schema import LayoutStyle, StemSample

KATEX_CSS_CDN = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
KATEX_JS_CDN = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
KATEX_AUTORENDER_CDN = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"

LAYOUT_CSS: dict[LayoutStyle, str] = {
    LayoutStyle.EXAM_SHEET: """
        body { font-family: "Times New Roman", Georgia, serif; background: #ffffff; color: #111; padding: 36px 44px; }
        .sheet { max-width: 780px; margin: 0 auto; padding: 20px 24px; background: #ffffff; }
        .header { border-bottom: 2px solid #111; padding-bottom: 8px; margin-bottom: 20px; font-size: 13px; letter-spacing: 0.04em; text-transform: uppercase; display: flex; justify-content: space-between; }
    """,
    LayoutStyle.WORKSHEET_BOXED: """
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #fafafa; color: #18181b; padding: 32px; }
        .sheet { max-width: 760px; margin: 0 auto; background: #fff; border: 1.5px solid #d4d4d8; border-radius: 10px; padding: 28px 32px; }
        .header { font-size: 12px; color: #52525b; border-bottom: 1px dashed #d4d4d8; padding-bottom: 8px; margin-bottom: 16px; display: flex; justify-content: space-between; }
    """,
    LayoutStyle.TEXTBOOK_CARD: """
        body { font-family: Georgia, "Noto Serif", serif; background: #fdfcf8; color: #1c1917; padding: 36px; }
        .sheet { max-width: 740px; margin: 0 auto; border-left: 4px solid #0f766e; padding: 18px 24px; background: #fdfcf8; }
        .header { font-size: 12px; color: #0f766e; font-weight: 700; margin-bottom: 14px; display: flex; justify-content: space-between; }
    """,
    LayoutStyle.COMPACT_PRINT: """
        body { font-family: Arial, Helvetica, sans-serif; background: #ffffff; color: #09090b; padding: 24px 28px; font-size: 15px; }
        .sheet { max-width: 680px; margin: 0 auto; padding: 16px 20px; background: #ffffff; }
        .header { font-size: 11px; color: #3f3f46; border-bottom: 1px solid #a1a1aa; padding-bottom: 6px; margin-bottom: 14px; display: flex; justify-content: space-between; }
    """,
}

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MATH_RE = re.compile(r"(\$\$[\s\S]+?\$\$|\$(?:[^$\\]|\\.)+?\$)")


def _format_inline(text: str) -> str:
    """Escape HTML and apply **bold** while preserving $...$ and $$...$$ verbatim."""
    parts = _MATH_RE.split(text)
    out: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            out.append(html.escape(part))
        else:
            escaped = html.escape(part)
            out.append(_BOLD_RE.sub(r"<strong>\1</strong>", escaped))
    return "".join(out)


def markdown_to_html(markdown: str) -> str:
    """Convert IndoSTEM-OCR Markdown (headings, paragraphs, tables, lists, options) to HTML."""
    lines = markdown.replace("\r\n", "\n").split("\n")
    html_blocks: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Headings
        if line.startswith("### "):
            html_blocks.append(f"<h3>{_format_inline(line[4:])}</h3>")
            i += 1
            continue
        if line.startswith("## "):
            html_blocks.append(f"<h2>{_format_inline(line[3:])}</h2>")
            i += 1
            continue

        # Markdown table
        if line.startswith("|") and line.endswith("|"):
            table_lines: list[str] = []
            while i < n and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            if len(table_lines) >= 2:
                header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
                body_rows = [
                    [c.strip() for c in row.strip("|").split("|")] for row in table_lines[2:]
                ]
                thead = "".join(f"<th>{_format_inline(c)}</th>" for c in header_cells)
                tbody = "".join(
                    "<tr>" + "".join(f"<td>{_format_inline(c)}</td>" for c in row) + "</tr>"
                    for row in body_rows
                )
                html_blocks.append(
                    f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"
                )
                continue

        # Display math block
        if line.startswith("$$") and line.endswith("$$") and len(line) > 4:
            html_blocks.append(f'<div class="math-block">{html.escape(line)}</div>')
            i += 1
            continue

        # Regular paragraph / options / numbered lines grouped until blank line
        para_lines: list[str] = []
        while i < n and lines[i].strip():
            curr = lines[i].strip()
            if curr.startswith(("#", "|")):
                break
            para_lines.append(_format_inline(curr))
            i += 1
        html_blocks.append("<p>" + "<br/>".join(para_lines) + "</p>")

    return "\n".join(html_blocks)


def build_sample_html(sample: StemSample) -> str:
    """Build a self-contained HTML page with pre-rendered Markdown HTML and KaTeX math."""
    header_subject = html.escape(sample.subject.value.upper())
    header_topic = html.escape(sample.topic.replace("_", " ").title())
    body_html = markdown_to_html(sample.markdown)
    layout_css = LAYOUT_CSS[sample.layout]
    return f"""<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8" />
  <link rel="stylesheet" href="{KATEX_CSS_CDN}" />
  <script src="{KATEX_JS_CDN}"></script>
  <script src="{KATEX_AUTORENDER_CDN}"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; line-height: 1.7; font-size: 17px; }}
    h2, h3 {{ margin-top: 0; margin-bottom: 12px; font-size: 19px; }}
    p {{ margin: 10px 0; }}
    .math-block {{ margin: 14px 0; text-align: center; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 15px; }}
    th, td {{ border: 1px solid #52525b; padding: 7px 12px; text-align: center; }}
    th {{ background: #f4f4f5; font-weight: 600; }}
    {layout_css}
  </style>
</head>
<body>
  <div class="sheet" id="capture">
    <div class="header">
      <span>LATIHAN STEM INDONESIA · {header_subject}</span>
      <span>{header_topic}</span>
    </div>
    <div id="content">
      {body_html}
    </div>
  </div>
  <script>
    renderMathInElement(document.getElementById("content"), {{
      delimiters: [
        {{ left: "$$", right: "$$", display: true }},
        {{ left: "$", right: "$", display: false }}
      ],
      throwOnError: false
    }});
    window.__PINDAI_RENDER_DONE = true;
  </script>
</body>
</html>
"""


def render_samples(
    samples: list[StemSample],
    output_dir: Path,
    viewport_width: int = 900,
    device_scale_factor: float = 2.0,
) -> list[StemSample]:
    """Render each StemSample to an augmented PNG image using Playwright Chromium."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for rendering. Run `uv sync --extra render && uv run playwright install chromium`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    updated: list[StemSample] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": viewport_width, "height": 1400},
            device_scale_factor=device_scale_factor,
        )
        page = context.new_page()

        for sample in samples:
            page.set_content(build_sample_html(sample), wait_until="networkidle")
            page.wait_for_function(
                "window.__PINDAI_RENDER_DONE === true && document.querySelectorAll('.katex').length > 0",
                timeout=15_000,
            )

            split_dir = output_dir / sample.split.value
            split_dir.mkdir(parents=True, exist_ok=True)
            img_path = split_dir / f"{sample.id}.png"

            locator = page.locator("#capture")
            raw_bytes = locator.screenshot(type="png")
            img = Image.open(BytesIO(raw_bytes)).convert("RGB")
            if img.height < 180:
                raise RuntimeError(
                    f"Rendered image for {sample.id} is unexpectedly short ({img.width}x{img.height}); "
                    "check HTML/KaTeX rendering."
                )
            augmented = apply_augmentation(img, sample.augmentation, sample.seed)
            augmented.save(img_path, format="PNG")

            updated.append(
                sample.model_copy(update={"image_path": str(img_path.relative_to(output_dir))})
            )

        browser.close()

    return updated
