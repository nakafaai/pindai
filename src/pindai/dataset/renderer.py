"""High-fidelity exam/worksheet HTML + KaTeX renderer using Playwright."""

from __future__ import annotations

import html
import json
from pathlib import Path

from PIL import Image

from pindai.dataset.augment import apply_augmentation
from pindai.dataset.schema import LayoutStyle, StemSample

KATEX_CSS_CDN = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
KATEX_JS_CDN = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
KATEX_AUTORENDER_CDN = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
MARKED_JS_CDN = "https://cdn.jsdelivr.net/npm/marked@15.0.0/marked.min.js"

LAYOUT_CSS: dict[LayoutStyle, str] = {
    LayoutStyle.EXAM_SHEET: """
        body { font-family: "Times New Roman", Georgia, serif; background: #ffffff; color: #111; padding: 36px 44px; }
        .sheet { max-width: 780px; margin: 0 auto; }
        .header { border-bottom: 2px solid #111; padding-bottom: 8px; margin-bottom: 20px; font-size: 13px; letter-spacing: 0.04em; text-transform: uppercase; display: flex; justify-content: space-between; }
    """,
    LayoutStyle.WORKSHEET_BOXED: """
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #fafafa; color: #18181b; padding: 32px; }
        .sheet { max-width: 760px; margin: 0 auto; background: #fff; border: 1.5px solid #d4d4d8; border-radius: 10px; padding: 28px 32px; }
        .header { font-size: 12px; color: #52525b; border-bottom: 1px dashed #d4d4d8; padding-bottom: 8px; margin-bottom: 16px; display: flex; justify-content: space-between; }
    """,
    LayoutStyle.TEXTBOOK_CARD: """
        body { font-family: Georgia, "Noto Serif", serif; background: #fdfcf8; color: #1c1917; padding: 36px; }
        .sheet { max-width: 740px; margin: 0 auto; border-left: 4px solid #0f766e; padding-left: 24px; }
        .header { font-size: 12px; color: #0f766e; font-weight: 700; margin-bottom: 14px; display: flex; justify-content: space-between; }
    """,
    LayoutStyle.COMPACT_PRINT: """
        body { font-family: Arial, Helvetica, sans-serif; background: #ffffff; color: #09090b; padding: 24px 28px; font-size: 15px; }
        .sheet { max-width: 680px; margin: 0 auto; }
        .header { font-size: 11px; color: #3f3f46; border-bottom: 1px solid #a1a1aa; padding-bottom: 6px; margin-bottom: 14px; display: flex; justify-content: space-between; }
    """,
}


def build_sample_html(sample: StemSample) -> str:
    """Build a self-contained HTML page that renders Markdown + LaTeX via KaTeX."""
    header_subject = html.escape(sample.subject.value.upper())
    header_topic = html.escape(sample.topic.replace("_", " ").title())
    md_json = json.dumps(sample.markdown)
    layout_css = LAYOUT_CSS[sample.layout]
    return f"""<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8" />
  <link rel="stylesheet" href="{KATEX_CSS_CDN}" />
  <script src="{MARKED_JS_CDN}"></script>
  <script src="{KATEX_JS_CDN}"></script>
  <script src="{KATEX_AUTORENDER_CDN}"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; line-height: 1.65; font-size: 17px; }}
    h3 {{ margin-top: 0; margin-bottom: 12px; font-size: 19px; }}
    p {{ margin: 10px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 15px; }}
    th, td {{ border: 1px solid #52525b; padding: 7px 12px; text-align: center; }}
    th {{ background: #f4f4f5; font-weight: 600; }}
    ol, ul {{ padding-left: 22px; }}
    {layout_css}
  </style>
</head>
<body>
  <div class="sheet" id="capture">
    <div class="header">
      <span>LATIHAN STEM INDONESIA · {header_subject}</span>
      <span>{header_topic}</span>
    </div>
    <div id="content"></div>
  </div>
  <script>
    const rawMarkdown = {md_json};
    const placeholders = [];
    const protectedMd = rawMarkdown.replace(/\\$\\$[\\s\\S]+?\\$\\$|\\$(?:[^$\\\\]|\\\\.)+?\\$/g, (m) => {{
      const idx = placeholders.length;
      placeholders.push(m);
      return `@@MATH_${{idx}}@@`;
    }});
    let renderedHtml = marked.parse(protectedMd, {{ breaks: true }});
    renderedHtml = renderedHtml.replace(/@@MATH_(\\d+)@@/g, (_, idx) => placeholders[Number(idx)]);
    const container = document.getElementById("content");
    container.innerHTML = renderedHtml;
    renderMathInElement(container, {{
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
            viewport={"width": viewport_width, "height": 1200},
            device_scale_factor=device_scale_factor,
        )
        page = context.new_page()

        for sample in samples:
            page.set_content(build_sample_html(sample), wait_until="networkidle")
            page.wait_for_function("window.__PINDAI_RENDER_DONE === true", timeout=15_000)

            split_dir = output_dir / sample.split.value
            split_dir.mkdir(parents=True, exist_ok=True)
            img_path = split_dir / f"{sample.id}.png"

            locator = page.locator("#capture")
            raw_bytes = locator.screenshot(type="png")
            from io import BytesIO

            img = Image.open(BytesIO(raw_bytes)).convert("RGB")
            augmented = apply_augmentation(img, sample.augmentation, sample.seed)
            augmented.save(img_path, format="PNG")

            updated.append(
                sample.model_copy(update={"image_path": str(img_path.relative_to(output_dir))})
            )

        browser.close()

    return updated
