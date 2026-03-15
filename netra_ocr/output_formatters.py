# output_formatters.py
"""
Multi-format OCR output formatters.

Segment schema
--------------
Text segment:
    { "type":"text",  "text":str, "crop":PIL.Image,
      "bbox":(x1,y1,x2,y2), "label":str }

Image segment (Table / Picture / Formula):
    { "type":"image", "crop":PIL.Image,
      "bbox":(x1,y1,x2,y2), "label":str }

Format behaviour
----------------
Format  │ Text segments          │ Visual segments
────────┼────────────────────────┼─────────────────────────────
.txt    │ OCR string             │ [Label] placeholder
.md     │ OCR string + markup    │ [Label] blockquote
.html   │ <div> with OCR text    │ <img> of crop
.pdf    │ text via Khmer font;   │ image crop
        │ falls back to crop img │
        │ if no font available   │
.docx   │ floating text box,     │ floating anchored image
        │ Khmer font specified   │
"""

from __future__ import annotations

import io, os, base64
from html import escape as html_escape
from pathlib import Path
from typing import List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

# ── label helpers ──────────────────────────────────────────────────────────
VISUAL_LABELS   = {"Table", "Picture", "Figure", "Formula"}
BOLD_LABELS     = {"Title", "Section-header", "Page-header"}
ITALIC_LABELS   = {"Caption", "Footnote", "Page-footer"}
LIST_LABEL      = "List-item"

LABEL_FONT_SCALE = {
    "Title":          1.6,
    "Section-header": 1.25,
    "Page-header":    1.1,
    "Caption":        0.85,
    "Footnote":       0.75,
    "Page-footer":    0.75,
}

# Khmer font candidates (checked in order on the host system)
_KHMER_FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/KhmerUI.ttf",
    "C:/Windows/Fonts/KhmerUIb.ttf",
    "C:/Windows/Fonts/leelawad.ttf",
    # Linux
    "/usr/share/fonts/truetype/khmeros/KhmerOS.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifKhmer-Regular.ttf",
    # Local project fonts/ directory
    os.path.join(os.path.dirname(__file__), "fonts", "KhmerOS.ttf"),
    os.path.join(os.path.dirname(__file__), "fonts", "NotoSansKhmer-Regular.ttf"),
]

# Preferred Khmer font name for DOCX (Word uses the system font by this name)
_DOCX_KHMER_FONTS = ["Khmer OS Siemreap", "Khmer OS", "Noto Sans Khmer", "Leelawadee UI"]


def _find_khmer_ttf() -> Optional[str]:
    for p in _KHMER_FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _to_png_bytes(crop) -> bytes:
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def _to_b64_uri(crop) -> str:
    return "data:image/png;base64," + base64.b64encode(_to_png_bytes(crop)).decode()


# ══════════════════════════════════════════════════════════════════════════════
#  TXT
# ══════════════════════════════════════════════════════════════════════════════
def save_txt(segments: List[dict], output_path: str) -> None:
    lines = []
    for seg in segments:
        if seg["type"] == "text":
            t = seg["text"].strip()
            if t:
                lines.append(t)
        else:
            lines.append(f"[{seg.get('label', 'Image')}]")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[Formatter] TXT  → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MARKDOWN
# ══════════════════════════════════════════════════════════════════════════════
def save_markdown(segments: List[dict], output_path: str) -> None:
    out: List[str] = []
    for seg in segments:
        if seg["type"] == "image":
            out.append(f"> [{seg.get('label', 'Image')}]\n\n")
            continue
        text  = seg["text"].strip()
        if not text:
            continue
        label = seg.get("label", "Text")
        if label == "Title":
            out.append(f"# {text}\n\n")
        elif label == "Section-header":
            out.append(f"## {text}\n\n")
        elif label == "Page-header":
            out.append(f"### {text}\n\n")
        elif label == LIST_LABEL:
            out.append(f"- {text}\n")
        elif label in ITALIC_LABELS:
            out.append(f"*{text}*\n\n")
        else:
            out.append(f"{text}\n\n")
    Path(output_path).write_text("".join(out), encoding="utf-8")
    print(f"[Formatter] MD   → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  HTML
#  Text segments → <div> with actual OCR text, absolutely positioned
#  Visual segments → <img> of the crop, absolutely positioned
# ══════════════════════════════════════════════════════════════════════════════
def save_html(
    segments: List[dict],
    output_path: str,
    image_size: Tuple[int, int],
) -> None:
    img_w, img_h = image_size
    aspect = (img_h / img_w) * 100
    els: List[str] = []

    for seg in segments:
        bbox  = seg.get("bbox")
        label = seg.get("label", "Text")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        l = (x1 / img_w) * 100
        t = (y1 / img_h) * 100
        w = max(0.1, (x2 - x1) / img_w * 100)
        h = max(0.05, (y2 - y1) / img_h * 100)
        # font-size as % of page height so it scales with zoom
        line_h_pct = h * 0.72 * LABEL_FONT_SCALE.get(label, 1.0)

        if seg["type"] == "image":
            crop = seg.get("crop")
            if crop is None:
                continue
            els.append(
                f'<img class="vis {label.lower().replace("-","_")}"'
                f' src="{_to_b64_uri(crop)}" alt="{html_escape(label)}"'
                f' style="left:{l:.3f}%;top:{t:.3f}%;width:{w:.3f}%;height:{h:.3f}%;">'
            )
        else:
            text  = seg["text"].strip()
            if not text:
                continue
            bold   = "font-weight:700;" if label in BOLD_LABELS   else ""
            italic = "font-style:italic;" if label in ITALIC_LABELS else ""
            els.append(
                f'<div class="txt {label.lower().replace("-","_")}"'
                f' style="left:{l:.3f}%;top:{t:.3f}%;width:{w:.3f}%;height:{h:.3f}%;'
                f'font-size:{line_h_pct:.2f}%;{bold}{italic}">'
                f'{html_escape(text)}</div>'
            )

    html = f"""<!DOCTYPE html>
<html lang="km">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>OCR Output</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{
      background:#888;
      display:flex; justify-content:center; padding:32px 16px;
      font-family:'Khmer OS','Khmer UI','Noto Sans Khmer','Leelawadee UI',sans-serif;
    }}
    .page {{
      position:relative; width:100%; max-width:{img_w}px;
      padding-bottom:{aspect:.3f}%;
      background:#fff; box-shadow:0 8px 40px rgba(0,0,0,.4);
    }}
    .txt, .vis {{
      position:absolute; overflow:hidden;
      line-height:1.2; white-space:pre-wrap; word-break:break-word;
      color:#111;
    }}
    .vis {{ object-fit:contain; }}
    .title        {{ color:#000; }}
    .section_header {{ border-bottom:1px solid #ccc; padding-bottom:1px; }}
    .caption, .footnote, .page_footer {{ color:#555; }}
  </style>
</head>
<body><div class="page">
{"".join(els)}
</div></body>
</html>"""
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"[Formatter] HTML → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PDF
#
#  Root cause of broken Khmer rendering in the old ReportLab implementation:
#  ReportLab copies Unicode codepoints directly into the PDF glyph stream
#  without applying any OpenType shaping.  Khmer requires two shaping steps:
#    1. GSUB: coeng (U+17D2) + consonant → subscript glyph lookup
#    2. GPOS: mark-to-base / mark-to-mark for vowel and diacritic positioning
#  Without these, coeng appears as a raw box and the following consonant sits
#  inline instead of stacking below the base — exactly the reported bug.
#
#  Fix: build an HTML page (identical layout to save_html) and render it
#  through a browser engine that uses HarfBuzz for shaping:
#
#    WeasyPrint  (pip install weasyprint)   — Pango/HarfBuzz, best CSS support
#    Playwright  (pip install playwright)   — Chromium/Blink, best on Windows
#    wkhtmltopdf (system binary)            — WebKit, widely installed
#    ReportLab                              — last resort, crop-image fallback
#
#  All three browser-engine paths produce correctly shaped Khmer glyphs with
#  proper CIDToGIDMap and ToUnicode embedding.
# ══════════════════════════════════════════════════════════════════════════════

def _build_pdf_html(
    segments: List[dict],
    image_size: Tuple[int, int],
) -> str:
    """
    Build a self-contained HTML page that mirrors the document layout exactly,
    sized for A4 print output.  Identical structure to save_html() but:
      - page is fixed at A4 dimensions (no viewport scaling)
      - @font-face embeds every discovered Khmer font as base64 so the
        renderer uses it even when system fonts are unavailable
      - visual images are base64-embedded
    """
    img_w, img_h = image_size

    # A4 in px at 96 dpi
    A4_W_PX = 793.7   # 210mm
    A4_H_PX = 1122.5  # 297mm

    scale_x = A4_W_PX / img_w
    scale_y = A4_H_PX / img_h
    scale   = min(scale_x, scale_y)

    page_w = img_w * scale
    page_h = img_h * scale

    # ── embed Khmer font as base64 so renderer always uses it ────────────────
    font_face = ""
    ttf_path  = _find_khmer_ttf()
    if ttf_path:
        with open(ttf_path, "rb") as fh:
            font_b64 = base64.b64encode(fh.read()).decode()
        font_face = (
            "@font-face {"
            "  font-family: 'KhmerPDF';"
            f" src: url('data:font/truetype;base64,{font_b64}') format('truetype');"
            "}"
        )
        khmer_family = "'KhmerPDF','Khmer OS','Khmer UI','Noto Sans Khmer',sans-serif"
    else:
        khmer_family = "'Khmer OS','Khmer UI','Noto Sans Khmer','Leelawadee UI',sans-serif"

    # ── build element list ────────────────────────────────────────────────────
    els: List[str] = []
    for seg in segments:
        bbox  = seg.get("bbox")
        label = seg.get("label", "Text")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        left   = x1 * scale
        top    = y1 * scale
        width  = max(1.0, (x2 - x1) * scale)
        height = max(1.0, (y2 - y1) * scale)
        # Font size: proportional to bbox height, scale-independent
        line_h_frac = (y2 - y1) / img_h
        font_sz_pt  = max(6.0, min(22.0,
            line_h_frac * 841.89 * 0.58
            * LABEL_FONT_SCALE.get(label, 1.0)))

        pos = (f"position:absolute;left:{left:.2f}px;top:{top:.2f}px;"
               f"width:{width:.2f}px;height:{height:.2f}px;")

        if seg["type"] == "image":
            crop = seg.get("crop")
            if crop is None:
                continue
            els.append(
                f'<img src="{_to_b64_uri(crop)}" alt="{html_escape(label)}"'
                f' style="{pos}object-fit:fill;display:block;">'
            )
        else:
            text = seg["text"].strip()
            if not text:
                continue
            bold   = "font-weight:700;" if label in BOLD_LABELS   else ""
            italic = "font-style:italic;" if label in ITALIC_LABELS else ""
            # Extend text div to the right edge so text never wraps
            right_w = page_w - left
            els.append(
                f'<div style="{pos}width:{right_w:.2f}px;'
                f'font-size:{font_sz_pt:.2f}pt;{bold}{italic}'
                f'line-height:1.15;white-space:nowrap;overflow:visible;">'
                f'{html_escape(text)}</div>'
            )

    return f"""<!DOCTYPE html>
<html lang="km">
<head>
  <meta charset="UTF-8">
  <style>
    {font_face}
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    html, body {{ width:{page_w:.2f}px; height:{page_h:.2f}px; background:#fff; }}
    body {{ font-family:{khmer_family}; }}
    .page {{
      position:relative;
      width:{page_w:.2f}px; height:{page_h:.2f}px;
      overflow:hidden;
    }}
  </style>
</head>
<body><div class="page">
{"".join(els)}
</div></body>
</html>"""


def _render_pdf_weasyprint(html: str, output_path: str) -> bool:
    """Render via WeasyPrint (Pango/HarfBuzz). Returns True on success."""
    try:
        import weasyprint
        wp_doc = weasyprint.HTML(string=html).write_pdf()
        with open(output_path, "wb") as fh:
            fh.write(wp_doc)
        print("  [PDF] Renderer: WeasyPrint (Pango/HarfBuzz)")
        return True
    except Exception as exc:
        print(f"  [PDF] WeasyPrint failed: {exc}")
        return False


def _render_pdf_playwright(html: str, output_path: str) -> bool:
    """Render via Playwright/Chromium (Blink/HarfBuzz). Returns True on success."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page    = browser.new_page()
            page.set_content(html, wait_until="networkidle")
            page.pdf(
                path=output_path,
                format="A4",
                print_background=True,
                margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
            )
            browser.close()
        print("  [PDF] Renderer: Playwright/Chromium (Blink/HarfBuzz)")
        return True
    except Exception as exc:
        print(f"  [PDF] Playwright failed: {exc}")
        return False


def _render_pdf_wkhtmltopdf(html: str, output_path: str) -> bool:
    """Render via wkhtmltopdf (WebKit). Returns True on success."""
    try:
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False,
                                         mode="w", encoding="utf-8") as fh:
            fh.write(html)
            tmp_html = fh.name
        result = subprocess.run(
            ["wkhtmltopdf", "--quiet",
             "--page-size", "A4",
             "--margin-top", "0", "--margin-bottom", "0",
             "--margin-left", "0", "--margin-right", "0",
             "--enable-local-file-access",
             tmp_html, output_path],
            capture_output=True, timeout=60,
        )
        os.unlink(tmp_html)
        if result.returncode == 0 and os.path.exists(output_path):
            print("  [PDF] Renderer: wkhtmltopdf (WebKit)")
            return True
        print(f"  [PDF] wkhtmltopdf exit {result.returncode}: {result.stderr[:200]}")
        return False
    except Exception as exc:
        print(f"  [PDF] wkhtmltopdf failed: {exc}")
        return False


def _render_pdf_reportlab_fallback(
    segments: List[dict],
    output_path: str,
    image_size: Tuple[int, int],
) -> None:
    """
    Last-resort ReportLab renderer.
    Text segments render as crop images (no shaping artifacts, always legible).
    Visual segments render as their crop image.
    Correct Khmer shaping is NOT possible via this path.
    """
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader

    img_w, img_h = image_size
    page_w, page_h = A4
    scale    = min(page_w / img_w, page_h / img_h)
    offset_x = (page_w - img_w * scale) / 2.0
    offset_y = (page_h - img_h * scale) / 2.0

    c = rl_canvas.Canvas(output_path, pagesize=A4)
    for seg in segments:
        bbox = seg.get("bbox")
        crop = seg.get("crop")
        if not bbox or crop is None:
            continue
        x1, y1, x2, y2 = bbox
        pdf_x = offset_x + x1 * scale
        pdf_y = page_h - offset_y - y2 * scale
        pdf_w = max(1.0, (x2 - x1) * scale)
        pdf_h = max(1.0, (y2 - y1) * scale)
        try:
            c.drawImage(
                ImageReader(io.BytesIO(_to_png_bytes(crop))),
                float(pdf_x), float(pdf_y),
                width=float(pdf_w), height=float(pdf_h),
                preserveAspectRatio=False, mask="auto",
            )
        except Exception:
            pass
    c.save()
    print("  [PDF] Renderer: ReportLab (crop-image fallback — Khmer shaping not supported)")


def save_pdf(
    segments: List[dict],
    output_path: str,
    image_size: Tuple[int, int],
) -> None:
    """
    Generate a PDF with correct Khmer OpenType shaping.

    Tries renderers in order of shaping quality:
      1. WeasyPrint  (pip install weasyprint)   — Pango + HarfBuzz
      2. Playwright  (pip install playwright)   — Chromium + HarfBuzz
      3. wkhtmltopdf (system binary)            — WebKit
      4. ReportLab                              — crop-image fallback

    All three browser-engine paths use HarfBuzz which applies Khmer GSUB
    (coeng substitution) and GPOS (mark positioning) rules correctly.
    """
    html = _build_pdf_html(segments, image_size)

    if _render_pdf_weasyprint(html, output_path):
        pass
    elif _render_pdf_playwright(html, output_path):
        pass
    elif _render_pdf_wkhtmltopdf(html, output_path):
        pass
    else:
        print("  [PDF] All shaping renderers failed — using ReportLab crop-image fallback")
        _render_pdf_reportlab_fallback(segments, output_path, image_size)

    print(f"[Formatter] PDF  → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  DOCX
#  Text segments → floating text boxes with explicit Khmer font name
#                  (Word resolves the font from the system)
#  Visual segments → floating anchored images
# ══════════════════════════════════════════════════════════════════════════════
def save_docx(
    segments: List[dict],
    output_path: str,
    image_size: Tuple[int, int],
) -> None:
    try:
        from docx import Document
        from docx.shared import Emu
        from docx.opc.part import Part
        from docx.opc.packuri import PackURI
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
        import lxml.etree as etree
    except ImportError:
        raise ImportError("pip install python-docx")

    img_w, img_h = image_size
    PAGE_W, PAGE_H = 11_906_400, 16_838_400   # A4 in EMU
    MARGIN    = 720_720
    CONTENT_W = PAGE_W - 2 * MARGIN
    CONTENT_H = PAGE_H - 2 * MARGIN
    scale     = min(CONTENT_W / img_w, CONTENT_H / img_h)

    doc = Document()
    sec = doc.sections[0]
    sec.page_width    = Emu(PAGE_W);  sec.page_height   = Emu(PAGE_H)
    sec.left_margin   = sec.right_margin  = Emu(MARGIN)
    sec.top_margin    = sec.bottom_margin = Emu(MARGIN)
    for p in doc.paragraphs:
        p._element.getparent().remove(p._element)

    draw_id   = 1
    img_index = 1

    # Choose the best available Khmer font name for Word to resolve
    khmer_font = _DOCX_KHMER_FONTS[0]   # "Khmer UI" — ships with Windows 8+

    def _anchor_head(emu_x, emu_y, emu_w, emu_h, did):
        return (
            f'<wp:anchor simplePos="0" relativeHeight="251658240" behindDoc="0"'
            f' locked="0" layoutInCell="1" allowOverlap="1"'
            f' distT="0" distB="0" distL="0" distR="0">'
            f'<wp:simplePos x="0" y="0"/>'
            f'<wp:positionH relativeFrom="page"><wp:posOffset>{emu_x}</wp:posOffset></wp:positionH>'
            f'<wp:positionV relativeFrom="page"><wp:posOffset>{emu_y}</wp:posOffset></wp:positionV>'
            f'<wp:extent cx="{emu_w}" cy="{emu_h}"/>'
            f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:wrapNone/>'
            f'<wp:docPr id="{did}" name="E{did}"/>'
        )

    for seg in segments:
        bbox = seg.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        emu_x = int(MARGIN + x1 * scale)
        emu_y = int(MARGIN + y1 * scale)
        emu_h = max(9_144, int((y2 - y1) * scale))
        label = seg.get("label", "Text")

        if seg["type"] == "image":
            # Images: preserve exact detected bbox width
            emu_w = max(9_144, int((x2 - x1) * scale))
        else:
            # Text: extend to right content boundary.
            # Surya bboxes are tight ink-pixel bounds. When Word renders Khmer UI
            # at the computed font size, even slightly wider glyph metrics cause
            # text to wrap inside the narrow box. spAutoFit then grows the box
            # downward, pushing it into the next element → clutter and overlap.
            # Extending to the right margin gives the text all available horizontal
            # space, completely preventing wrapping for single-column documents.
            emu_w = max(9_144, PAGE_W - MARGIN - emu_x)

        # ── visual segment → anchored image ───────────────────────────────
        if seg["type"] == "image":
            crop = seg.get("crop")
            if crop is None:
                continue
            png   = _to_png_bytes(crop)
            puri  = PackURI(f"/word/media/img_{img_index:04d}.png")
            part  = Part(puri, "image/png", png, doc.part.package)
            rId   = doc.part.relate_to(part, RT.IMAGE)
            img_index += 1

            xml = (
                f'<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
                f' xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
                f' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
                f' xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
                f' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f'<w:r><w:drawing>'
                + _anchor_head(emu_x, emu_y, emu_w, emu_h, draw_id) +
                f'<wp:cNvGraphicFramePr>'
                f'<a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
                f'<a:graphic><a:graphicData'
                f' uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                f'<pic:pic>'
                f'<pic:nvPicPr><pic:cNvPr id="{draw_id}" name="E{draw_id}"/>'
                f'<pic:cNvPicPr/></pic:nvPicPr>'
                f'<pic:blipFill>'
                f'<a:blip r:embed="{rId}"/>'
                f'<a:stretch><a:fillRect/></a:stretch>'
                f'</pic:blipFill>'
                f'<pic:spPr>'
                f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{emu_w}" cy="{emu_h}"/></a:xfrm>'
                f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
                f'</pic:spPr>'
                f'</pic:pic></a:graphicData></a:graphic>'
                f'</wp:anchor></w:drawing></w:r></w:p>'
            )

        # ── text segment → floating text box ──────────────────────────────
        else:
            text = seg["text"].strip()
            if not text:
                continue

            bold     = label in BOLD_LABELS
            italic   = label in ITALIC_LABELS
            # Derive font size from bbox pixel height relative to image height,
            # then project onto A4 page points.
            #
            # OLD (buggy): half_pt = f(emu_h)
            #   emu_h = bbox_px × scale, and scale can be 17 000+ EMU/px for small
            #   images, so emu_h becomes huge → font formula produces 28–36pt body
            #   text → characters are wider than the tight bbox → text wraps.
            #
            # NEW: font_sz ∝ (bbox_px / img_h) × A4_page_height_pt
            #   This is scale-independent: a line that is 3% of the image height
            #   maps to 3% of the A4 page height regardless of image resolution.
            _A4_PT = 841.89
            line_h_frac = (y2 - y1) / img_h
            font_sz_pt  = max(7.0, min(20.0,
                line_h_frac * _A4_PT * 0.60
                * LABEL_FONT_SCALE.get(label, 1.0)))
            half_pt = int(font_sz_pt * 2)
            safe_t   = xml_escape(text)

            rpr_bold   = "<w:b/><w:bCs/>"         if bold   else ""
            rpr_italic = "<w:i/><w:iCs/>"          if italic else ""
            rpr = (
                f'<w:rFonts w:ascii="{khmer_font}" w:hAnsi="{khmer_font}"'
                f' w:cs="{khmer_font}"/>'
                f'<w:lang w:bidi="km"/>'
                f'{rpr_bold}{rpr_italic}'
                f'<w:sz w:val="{half_pt}"/><w:szCs w:val="{half_pt}"/>'
            )

            xml = (
                f'<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
                f' xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
                f' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
                f' xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"'
                f' xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
                f' xmlns:v="urn:schemas-microsoft-com:vml">'
                f'<w:r><w:rPr><w:noProof/></w:rPr>'
                f'<mc:AlternateContent><mc:Choice Requires="wps">'
                f'<w:drawing>'
                + _anchor_head(emu_x, emu_y, emu_w, emu_h, draw_id) +
                f'<wp:cNvGraphicFramePr/>'
                f'<a:graphic><a:graphicData'
                f' uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
                f'<wps:wsp>'
                f'<wps:cNvSpPr txBox="1"><a:spLocks noChangeArrowheads="1"/></wps:cNvSpPr>'
                f'<wps:spPr>'
                f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{emu_w}" cy="{emu_h}"/></a:xfrm>'
                f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
                f'<a:noFill/><a:ln><a:noFill/></a:ln>'
                f'</wps:spPr>'
                f'<wps:txbx><w:txbxContent>'
                f'<w:p><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr>'
                f'<w:r><w:rPr>{rpr}</w:rPr>'
                f'<w:t xml:space="preserve">{safe_t}</w:t>'
                f'</w:r></w:p>'
                f'</w:txbxContent></wps:txbx>'
                # spAutoFit: Word expands the textbox height to fit rendered
                # content, so tall Khmer stacked diacritics / subscripts /
                # superscripts are never clipped.
                # insT/insB=0 removes vertical padding so the baseline sits
                # where the detected bbox expects it.
                f'<wps:bodyPr anchor="t" insL="45720" insR="45720"'
                f' insT="0" insB="0">'
                f'<a:spAutoFit/>'
                f'</wps:bodyPr>'
                f'</wps:wsp></a:graphicData></a:graphic>'
                f'</wp:anchor></w:drawing>'
                f'</mc:Choice>'
                f'<mc:Fallback><w:pict><v:textbox>'
                f'<w:txbxContent><w:p><w:r>'
                f'<w:t xml:space="preserve">{safe_t}</w:t>'
                f'</w:r></w:p></w:txbxContent>'
                f'</v:textbox></w:pict></mc:Fallback>'
                f'</mc:AlternateContent></w:r></w:p>'
            )

        try:
            doc.element.body.append(etree.fromstring(xml))
            draw_id += 1
        except Exception as exc:
            print(f"  [DOCX] skipped segment at {bbox}: {exc}")

    doc.save(output_path)
    print(f"[Formatter] DOCX → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Dispatcher
# ══════════════════════════════════════════════════════════════════════════════
_FORMAT_MAP = {
    ".txt":  save_txt,
    ".md":   save_markdown,
    ".html": save_html,
    ".htm":  save_html,
    ".pdf":  save_pdf,
    ".docx": save_docx,
}
SUPPORTED_FORMATS = list(_FORMAT_MAP.keys())


def save_output(
    segments: List[dict],
    output_path: str,
    image_size: Optional[Tuple[int, int]] = None,
) -> None:
    ext = Path(output_path).suffix.lower()
    fn  = _FORMAT_MAP.get(ext)
    if fn is None:
        print(f"[Formatter] Unknown extension '{ext}' — falling back to .txt")
        fn, output_path = save_txt, str(Path(output_path).with_suffix(".txt"))

    import inspect
    if "image_size" in inspect.signature(fn).parameters:
        if image_size is None:
            raise ValueError(f"image_size=(w,h) required for '{ext}'")
        fn(segments, output_path, image_size)
    else:
        fn(segments, output_path)