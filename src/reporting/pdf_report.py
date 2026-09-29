"""Generate a cautious, contract-aligned PDF entirely in memory."""
from datetime import datetime
from io import BytesIO
from textwrap import wrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


PAGE_WIDTH, PAGE_HEIGHT = letter
INK = colors.HexColor("#172D38")
TEAL = colors.HexColor("#17665F")
INK_SOFT = colors.HexColor("#516572")
LINE = colors.HexColor("#D6DFDC")
SURFACE_SOFT = colors.HexColor("#EEF4F1")
AMBER = colors.HexColor("#A66513")
AMBER_SOFT = colors.HexColor("#FFF0CF")


def _optional(document, method, *args, **kwargs):
    """Keep the tiny canvas test double compatible with visual helpers."""
    callback = getattr(document, method, None)
    if callback is not None:
        callback(*args, **kwargs)


def create_report(disease, values, result) -> bytes:
    """Return one independent PDF byte stream without writing to disk."""
    output = BytesIO()
    document = canvas.Canvas(output, pagesize=letter, pageCompression=0)
    page_number = 1
    y = PAGE_HEIGHT - 62

    def start_page():
        nonlocal y, page_number
        _optional(document, "setFillColor", INK)
        _optional(document, "setFont", "Helvetica-Bold", 8)
        _optional(document, "drawString", 50, PAGE_HEIGHT - 34, "MEDICAL AI SUITE  /  RESEARCH REPORT")
        _optional(document, "setStrokeColor", LINE)
        _optional(document, "line", 50, PAGE_HEIGHT - 43, PAGE_WIDTH - 50, PAGE_HEIGHT - 43)
        y = PAGE_HEIGHT - 68

    def finish_page():
        _optional(document, "setStrokeColor", LINE)
        _optional(document, "line", 50, 43, PAGE_WIDTH - 50, 43)
        _optional(document, "setFillColor", INK_SOFT)
        _optional(document, "setFont", "Helvetica", 8)
        _optional(document, "drawString", 50, 28, "Educational and research use only - not for diagnosis or clinical screening")
        _optional(document, "drawRightString", PAGE_WIDTH - 50, 28, f"Page {page_number}")

    def ensure_space(lines=1):
        nonlocal y, page_number
        if y < 72 + (lines * 15):
            finish_page()
            document.showPage()
            page_number += 1
            start_page()

    def draw_text(value, *, font="Helvetica", size=10, color=INK, indent=0, leading=15, before=0, after=0):
        nonlocal y
        y -= before
        _optional(document, "setFont", font, size)
        _optional(document, "setFillColor", color)
        for segment in wrap(str(value), width=91, subsequent_indent="  ") or [""]:
            ensure_space()
            _optional(document, "drawString", 50 + indent, y, segment)
            y -= leading
        y -= after

    start_page()
    relation = "at or above" if result["threshold_result"] == "at_or_above" else "below"
    status = result["release_status"]
    if status == "experimental":
        status = "EXPERIMENTAL - limited subject-disjoint evidence"

    _optional(document, "setFillColor", INK)
    _optional(document, "setFont", "Helvetica-Bold", 18)
    _optional(document, "drawString", 50, y, "Educational model report")
    y -= 27
    draw_text("Medical AI Suite", font="Helvetica-Bold", size=10, color=TEAL, after=10)
    _optional(document, "setFillColor", AMBER if result["release_status"] == "experimental" else TEAL)
    _optional(document, "setFont", "Helvetica-Bold", 9)
    _optional(document, "drawString", 50, y, f"Module: {disease}")
    y -= 14
    _optional(document, "drawString", 50, y, f"Release status: {status}")
    y -= 25

    facts = [
        f"Model classification: threshold class {result['prediction']}",
        f"Uncalibrated model score: {result['model_score']:.4f}",
        f"Decision threshold: {result['decision_threshold']:.4f}",
        f"Threshold result: {relation} the decision threshold",
        "Score type: uncalibrated model score (not disease probability)",
        f"Model identifier: {result['model_identifier']}",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
    ]
    for fact in facts:
        draw_text(fact, size=10, color=INK, leading=16)

    draw_text("Intended use", font="Helvetica-Bold", size=11, color=TEAL, before=12, after=4)
    draw_text(result["intended_use"], color=INK_SOFT, leading=15)
    draw_text("Known limitations:", font="Helvetica-Bold", size=11, color=TEAL, before=14, after=4)
    for limitation in result["limitations"]:
        draw_text(f"- {limitation}", color=INK_SOFT, indent=4, leading=15)

    draw_text("Input measurements:", font="Helvetica-Bold", size=11, color=TEAL, before=14, after=4)
    for key, value in values.items():
        draw_text(f"{key}: {value if value is not None else 'Not provided'}", color=INK_SOFT, leading=15)

    # Keep the safety statement together and visually distinct near the end.
    ensure_space(4)
    _optional(document, "setFillColor", SURFACE_SOFT)
    _optional(document, "roundRect", 46, max(56, y - 62), PAGE_WIDTH - 92, 58, 6, fill=1, stroke=0)
    draw_text("RESEARCH-USE DISCLAIMER:", font="Helvetica-Bold", size=9, color=TEAL, indent=10, before=2, after=3)
    draw_text(result["disclaimer"], color=INK, indent=10, leading=14)
    draw_text("Consult a qualified healthcare professional for medical advice.", color=INK_SOFT, indent=10, before=2)
    finish_page()
    document.save()
    return output.getvalue()
