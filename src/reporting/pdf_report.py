"""Generate a cautious, contract-aligned PDF model report."""
from datetime import datetime
from pathlib import Path
from textwrap import wrap

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_report(path, disease, values, result):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    document = canvas.Canvas(str(path), pagesize=letter)
    y = 760
    document.setFont("Helvetica-Bold", 16)
    document.drawString(50, y, "Medical AI Suite - Educational model report")
    y -= 35
    document.setFont("Helvetica", 10)

    relation = "at or above" if result["threshold_result"] == "at_or_above" else "below"
    status = result["release_status"]
    if status == "experimental":
        status = "EXPERIMENTAL - limited subject-disjoint evidence"
    lines = [
        f"Module: {disease}",
        f"Release status: {status}",
        f"Model classification: threshold class {result['prediction']}",
        f"Uncalibrated model score: {result['model_score']:.4f}",
        f"Decision threshold: {result['decision_threshold']:.4f}",
        f"Threshold result: {relation} the decision threshold",
        "Score type: uncalibrated model score (not disease probability)",
        f"Model identifier: {result['model_identifier']}",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Intended use: {result['intended_use']}",
        "",
        "Known limitations:",
    ] + [f"- {item}" for item in result["limitations"]] + [
        "",
        "Input measurements:",
    ] + [f'{key}: {value if value is not None else "Not provided"}' for key, value in values.items()] + [
        "",
        f"RESEARCH-USE DISCLAIMER: {result['disclaimer']}",
    ]

    for line in lines:
        wrapped = wrap(line, width=95, subsequent_indent="  ") or [""]
        for segment in wrapped:
            if y < 55:
                document.showPage()
                document.setFont("Helvetica", 10)
                y = 760
            document.drawString(50, y, segment)
            y -= 18
    document.save()
