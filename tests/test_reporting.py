from src.reporting import pdf_report


class _Canvas:
    def __init__(self, *_args, **_kwargs):
        self.text = []

    def setFont(self, *_args):
        pass

    def drawString(self, _x, _y, value):
        self.text.append(value)

    def showPage(self):
        pass

    def save(self):
        pass


def test_pdf_uses_the_prediction_communication_contract(monkeypatch):
    output = _Canvas()
    monkeypatch.setattr(pdf_report.canvas, "Canvas", lambda *_a, **_k: output)
    result = {
        "disease": "parkinsons",
        "prediction": 1,
        "model_score": 0.9690051433877159,
        "decision_threshold": 0.5,
        "threshold_result": "at_or_above",
        "score_type": "uncalibrated_model_score",
        "model_identifier": "parkinsons:logistic_regression:43bfa333fd5f",
        "release_status": "experimental",
        "intended_use": "Educational research demonstration only.",
        "limitations": ["Pre-computed voice biomarkers; no raw audio."],
        "disclaimer": "Not a probability of disease or a diagnosis.",
    }

    generated = pdf_report.create_report("Parkinson's disease", {"PPE": 0.2}, result)
    text = " ".join(output.text)

    assert generated == b""
    assert "EXPERIMENTAL" in text
    assert "Model classification: threshold class 1" in text
    assert "Uncalibrated model score: 0.9690" in text
    assert "Decision threshold: 0.5000" in text
    assert "at or above the decision threshold" in text
    assert "not disease probability" in text
    assert result["model_identifier"] in text
    assert result["intended_use"] in text
    assert result["limitations"][0] in text
    assert result["disclaimer"] in text
    assert "RESEARCH-USE DISCLAIMER" in text
