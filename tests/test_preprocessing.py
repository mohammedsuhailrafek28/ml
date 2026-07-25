from src.preprocessing.common import build_preprocessor
from src.utils.config import DISEASES
def test_preprocessor_has_transformers():
    assert build_preprocessor(DISEASES["diabetes"]).transformers
