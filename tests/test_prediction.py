from src.prediction.input_validation import validate_input
import pytest
def test_invalid_input_rejected():
    with pytest.raises(ValueError): validate_input({}, ("age",))
