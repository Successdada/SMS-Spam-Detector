# Pytest tests for src/spam_detector/prediction.py: model loading, output format, label/score consistency and input validation.
import pytest

from src.spam_detector.prediction import load_model, predict_message

SPAM_EXAMPLE = "WINNER! You have won a free prize. Call 09061701461 now to claim."
HAM_EXAMPLE = "Are you coming to dinner tonight?"


@pytest.fixture(scope="module")
def model():
    """Load the saved pipeline once for all tests in this file."""
    return load_model()


def test_loaded_model_has_prediction_methods(model):
    assert hasattr(model, "predict")
    assert hasattr(model, "decision_function")


def test_prediction_format(model):
    result = predict_message(SPAM_EXAMPLE, model)
    assert set(result.keys()) == {"label", "decision_score"}
    assert result["label"] in {"spam", "ham"}
    assert isinstance(result["decision_score"], float)


def test_clear_spam_and_ham_examples(model):
    assert predict_message(SPAM_EXAMPLE, model)["label"] == "spam"
    assert predict_message(HAM_EXAMPLE, model)["label"] == "ham"


def test_label_matches_score_sign(model):
    # LinearSVC predicts spam exactly when its decision score is above 0.
    for message in [SPAM_EXAMPLE, HAM_EXAMPLE, "ok", "Call 08001234567 now"]:
        result = predict_message(message, model)
        assert (result["label"] == "spam") == (result["decision_score"] > 0)


def test_same_message_gives_same_result(model):
    assert predict_message(SPAM_EXAMPLE, model) == predict_message(SPAM_EXAMPLE, model)


@pytest.mark.parametrize("bad_message", ["", "   ", "\n\t"])
def test_empty_or_whitespace_message_raises_value_error(model, bad_message):
    with pytest.raises(ValueError):
        predict_message(bad_message, model)


@pytest.mark.parametrize("bad_message", [None, 123, ["hello"]])
def test_non_string_message_raises_type_error(model, bad_message):
    with pytest.raises(TypeError):
        predict_message(bad_message, model)


def test_missing_model_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_model("models/this_file_does_not_exist.joblib")
        
        
# python -m pytest tests -v