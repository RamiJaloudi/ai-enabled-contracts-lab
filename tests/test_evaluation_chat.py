from contractlab.evaluation_chat import (
    answer_metrics_question,
    evaluation_diagnosis,
    prediction_markdown,
)

SUMMARY = {
    "test_metrics": {
        "accuracy": 0.75,
        "macro_precision": 0.70,
        "macro_recall": 0.68,
        "macro_f1": 0.69,
    },
    "split_rows": {"test": 24},
    "label_names": ["Termination", "Renewal"],
    "classification_report": {
        "Termination": {"f1-score": 0.82},
        "Renewal": {"f1-score": 0.56},
    },
}


def test_metric_questions_are_answered_from_saved_summary():
    assert "0.690" in answer_metrics_question("What was macro F1?", SUMMARY)
    assert "24" in answer_metrics_question("How many held-out examples?", SUMMARY)
    assert "Renewal" in answer_metrics_question("Which label was weakest?", SUMMARY)
    assert "Termination" in answer_metrics_question("Which label was strongest?", SUMMARY)


def test_non_metric_prompt_is_left_for_checkpoint_classification():
    assert answer_metrics_question("Either party may terminate for breach.", SUMMARY) is None


def test_overfitting_questions_require_saved_generalization_evidence():
    assert "not been verified" in answer_metrics_question("Is this overfitting?", SUMMARY)
    verified = {
        **SUMMARY,
        "overfitting_diagnostics": {
            "status": "passed",
            "train_macro_f1": 0.82,
            "validation_macro_f1": 0.78,
            "generalization_gap": 0.04,
        },
    }
    assert "0.040" in answer_metrics_question("What is the generalization gap?", verified)
    failed = {
        **SUMMARY,
        "overfitting_diagnostics": {
            "status": "warning",
            "reasons": ["Train macro-F1 gap is too large."],
        },
    }
    assert "not release-ready" in answer_metrics_question("Is it release-ready?", failed)


def test_diagnosis_explains_accuracy_macro_f1_gap():
    reply = evaluation_diagnosis(SUMMARY)
    assert "Evaluation diagnosis" in reply
    assert "0.060 gap" in reply
    assert "What to try next" in reply
    assert "Contracts-BERT" in reply


def test_prediction_reply_is_transparent_about_classifier_scope():
    reply = prediction_markdown(
        [
            {"label": "Termination", "confidence": 0.8},
            {"label": "Renewal", "confidence": 0.15},
        ]
    )
    assert "Termination" in reply
    assert "80.0%" in reply
    assert "not a legal interpretation" in reply
