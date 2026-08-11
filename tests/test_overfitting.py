from contractlab.overfitting import EarlyStoppingMonitor, overfitting_diagnostics


def _history(train_f1: float, validation_f1: float, train_loss: float, validation_loss: float):
    return [
        {
            "epoch": 1,
            "train_macro_f1": train_f1,
            "validation_macro_f1": validation_f1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
    ]


def test_early_stopping_requires_configured_non_improving_epochs():
    monitor = EarlyStoppingMonitor(patience=2, min_delta=0.01)

    assert monitor.observe(0.60, 1) == (True, False)
    assert monitor.observe(0.605, 2) == (False, False)
    assert monitor.observe(0.59, 3) == (False, True)
    assert monitor.best_epoch == 1


def test_overfitting_diagnostics_pass_balanced_train_validation_metrics():
    result = overfitting_diagnostics(
        _history(0.82, 0.78, 0.42, 0.50),
        generalization_gap_threshold=0.15,
        loss_ratio_threshold=1.5,
    )

    assert result["status"] == "passed"
    assert result["overfitting_detected"] is False
    assert result["release_gate_passed"] is True


def test_overfitting_diagnostics_fail_large_f1_and_loss_gaps():
    result = overfitting_diagnostics(
        _history(0.98, 0.62, 0.15, 0.55),
        generalization_gap_threshold=0.15,
        loss_ratio_threshold=1.5,
    )

    assert result["status"] == "warning"
    assert result["overfitting_detected"] is True
    assert result["release_gate_passed"] is False
    assert len(result["reasons"]) == 2


def test_overfitting_diagnostics_do_not_pass_without_comparable_metrics():
    result = overfitting_diagnostics(
        [{"epoch": 1, "validation_macro_f1": 0.8}],
        generalization_gap_threshold=0.15,
        loss_ratio_threshold=1.5,
    )

    assert result["status"] == "not_available"
    assert result["release_gate_passed"] is False
