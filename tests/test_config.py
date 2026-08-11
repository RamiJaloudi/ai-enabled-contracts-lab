from contractlab.config import TrainingConfig


def test_config_round_trip(tmp_path):
    original = TrainingConfig(run_name="A useful run", dataset_path="data.csv", epochs=2)
    path = original.save(tmp_path / "config.json")
    loaded = TrainingConfig.load(path)
    assert loaded == original
    assert loaded.safe_run_name == "A-useful-run"


def test_config_validation_catches_invalid_values():
    config = TrainingConfig(
        epochs=0, learning_rate=0, train_batch_size=0, precision="magic", split_strategy="magic"
    )
    errors = config.validate()
    assert len(errors) >= 4
    assert any("Split strategy" in error for error in errors)


def test_config_validation_catches_invalid_overfitting_safeguards():
    config = TrainingConfig(
        early_stopping_patience=-1,
        early_stopping_min_delta=1.0,
        overfitting_warning_gap=0.0,
        overfitting_loss_ratio=1.0,
    )

    errors = config.validate()

    assert any("patience" in error for error in errors)
    assert any("minimum delta" in error for error in errors)
    assert any("warning gap" in error for error in errors)
    assert any("loss ratio" in error for error in errors)
