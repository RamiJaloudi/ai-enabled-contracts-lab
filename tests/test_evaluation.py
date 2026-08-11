from contractlab.evaluation import classification_metrics, confusion_records


def test_metrics_are_numeric_and_confusion_is_sparse():
    labels = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]
    metrics = classification_metrics(labels, predictions)
    assert 0 < metrics["macro_f1"] < 1
    records = confusion_records(labels, predictions, ["A", "B"])
    assert len(records) == 3
    assert sum(record["count"] for record in records) == 4
