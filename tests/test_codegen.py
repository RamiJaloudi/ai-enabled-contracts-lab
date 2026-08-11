import nbformat

from contractlab.codegen import generated_notebook, generated_script
from contractlab.config import TrainingConfig


def test_generated_script_contains_visible_configuration():
    config = TrainingConfig(run_name="learning-run", dataset_path="clauses.csv")
    script = generated_script(config)
    assert "run_training" in script
    assert "'run_name': 'learning-run'" in script
    assert "'freeze_encoder': False" in script
    compile(script, "generated.py", "exec")


def test_generated_notebook_is_valid():
    notebook = generated_notebook(TrainingConfig(dataset_path="clauses.csv"))
    nbformat.validate(notebook)
    assert len(notebook.cells) >= 5
    configuration_source = notebook.cells[2].source
    namespace = {}
    exec(configuration_source, namespace)
    assert namespace["config"].dataset_path == "clauses.csv"
