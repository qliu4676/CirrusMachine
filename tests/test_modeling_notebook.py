"""Progress remains readable in notebook viewers without widget rendering."""

import sys

import pytest


def test_progress_emits_text_in_a_jupyter_kernel(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    jupyter_client = pytest.importorskip("jupyter_client")
    pytest.importorskip("ipykernel")
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                """
from cirrusmachine.modeling.logging import ModelingProgress
with ModelingProgress(10, 0, 1, True) as progress:
    for step in range(1, 11):
        progress.update(step, 1 / step, 0.005)
"""
            )
        ]
    )
    manager = jupyter_client.KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ]
    executed = nbclient.NotebookClient(notebook, km=manager, timeout=60).execute(
        cleanup_kc=True
    )
    outputs = executed.cells[0].outputs
    mime = "application/vnd.jupyter.widget-view+json"
    assert not any(mime in output.get("data", {}) for output in outputs)
    assert any("100%" in output.get("text", "") for output in outputs)
