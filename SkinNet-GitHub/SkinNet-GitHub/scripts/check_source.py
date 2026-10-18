"""Static checks for the SkinNet research-code package."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "SkinNet.ipynb"
with NOTEBOOK.open(encoding="utf-8") as file:
    nb = json.load(file)
assert nb["nbformat"] == 4
sources = ["".join(cell.get("source", [])) for cell in nb["cells"] if cell["cell_type"] == "code"]
assert len(sources) == 44, f"Unexpected number of code cells: {len(sources)}"
for index, source in enumerate(sources):
    ast.parse(source, filename=f"cell {index}")
for file in [ROOT / "skinnet_experiments.py", ROOT / "src/skinnet_model.py",
             ROOT / "src/skinnet_worker.py", ROOT / "scripts/predict.py",
             ROOT / "scripts/check_release.py"]:
    ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
assert "model.load_weights" in (ROOT / "scripts/predict.py").read_text()
assert not any(cell.get("outputs") for cell in nb["cells"] if cell["cell_type"] == "code")
print("Source checks passed: notebook, model, worker, inference, and release check parse successfully.")
