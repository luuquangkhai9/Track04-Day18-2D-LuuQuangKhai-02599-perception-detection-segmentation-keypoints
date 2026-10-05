"""Run the notebook's existing algorithm checks without model downloads or training.

Usage: .venv/Scripts/python.exe verify_lab.py
Requires the notebook's Ultralytics version plus nbformat.
"""
import ast
import json
import math
import sys
from pathlib import Path

import cv2
import nbformat
import numpy as np
import torch
import ultralytics
import yaml


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = Path(__file__).with_name("lab_2d_perception_student.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    cells = {c.id: c.source for c in notebook.cells}
    todo_ids = ["c016", "c018", "c020", "c032", "c040", "c042", "c059", "c066", "c079"]
    for cell in notebook.cells:
        if cell.cell_type == "code" and not cell.source.lstrip().startswith("%"):
            tree = ast.parse(cell.source)
            if cell.id in todo_ids:
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Assign, ast.Return, ast.Expr)):
                        value = getattr(node, "value", None)
                        assert not (isinstance(value, ast.Constant) and value.value is Ellipsis), cell.id
    assert "Lưu Quang Khải" in cells["student-identity"]
    assert "2A202602599" in cells["student-identity"]
    print("Notebook schema, Python syntax, identity and TODO placeholders: OK")

    namespace = {"torch": torch, "np": np, "cv2": cv2, "math": math, "Path": Path}
    # Display functions are not called by these checks; avoid requiring IPython locally.
    helper = ast.parse(cells["c005"])
    helper.body = [node for node in helper.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "IPython.display"
    )]
    exec(compile(helper, "notebook-helpers", "exec"), namespace)
    config_path = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "tiger-pose.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    namespace["TIGER_KPTS"] = config["kpt_names"][0]
    for cell_id in todo_ids:
        exec(compile(cells[cell_id], f"notebook-{cell_id}", "exec"), namespace)
    for section in ["1B", "2B", "3B", "3C", "4A"]:
        namespace["gate"](section)
    expected = [*namespace["CHECKS"], "average_precision"]
    assert all(namespace["PROGRESS"].get(name) == "ok" for name in expected), namespace["PROGRESS"]
    print("FLIP_IDX:", namespace["FLIP_IDX"])
    print("All 10 algorithm deliverables passed the original notebook checks.")
    print("Not executed: model inference, latency, SAM autolabel, GPU training and Q1–Q12.")


if __name__ == "__main__":
    main()
