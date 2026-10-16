# --------------------------------------------------------------------------------------
# Project: OpenMicroManipulator
# License: MIT (see LICENSE file for full description)
#          All text in here must be included in any redistribution.
# Author:  M. S. (diffraction limited)
# --------------------------------------------------------------------------------------

import importlib.util
import os

from PySide6.QtUiTools import loadUiType


def load_ui_type(ui_path):
    """
    Returns the generated form class (Ui_...) for a Qt Designer .ui file.

    loadUiType() compiles the .ui file on every start, which is slow. If a precompiled
    'ui_<name>.py' exists next to the .ui file and is not older than it, that file is
    used instead. Create / update it with:  pyside6-uic <name>.ui -o ui_<name>.py
    Without it (or when it is outdated) the .ui file is compiled at runtime as before.
    """
    base_name = os.path.splitext(os.path.basename(ui_path))[0]
    module_name = f"ui_{base_name}"
    py_path = os.path.join(os.path.dirname(ui_path), module_name + ".py")

    try:
        if os.path.getmtime(py_path) >= os.path.getmtime(ui_path):
            spec = importlib.util.spec_from_file_location(module_name, py_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name in dir(module):
                if name.startswith("Ui_"):
                    return getattr(module, name)
    except Exception as exc:
        if os.path.exists(py_path):
            print(f"Could not use precompiled '{py_path}' ({exc}), compiling '{ui_path}' instead.")

    form_class, _ = loadUiType(ui_path)
    return form_class
