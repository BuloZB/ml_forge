"""
shortcuts.py
Keyboard shortcuts and related helper functions.
"""

import dearpygui.dearpygui as dpg

def shortcuts() -> None:
    if dpg.is_key_pressed(dpg.mvKey_Delete):
        delete_selected_nodes()

    if dpg.is_key_down(dpg.mvKey_LControl):
        if dpg.is_key_pressed(dpg.mvKey_Back):
            delete_selected_nodes()
        if dpg.is_key_pressed(dpg.mvKey_S):
            from ml_forge.filesystem.save import save_current
            save_current()
        if dpg.is_key_pressed(dpg.mvKey_Z):
            from ml_forge.graph.undo import undo
            undo()
        if dpg.is_key_pressed(dpg.mvKey_Y):
            from ml_forge.graph.undo import redo
            redo()