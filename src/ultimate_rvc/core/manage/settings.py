"""Module which defines functions used for managing various settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import shutil

from ultimate_rvc.common import TEMP_DIR
from ultimate_rvc.core.common import display_progress

if TYPE_CHECKING:
    import gradio as gr


def delete_temp_files(
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> None:
    """

    Delete all temporary files.

    Parameters
    ----------
    progress_bar : gr.Progress, optional
        Progress bar to update.
    percentage : float, optional
        The percentage to display in the progress bar.

    """
    display_progress("[~] Deleting all temporary files...", percentage, progress_bar)
    if TEMP_DIR.is_dir():
        shutil.rmtree(TEMP_DIR)
