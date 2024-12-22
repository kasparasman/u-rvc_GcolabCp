"""
Module which defines functions for initializing the core of the Ultimate
RVC project.
"""

from __future__ import annotations

from pathlib import Path

from rich import print as rprint

from ultimate_rvc.common import VOICE_MODELS_DIR
from ultimate_rvc.core.common import FLAG_FILE
from ultimate_rvc.core.generate.song_cover import initialize_audio_separator
from ultimate_rvc.core.manage.models import download_voice_model
from ultimate_rvc.rvc.lib.tools.prerequisites_download import (
    prequisites_download_pipeline,
)


def download_sample_models() -> None:
    """Download sample RVC models."""
    named_model_links = [
        (
            "https://huggingface.co/damnedraxx/TaylorSwift/resolve/main/TaylorSwift.zip",
            "Taylor Swift",
        ),
        (
            "https://huggingface.co/Vermiculos/balladjames/resolve/main/Ballad%20James.zip?download=true",
            "James Hetfield",
        ),
        ("https://huggingface.co/ryolez/MMLP/resolve/main/MMLP.zip", "Eminem"),
    ]
    for model_url, model_name in named_model_links:
        if not Path(VOICE_MODELS_DIR / model_name).is_dir():
            rprint(f"Downloading {model_name}...")
            try:
                download_voice_model(model_url, model_name)
            except Exception as e:
                rprint(f"Failed to download {model_name}: {e}")


def initialize() -> None:
    """Initialize the Ultimate RVC project."""
    prequisites_download_pipeline(
        pretraineds_v1_f0=False,
        pretraineds_v1_nof0=False,
        pretraineds_v2_f0=False,
        pretraineds_v2_nof0=False,
        models=True,
        exe=False,
    )
    if not FLAG_FILE.is_file():
        download_sample_models()
        initialize_audio_separator()
        FLAG_FILE.touch()


if __name__ == "__main__":
    initialize()
