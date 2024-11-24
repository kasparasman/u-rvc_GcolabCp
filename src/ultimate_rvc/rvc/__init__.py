"""
The rvc package is a collection of tools for voice cloning using the RVC
method.
"""

from __future__ import annotations

import static_ffmpeg

from ultimate_rvc.rvc.lib.tools.prerequisites_download import (
    prequisites_download_pipeline,
)

prequisites_download_pipeline(
    pretraineds_v1_f0=False,
    pretraineds_v1_nof0=False,
    pretraineds_v2_f0=False,
    pretraineds_v2_nof0=False,
    models=True,
    exe=False,
)
static_ffmpeg.add_paths()
