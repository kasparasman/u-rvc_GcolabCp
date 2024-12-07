"""
Module which defines functions and other definitions that facilitate
RVC-based TTS generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

import anyio

from edge_tts import Communicate

from ultimate_rvc.core.common import (
    TTS_AUDIO_BASE_DIR,
    display_progress,
    get_unique_base_path,
    json_dump,
)
from ultimate_rvc.core.exceptions import (
    Entity,
    NotProvidedError,
    UIMessage,
)
from ultimate_rvc.core.typing_extra import (
    EdgeTTSAudioMetaData,
)

if TYPE_CHECKING:
    import gradio as gr


async def run_edge_tts(
    source: str,
    voice: str,
    pitch_shift: int = 0,
    speed_change: int = 0,
    volume_change: int = 0,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Convert text to speech using edge TTS.

    Parameters
    ----------
    source : str
        A string or path to a file containing the text to be converted.

    voice : str
        The short name of the voice which should speak the provided
        text.

    pitch_shift : int, default=0
        The number of hertz to shift the pitch of the voice speaking
        the provided text.

    speed_change : int, default=0
        The absolute change to the speed of the voice speaking the
        provided text.

    volume_change : int, default=0
        The absolute change to the volume of the voice speaking the
        provided text.

    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The path to an audio track containing the spoken text.

    Raises
    ------
    NotProvidedError
        If no source is provided.

    """
    if not source:
        raise NotProvidedError(entity=Entity.SOURCE, ui_msg=UIMessage.NO_TTS_SOURCE)

    source_path = Path(source)
    source_is_file = source_path.is_file()
    if source_is_file:
        async with await anyio.open_file(source_path, "r", encoding="utf-8") as file:
            text = await file.read()
    else:
        text = source

    args_dict = EdgeTTSAudioMetaData(
        text=text,
        voice=voice,
        pitch_shift=pitch_shift,
        speed_change=speed_change,
        volume_change=volume_change,
    ).model_dump()
    TTS_AUDIO_BASE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        get_unique_base_path(
            TTS_AUDIO_BASE_DIR,
            "1_EdgeTTS_Audio",
            args_dict,
        ).with_suffix(suffix)
        for suffix in [".wav", ".json"]
    ]

    converted_audio_path, converted_audio_json_path = paths

    if not all(path.exists() for path in paths):
        display_progress(
            "[~] Converting text using Edge TTS...",
            percentage,
            progress_bar,
        )

        pitch_shift_str = f"{pitch_shift:+}Hz"
        speed_change_str = f"{speed_change:+}%"
        volume_change_str = f"{volume_change:+}%"

        communicate = Communicate(
            text,
            voice,
            pitch=pitch_shift_str,
            rate=speed_change_str,
            volume=volume_change_str,
        )

        await communicate.save(str(converted_audio_path))

        json_dump(args_dict, converted_audio_json_path)

    return converted_audio_path
