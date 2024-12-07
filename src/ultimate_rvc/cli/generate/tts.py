"""
Module which defines the command-line interface for using RVC-based
TTS.
"""

from __future__ import annotations

from typing import Annotated

import asyncio
import time

from edge_tts import list_voices as _list_voices

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from ultimate_rvc.cli.common import format_duration
from ultimate_rvc.core.generate.tts import run_edge_tts as _run_edge_tts

app = typer.Typer(
    name="tts",
    no_args_is_help=True,
    help="Generate text from speech using RVC",
    rich_markup_mode="markdown",
)


@app.command(no_args_is_help=True)
def run_edge_tts(
    source: Annotated[
        str,
        typer.Argument(
            help="A string or path to a file containing the text to be converted.",
        ),
    ],
    voice: Annotated[
        str,
        typer.Option(
            help=(
                "The short name of the voice which should speak the provided text. Use"
                " the `list-voices` command to get a list of available voices."
            ),
        ),
    ] = "en-US-ChristopherNeural",
    pitch_shift: Annotated[
        int,
        typer.Option(
            help=(
                "The number of hertz to shift the pitch of the voice speaking the"
                " provided text."
            ),
        ),
    ] = 0,
    speed_change: Annotated[
        int,
        typer.Option(
            help=(
                "The absolute change to the speed of the voice speaking the provided"
                " text."
            ),
        ),
    ] = 0,
    volume_change: Annotated[
        int,
        typer.Option(
            help=(
                "The absolute change to the volume of the voice speaking the provided"
                " text."
            ),
        ),
    ] = 0,
) -> None:
    """Convert text to speech using Edge TTS."""
    start_time = time.perf_counter()

    rprint()
    rprint("[~] Converting text to speech using Edge TTS...")

    audio_path = asyncio.run(
        _run_edge_tts(
            source,
            voice,
            pitch_shift,
            speed_change,
            volume_change,
        ),
    )

    rprint("[+] Text successfully converted to speech!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
    rprint(Panel(f"[green]{audio_path}", title="Converted Speech Path"))


@app.command()
def list_voices(
    locale: Annotated[
        str | None,
        typer.Option(
            help="The locale to filter voices by.",
        ),
    ] = None,
    content_category: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "The content category to filter voices by. This option can be supplied"
                " multiple times to filter by multiple content categories."
            ),
        ),
    ] = None,
    voice_personality: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "The voice personality to filter voices by. This option can be supplied"
                " multiple times to filter by multiple voice personalities."
            ),
        ),
    ] = None,
    offset: Annotated[
        int,
        typer.Option(
            min=0,
            help="The offset to start listing voices from.",
        ),
    ] = 0,
    limit: Annotated[
        int,
        typer.Option(
            min=0,
            help="The limit on how many voices to list.",
        ),
    ] = 20,
    include_status_info: Annotated[
        bool,
        typer.Option(
            help="Include status information for each voice.",
        ),
    ] = False,
    include_codec_info: Annotated[
        bool,
        typer.Option(
            help="Include codec information for each voice.",
        ),
    ] = False,
) -> None:
    """List all available edge TTS voices."""
    start_time = time.perf_counter()
    rprint("[~] Retrieving information on all available edge TTS voices...")
    voices = asyncio.run(_list_voices())
    keys = [
        "Name",
        "FriendlyName",
        "ShortName",
        "Locale",
        "ContentCategories",
        "VoicePersonalities",
    ]
    filtered_voices = [
        v
        for v in voices
        if (
            (locale is None or locale in v["Locale"])
            and (
                content_category is None
                or any(
                    c in ", ".join(v["VoiceTag"]["ContentCategories"])
                    for c in content_category
                )
            )
            and (
                voice_personality is None
                or any(
                    p in ", ".join(v["VoiceTag"]["VoicePersonalities"])
                    for p in voice_personality
                )
            )
        )
    ]

    if include_status_info:
        keys.append("Status")
    if include_codec_info:
        keys.append("SuggestedCodec")

    table = Table()
    for key in keys:
        table.add_column(key)
    for voice in filtered_voices[offset : offset + limit]:

        values = [
            (
                ", ".join(voice["VoiceTag"][key])
                if key in {"ContentCategories", "VoicePersonalities"}
                else voice[key]
            )
            for key in keys
        ]

        table.add_row(*[f"[green]{value}" for value in values])

    rprint("[+] Information successfully retrieved!")
    rprint()

    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
    rprint(Panel(table, title="Available Edge TTS Voices"))
