"""
Module which defines the command-line interface for the Ultimate RVC
project.
"""

from __future__ import annotations

import typer

from ultimate_rvc.cli.generate.song_cover import app as song_cover_app
from ultimate_rvc.cli.generate.tts import app as tts_app

app = typer.Typer(
    name="urvc-cli",
    no_args_is_help=True,
    help="CLI for the Ultimate RVC project",
    rich_markup_mode="markdown",
)

app.add_typer(song_cover_app)
app.add_typer(tts_app)


if __name__ == "__main__":
    app()
