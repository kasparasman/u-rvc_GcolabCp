"""
Module which defines the command-line interface for training voice
models using RVC.
"""

from __future__ import annotations

from typing import Annotated

import time
from multiprocessing import cpu_count
from pathlib import Path  # noqa: TC003

import typer
from rich import print as rprint
from rich.panel import Panel

from ultimate_rvc.cli.common import format_duration
from ultimate_rvc.core.train.prepare import populate_dataset as _populate_dataset
from ultimate_rvc.core.train.prepare import preprocess_dataset as _preprocess_dataset

app = typer.Typer(
    name="train",
    no_args_is_help=True,
    help="Train voice models using RVC",
    rich_markup_mode="markdown",
)

CORES = cpu_count()


@app.command(no_args_is_help=True)
def populate_dataset(
    name: Annotated[
        str,
        typer.Argument(help="The name of the dataset to populate."),
    ],
    audio_files: Annotated[
        list[Path],
        typer.Argument(
            help="The audio files to populate the dataset with.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
            file_okay=True,
        ),
    ],
) -> None:
    """
    Populate the dataset with the provided name with the provided audio
    files.
    """
    start_time = time.perf_counter()

    rprint()

    dataset_path = _populate_dataset(name, audio_files)

    rprint("[+] Dataset succesfully populated with the provided audio files!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
    rprint(Panel(f"[green]{dataset_path}", title="Dataset Path"))


@app.command(no_args_is_help=True)
def preprocess_dataset(
    model_name: Annotated[
        str,
        typer.Argument(help="The name of the model to train"),
    ],
    dataset: Annotated[
        Path,
        typer.Argument(
            help="The path to the dataset to preprocess",
            exists=True,
            resolve_path=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    sample_rate: Annotated[
        int,
        typer.Option(
            help="The target sample rate for the audio files in the provided dataset",
        ),
    ] = 40000,
    cpu_cores: Annotated[
        int,
        typer.Option(
            min=1,
            max=CORES,
            help="The number of CPU cores to use for preprocessing",
        ),
    ] = CORES,
    split_audio: Annotated[
        bool,
        typer.Option(
            help=(
                "Whether to split the audio files in the provided dataset into smaller"
                " segments before pre-processing. This help can improve the"
                " pre-processing speed for large audio files."
            ),
        ),
    ] = True,
    filter_audio: Annotated[
        bool,
        typer.Option(
            help=(
                "Whether to remove low-frequency sounds from the audio files in the"
                " provided dataset by applying a high-pass butterworth filter."
            ),
        ),
    ] = True,
    clean_audio: Annotated[
        bool,
        typer.Option(
            help=(
                "Whether to clean the audio files in the provided dataset using noise"
                " reduction algorithms."
            ),
        ),
    ] = False,
    clean_strength: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help=(
                "The intensity of the cleaning to apply to the audio files in the"
                " provided dataset."
            ),
        ),
    ] = 0.7,
) -> None:
    """Preprocess a dataset of audio files for training."""
    start_time = time.perf_counter()

    rprint()

    _preprocess_dataset(
        model_name,
        dataset,
        sample_rate,
        cpu_cores,
        split_audio,
        filter_audio,
        clean_audio,
        clean_strength,
    )

    rprint("[+] Dataset succesfully preprocessed!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
