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
from rich.table import Table

from ultimate_rvc.cli.common import (
    complete_audio_split_method,
    complete_embedder_model,
    complete_f0_method,
    complete_rvc_version,
    complete_training_sample_rate,
    format_duration,
)
from ultimate_rvc.core.train.common import get_gpu_info as _get_gpu_info
from ultimate_rvc.core.train.extract import extract_features as _extract_features
from ultimate_rvc.core.train.prepare import populate_dataset as _populate_dataset
from ultimate_rvc.core.train.prepare import preprocess_dataset as _preprocess_dataset
from ultimate_rvc.typing_extra import (
    AudioSplitMethod,
    EmbedderModel,
    RVCVersion,
    TrainingF0Method,
    TrainingSampleRate,
)

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

    dataset_path = _populate_dataset(name, audio_files)

    rprint("[+] Dataset succesfully populated with the provided audio files!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
    rprint(Panel(f"[green]{dataset_path}", title="Dataset Path"))


@app.command(no_args_is_help=True)
def preprocess_dataset(
    model_name: Annotated[
        str,
        typer.Argument(
            help=(
                "The name of the model to train. If the model does not exist, it will"
                " be created."
            ),
        ),
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
        TrainingSampleRate,
        typer.Option(
            autocompletion=complete_training_sample_rate,
            help="The target sample rate for the audio files in the provided dataset",
        ),
    ] = TrainingSampleRate.HZ_40K,
    cpu_cores: Annotated[
        int,
        typer.Option(
            min=1,
            max=CORES,
            help="The number of CPU cores to use for preprocessing",
        ),
    ] = CORES,
    split_method: Annotated[
        AudioSplitMethod,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_audio_split_method,
            help=(
                "The method to use for splitting the audio files in the provided"
                " dataset. Use the `Skip` method to skip splitting if the audio files"
                " are already split. Use the `Simple` method if excessive silence has"
                " already been removed from the audio files. Use the `Automatic` method"
                " for automatic silence detection and splitting around it."
            ),
        ),
    ] = AudioSplitMethod.AUTOMATIC,
    chunk_len: Annotated[
        float,
        typer.Option(
            min=0.5,
            max=5.0,
            help="Length of split audio chunks when using the `Simple` split method.",
        ),
    ] = 3.0,
    overlap_len: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=0.4,
            help=(
                "Length of overlap between split audio chunks when using the `Simple`"
                " split method."
            ),
        ),
    ] = 0.3,
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
    """
    Preprocess a dataset of audio files for training a given
    model.
    """
    start_time = time.perf_counter()

    _preprocess_dataset(
        model_name,
        dataset,
        sample_rate,
        cpu_cores,
        split_method,
        chunk_len,
        overlap_len,
        filter_audio,
        clean_audio,
        clean_strength,
    )

    rprint("[+] Dataset succesfully preprocessed!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))


@app.command()
def get_gpu_information() -> None:
    """Retrieve information on locally available GPUs."""
    start_time = time.perf_counter()
    rprint("[+] Retrieving GPU Information...")
    gpu_infos = _get_gpu_info()

    rprint("[+] GPU Information successfully retrieved!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))

    table = Table()
    table.add_column("Name", style="green")
    table.add_column("Index", style="green")

    for gpu_name, gpu_index in gpu_infos:
        table.add_row(gpu_name, str(gpu_index))

    rprint(table)


@app.command(no_args_is_help=True)
def extract_features(
    model_name: Annotated[
        str,
        typer.Argument(help="The name of the model to be trained."),
    ],
    rvc_version: Annotated[
        RVCVersion,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_rvc_version,
            help="Version of RVC to use for training the model.",
        ),
    ] = RVCVersion.V2,
    f0_method: Annotated[
        TrainingF0Method,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_f0_method,
            help="The method to use for extracting pitch features.",
        ),
    ] = TrainingF0Method.RMVPE,
    hop_length: Annotated[
        int,
        typer.Option(
            min=1,
            max=512,
            help=(
                "The hop length to use for extracting pitch features. Only used"
                " with the CREPE pitch extraction method."
            ),
        ),
    ] = 128,
    cpu_cores: Annotated[
        int,
        typer.Option(
            help="The number of CPU cores to use for feature extraction.",
            min=1,
            max=cpu_count(),
        ),
    ] = cpu_count(),
    gpus: Annotated[
        list[int] | None,
        typer.Option(
            min=0,
            help="The device ids of the GPUs to use for extracting audio embeddings.",
        ),
    ] = None,
    sample_rate: Annotated[
        TrainingSampleRate,
        typer.Option(
            help=(
                "The sample rate of the audio files in the preprocessed dataset"
                " associated with the model to be trained."
            ),
            min=1,
        ),
    ] = TrainingSampleRate.HZ_40K,
    embedder_model: Annotated[
        EmbedderModel,
        typer.Option(
            autocompletion=complete_embedder_model,
            help="The model to use for extracting audio embeddings.",
            case_sensitive=False,
        ),
    ] = EmbedderModel.CONTENTVEC,
    custom_embedder_model: Annotated[
        str | None,
        typer.Option(
            exists=True,
            resolve_path=True,
            dir_okay=True,
            file_okay=False,
            help="The path to a custom model to use for extracting audio embeddings.",
        ),
    ] = None,
    include_mutes: Annotated[
        int,
        typer.Option(
            help=(
                "The number of mute audio files to include in the generated"
                " training file list. Adding silent files enables the model to handle"
                " pure silence in inferred audio files. If the preprocessed audio"
                " dataset already contains segments of pure silence, set this to 0."
            ),
            min=0,
            max=10,
        ),
    ] = 2,
) -> None:
    """
    Extract features from the preprocessed dataset associated with a
    model to be trained.
    """
    start_time = time.perf_counter()

    gpu_set = set(gpus) if gpus is not None else None
    _extract_features(
        model_name,
        rvc_version,
        f0_method,
        hop_length,
        cpu_cores,
        gpu_set,
        sample_rate,
        embedder_model,
        custom_embedder_model,
        include_mutes,
    )

    rprint("[+] Dataset features succesfully extracted!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
