"""
Module which exposes functionality for creating and preprocessing
datasets for training voice conversion models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import shutil
from multiprocessing import cpu_count

from ultimate_rvc.common import lazy_import
from ultimate_rvc.core.common import (
    TRAINING_AUDIO_DIR,
    TRAINING_MODELS_DIR,
    display_progress,
    validate_exists,
)
from ultimate_rvc.core.exceptions import (
    Entity,
    InvalidAudioFormatError,
    NotProvidedError,
    UIMessage,
)
from ultimate_rvc.typing_extra import AudioExt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import gradio as gr

    import static_ffmpeg
    import static_sox

    from ultimate_rvc.typing_extra import StrPath
else:
    static_ffmpeg = lazy_import("static_ffmpeg")
    static_sox = lazy_import("static_sox")


def populate_dataset(
    name: str,
    audio_files: Sequence[StrPath],
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Populate the dataset with the provided name with the provided audio
    files.

    If no dataset with the provided name exists, a new dataset with the
    provided name will be created. If any of audio files already exist
    in the dataset, they will be overwritten.

    Parameters
    ----------
    name : str
        The name of the dataset to populate.
    audio_files : list[StrPath]
        The audio files to populate the dataset with.

    progress_bar : gr.Progress, optional
        The progress bar to update as the dataset is populated.
    percentage : float, optional
        The percentage to display on the progress bar.

    Returns
    -------
    The path to the dataset with the provided name.

    Raises
    ------
    NotProvidedError
        If no dataset name or no audio files are provided.

    InvalidAudioFormatError
        If any of the provided audio files are not in a valid format.

    """
    if not name:
        raise NotProvidedError(Entity.DATASET_NAME)

    if not audio_files:
        raise NotProvidedError(Entity.FILES, ui_msg=UIMessage.NO_UPLOADED_FILES)

    audio_paths: list[Path] = []
    for audio_file in audio_files:
        audio_path = validate_exists(audio_file, Entity.FILE)
        # TODO we should really use pydub.utils.mediainfo to check the
        # audio format instead of just checking the extension
        if audio_path.suffix.lstrip(".") not in set(AudioExt):
            raise InvalidAudioFormatError(
                audio_path,
                [e.value for e in AudioExt],
            )
        audio_paths.append(audio_path)

    display_progress(
        "[~] Populating dataset with provided audio files ",
        percentage,
        progress_bar,
    )

    dataset_path = TRAINING_AUDIO_DIR / name

    dataset_path.mkdir(parents=True, exist_ok=True)

    for audio_path in audio_paths:
        shutil.copyfile(audio_path, dataset_path / audio_path.name)

    return dataset_path


def preprocess_dataset(
    model_name: str,
    dataset: StrPath,
    sample_rate: int = 40000,
    cpu_cores: int = cpu_count(),
    split_audio: bool = True,
    filter_audio: bool = True,
    clean_audio: bool = False,
    clean_strength: float = 0.7,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> None:
    """
    Preprocess a dataset of audio files for training.

    Parameters
    ----------
    model_name : str
        The name of the model to train. If the model does not exist, a
        new folder will be created in the training models directory.
    dataset : StrPath
        The path to the dataset to preprocess.
    sample_rate : int, default=40000
        The target sample rate for the audio files in the provided
        dataset.
    cpu_cores : int, default=cpu_count()
        The number of CPU cores to use for preprocessing.
    split_audio : bool, default=True
        Whether to split the audio files in the provided dataset into
        smaller segments before pre-processing. This help can improve
        the pre-processing speed for large audio files.
    filter_audio : bool, default=True
        Whether to remove low-frequency sounds from the audio files in
        the provided dataset by applying a high-pass butterworth filter.
    clean_audio : bool, default=False
        Whether to clean the audio files in the provided dataset using
        noise reduction algorithms.
    clean_strength : float, default=0.7
        The intensity of the cleaning to apply to the audio files in the
        provided dataset.
    progress_bar : gr.Progress, optional
        The progress bar to update as the dataset is preprocessed.
    percentage : float, optional
        The percentage to display on the progress bar.


    Raises
    ------
    NotProvidedError
        If no model name or dataset is provided.

    """
    static_ffmpeg.add_paths()
    static_sox.add_paths()

    from ultimate_rvc.rvc.configs.config import Config  # noqa: PLC0415
    from ultimate_rvc.rvc.train.preprocess import (  # noqa: PLC0415
        preprocess as train_preprocess,
    )

    if not model_name:
        raise NotProvidedError(Entity.MODEL_NAME)

    dataset_path = validate_exists(dataset, Entity.DATASET)

    display_progress(
        "[~] preprocessing dataset for training ",
        percentage,
        progress_bar,
    )
    training_model_directory = TRAINING_MODELS_DIR / model_name
    training_model_directory.mkdir(parents=True, exist_ok=True)
    config = Config()
    split_percentage = 3.0 if config.is_half else 3.7
    train_preprocess.preprocess_training_set(
        str(dataset_path),
        sample_rate,
        cpu_cores,
        str(training_model_directory),
        split_percentage,
        split_audio,
        filter_audio,
        clean_audio,
        clean_strength,
    )
