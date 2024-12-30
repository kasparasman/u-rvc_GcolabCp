"""
Module which exposes functionality for training voice conversion
models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

from ultimate_rvc.core.common import display_progress, validate_model_exists
from ultimate_rvc.core.exceptions import DatasetFileListNotFoundError, Entity
from ultimate_rvc.typing_extra import (
    IndexAlgorithm,
    RVCVersion,
    TrainingSampleRate,
    Vocoder,
)

if TYPE_CHECKING:
    import gradio as gr


def run_training(
    model_name: str,
    sample_rate: TrainingSampleRate = TrainingSampleRate.HZ_40K,
    rvc_version: RVCVersion = RVCVersion.V2,
    vocoder: Vocoder = Vocoder.HIFI_GAN,
    index_algorithm: IndexAlgorithm = IndexAlgorithm.AUTO,
    num_epochs: int = 500,
    batch_size: int = 8,
    detect_overtraining: bool = False,
    overtraining_threshold: int = 50,
    save_interval: int = 10,
    save_all_checkpoints: bool = False,
    save_all_weights: bool = True,
    clear_saved_data: bool = False,
    use_pretrained: bool = True,
    use_custom_pretrained: bool = False,
    generator_name: str | None = None,
    discriminator_name: str | None = None,
    preload_dataset: bool = False,
    enable_checkpointing: bool = False,
    gpus: set[int] | None = None,
    progress_bar: gr.Progress | None = None,
    percentage: tuple[float, float] = (0.0, 0.5),
) -> tuple[Path, Path]:
    """

    Train a voice model using its associated preprocessed dataset and
    extracted features.

    Parameters
    ----------
    model_name : str
        The name of the model to train.
    sample_rate : TrainingSampleRate, default=TrainingSampleRate.HZ_40K
        The sample rate of the audio files in the preprocessed dataset
        associated with the model.
    rvc_version : RVCVersion, default=RVCVersion.V2
        The version of RVC architecture that the model should use.
    vocoder : Vocoder, default=Vocoder.HIFI_GAN
        The vocoder to use for audio synthesis during training. HiFi-GAN
        provides basic audio fidelity, while RefineGAN provides the
        highest audio fidelity.
    index_algorithm : IndexAlgorithm, default=IndexAlgorithm.AUTO
        The method to use for generating an index file for the trained
        model. KMeans is a clustering algorithm that divides the dataset
        into K clusters. This setting is particularly useful for large
        datasets.
    num_epochs : int, default=500
        The number of epochs to train the model.
    batch_size : int, default=8
        The number of samples to include in each training batch. It is
        advisable to align this value with the available VRAM of your
        GPU. A setting of 4 offers improved accuracy but slower
        processing, while 8 provides faster and standard results.
    detect_overtraining : bool, default=False
        Whether to detect overtraining to prevent the model from
        learning the training data too well and losing the ability to
        generalize to new data.
    overtraining_threshold : int, default=50
        The maximum number of epochs to continue training without any
        observed improvement in model performance.
    save_interval : int, default=10
        The interval at which to save model weights and checkpoints
        during training, measured in epochs.
    save_all_checkpoints : bool, default=False
        Whether to save separate model checkpoints for each epoch or
        only save a checkpoint for the last epoch.
    save_all_weights : bool, default=True
        Whether to save separate model weights for each epoch or only
        save the best model weights.
    clear_saved_data : bool, default=False
        Whether to delete any existing saved training data associated
        with the model before starting a new training session. Enable
        this setting only if you are training a new model from scratch
        or restarting training.
    use_pretrained : bool, default=True
        Whether to use pretrained generator and discriminator models for
        training. This reduces training time and improves overall model
        performance.
    use_custom_pretrained : bool, default=False
        Whether to use custom pretrained generator and discriminator
        models for training. This can lead to superior results, as
        selecting the most suitable pretrained models tailored to the
        specific use case can significantly enhance performance.
    generator_name : StrPath, optional
        The name of a custom pretrained generator model to use for
        training. This is only used if `use_custom_pretrained` is set to
        True.
    discriminator_name : StrPath, optional
        The name of a custom pretrained discriminator model to use for
        training. This is only used if `use_custom_pretrained` is set to
        True.
    preload_dataset : bool, default=False
        Whether to preload all training data into GPU memory. This can
        improve training speed but requires a lot of VRAM.
    enable_checkpointing : bool, default=False
        Whether to enable memory-efficient training . This reduces VRAM
        usage at the cost of slower training speed. It is useful for
        GPUs with limited memory (e.g., <6GB VRAM) or when training with
        a batch size larger than what your GPU can normally accommodate.
    gpus : set[int], optional
        The device ids of the GPUs to use for training. If None, only
        CPU will be used.
    progress_bar : gr.Progress, optional
        The progress bar to display during training.
    percentage : tuple[float, float], default=(0.0, 0.5)
        The percentage of the progress bar to display during training.

    Returns
    -------
    model_file : Path
        The path to the trained voice model file.
    index_file : Path
        The path to the index file generated during training.

    Raises
    ------
    DatasetFileListNotFoundError
        If no dataset file list is associated with the model.

    """
    model_path = validate_model_exists(model_name, Entity.TRAINING_MODEL)
    filelist_path = model_path / "filelist.txt"
    if not filelist_path.exists():
        raise DatasetFileListNotFoundError(model_name)
    pg, pd = "", ""
    if use_pretrained:
        if use_custom_pretrained:
            pg = str(validate_model_exists(generator_name, Entity.CUSTOM_GENERATOR))
            pd = str(
                validate_model_exists(discriminator_name, Entity.CUSTOM_DISCRIMINATOR),
            )
        else:
            from ultimate_rvc.rvc.lib.tools.pretrained_selector import (  # noqa: PLC0415
                pretrained_selector,
            )

            pg, pd = pretrained_selector(
                rvc_version,
                vocoder,
                pitch_guidance=True,
                sample_rate=int(sample_rate),
            )

    from ultimate_rvc.rvc.train.train import main as train_main  # noqa: PLC0415

    display_progress("[~] training voice model...", percentage[0], progress_bar)
    train_main(
        model_name,
        int(sample_rate),
        rvc_version,
        vocoder,
        num_epochs,
        batch_size,
        save_interval,
        save_all_checkpoints,
        save_all_weights,
        pg,
        pd,
        detect_overtraining,
        overtraining_threshold,
        clear_saved_data,
        preload_dataset,
        enable_checkpointing,
        gpus if gpus is not None else {0},
    )
    display_progress(
        "[~] Generating index file for trained voice model",
        percentage[1],
        progress_bar,
    )
    from ultimate_rvc.rvc.train.process.extract_index import (  # noqa: PLC0415
        main as extract_index_main,
    )

    extract_index_main(str(model_path), rvc_version, index_algorithm)

    # TODO should return the model and index file paths
    return Path(), Path()
