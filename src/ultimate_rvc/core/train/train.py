"""
Module which exposes functionality for training voice conversion
models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

from ultimate_rvc.core.common import (
    VOICE_MODELS_DIR,
    copy_files_to_new_dir,
    display_progress,
    validate_model_exists,
)
from ultimate_rvc.core.exceptions import (
    Entity,
    ModelAsssociatedEntityNotFoundError,
    ModelExistsError,
    NotProvidedError,
    Step,
)
from ultimate_rvc.core.train.common import validate_devices
from ultimate_rvc.typing_extra import DeviceType, IndexAlgorithm, Vocoder

if TYPE_CHECKING:
    from pathlib import Path

    import gradio as gr


def run_training(
    model_name: str,
    num_epochs: int = 500,
    batch_size: int = 8,
    detect_overtraining: bool = False,
    overtraining_threshold: int = 50,
    vocoder: Vocoder = Vocoder.HIFI_GAN,
    index_algorithm: IndexAlgorithm = IndexAlgorithm.AUTO,
    finetune_pretrained: bool = True,
    custom_pretrained: str | None = None,
    save_interval: int = 10,
    save_all_checkpoints: bool = False,
    save_all_weights: bool = False,
    clear_saved_data: bool = False,
    upload_model: bool = False,
    upload_name: str | None = None,
    hardware_acceleration: DeviceType = DeviceType.AUTOMATIC,
    gpu_ids: set[int] | None = None,
    preload_dataset: bool = False,
    reduce_memory_usage: bool = False,
    progress_bar: gr.Progress | None = None,
    percentage: tuple[float, float] = (0.0, 0.5),
) -> tuple[Path, Path]:
    """

    Train a voice model using its associated preprocessed dataset and
    extracted features.

    Parameters
    ----------
    model_name : str
        The name of the voice model to train.
    num_epochs : int, default=500
        The number of epochs to train the voice model. A higher number
        can improve voice model performance but may lead to
        overtraining.
    batch_size : int, default=8
        The number of samples to include in each training batch. It is
        advisable to align this value with the available VRAM of your
        GPU. A setting of 4 offers improved accuracy but slower
        processing, while 8 provides faster and standard results.
    detect_overtraining : bool, default=False
        Whether to detect overtraining to prevent the voice model from
        learning the training data too well and losing the ability to
        generalize to new data.
    overtraining_threshold : int, default=50
        The maximum number of epochs to continue training without any
        observed improvement in voice model performance.
    vocoder : Vocoder, default=Vocoder.HIFI_GAN
        The vocoder to use for audio synthesis during training. HiFi-GAN
        provides basic audio fidelity, while RefineGAN provides the
        highest audio fidelity.
    index_algorithm : IndexAlgorithm, default=IndexAlgorithm.AUTO
        The method to use for generating an index file for the trained
        voice model. KMeans is particularly useful for large datasets.
    finetune_pretrained : bool, default=True
        Whether to finetune a pretrained model rather than train a
        new voice model from scratch. This can reduce training time and
        improve overall voice model performance.
    custom_pretrained: str, optional
        The name of a custom pretrained model to finetune. Finetuning a
        custom pretrained model can lead to superior results, as
        selecting the most suitable pretrained model tailored to the
        specific use case can significantly enhance performance.
        If none is provided, a default pretrained model is used.
    save_interval : int, default=10
        The epoch interval at which to to save voice model weights and
        checkpoints. The best model weights are always saved regardless
        of this setting.
    save_all_checkpoints : bool, default=False
        Whether to save a unique checkpoint at each save interval. If
        not enabled, only the latest checkpoint will be saved at each
        interval.
    save_all_weights : bool, default=False
        Whether to save unique voice model weights at each save
        interval. If not enabled, only the best voice model weights will
        be saved.
    clear_saved_data : bool, default=False
        Whether to delete any existing training data associated
        with the voice model before training commences. Enable this
        setting only if you are training a new voice model from scratch
        or restarting training.
    upload_model : bool, default=False
        Whether to automatically upload the trained voice model so that
        it can be used for audio generation tasks within the Ultimate
        RVC app.
    upload_name : str, optional
        The name to give the uploaded voice model.
    hardware_acceleration : DeviceType, default=DeviceType.AUTOMATIC
        The type of hardware acceleration to use when training the voice
        model. `AUTOMATIC` will select the first available GPU and fall
        back to CPU if no GPUs are available.
    gpu_ids : set[int], optional
        Set of ids of the GPUs to use for training the voice model when
        `GPU` is selected for hardware acceleration.
    preload_dataset : bool, default=False
        Whether to preload all training data into GPU memory. This can
        improve training speed but requires a lot of VRAM.
    reduce_memory_usage : bool, default=False
        Whether to reduce VRAM usage at the cost of slower training
        speed by enabling activation checkpointing. This is useful for
        GPUs with limited memory (e.g., <6GB VRAM) or when training with
        a batch size larger than what your GPU can normally accommodate.
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
    ModelAsssociatedEntityNotFoundError
        If the voice model toe be trained does not have an associated
        dataset file list or if a custom pretrained
        generator/discriminator model does not have an associated
        generator or discriminator.
    NotProvidedError
        If an upload name is not provided when the upload parameter is
        set
    ModelExistsError
        If a voice with the provided upload name already exists when the
        upload parameter is set

    """
    model_path = validate_model_exists(model_name, Entity.TRAINING_MODEL)
    filelist_path = model_path / "filelist.txt"
    if not filelist_path.exists():
        raise ModelAsssociatedEntityNotFoundError(
            Entity.DATASET_FILE_LIST,
            model_name,
            Step.FEATURE_EXTRACTION,
        )
    upload_model_path = None
    if upload_model:
        if not upload_name:
            raise NotProvidedError(Entity.UPLOAD_NAME)
        upload_model_path = VOICE_MODELS_DIR / upload_name.strip()
        if upload_model_path.exists():
            raise ModelExistsError(Entity.VOICE_MODEL, upload_name)

    with (model_path / "model_info.json").open("r") as f:
        model_info = json.load(f)

    rvc_version: str = model_info["rvc_version"]
    sample_rate: int = model_info["sample_rate"]

    pg, pd = "", ""
    if finetune_pretrained:
        if custom_pretrained is not None:
            custom_pretrained_path = validate_model_exists(
                custom_pretrained,
                Entity.CUSTOM_PRETRAINED_MODEL,
            )
            generator_path = next(custom_pretrained_path.glob("G*.pth"), None)
            if generator_path is None:
                raise ModelAsssociatedEntityNotFoundError(
                    Entity.GENERATOR,
                    custom_pretrained,
                )
            discriminator_path = next(custom_pretrained_path.glob("D*.pth"), None)
            if discriminator_path is None:
                raise ModelAsssociatedEntityNotFoundError(
                    Entity.DISCRIMINATOR,
                    custom_pretrained,
                )
            pg, pd = str(generator_path), str(discriminator_path)

        else:
            from ultimate_rvc.rvc.lib.tools.pretrained_selector import (  # noqa: PLC0415
                pretrained_selector,
            )

            pg, pd = pretrained_selector(
                rvc_version,
                pitch_guidance=True,
                sample_rate=sample_rate,
            )

    display_progress("[~] training voice model...", percentage[0], progress_bar)
    from ultimate_rvc.rvc.train.train import main as train_main  # noqa: PLC0415

    device_type, device_ids = validate_devices(hardware_acceleration, gpu_ids)

    train_main(
        model_name,
        sample_rate,
        rvc_version,
        vocoder,
        num_epochs,
        batch_size,
        save_interval,
        not save_all_checkpoints,
        save_all_weights,
        pg,
        pd,
        detect_overtraining,
        overtraining_threshold,
        clear_saved_data,
        preload_dataset,
        reduce_memory_usage,
        device_type,
        device_ids,
    )

    display_progress(
        "[~] Generating index file for trained voice model...",
        percentage[1],
        progress_bar,
    )
    from ultimate_rvc.rvc.train.process.extract_index import (  # noqa: PLC0415
        main as extract_index_main,
    )

    extract_index_main(str(model_path), rvc_version, index_algorithm)

    model_file = model_path / f"{model_name}_best.pth"
    index_file = model_path / f"added_{model_name}_{rvc_version}.index"
    if upload_model_path and model_file.is_file() and index_file.is_file():
        copy_files_to_new_dir([index_file, model_file], upload_model_path)
    return model_file, index_file
