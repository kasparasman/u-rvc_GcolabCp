"""
Module which exposes functionality for extracting training features from
audio datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from multiprocessing import cpu_count

from ultimate_rvc.common import lazy_import
from ultimate_rvc.core.common import (
    display_progress,
    get_combined_file_hash,
    validate_model_exists,
)
from ultimate_rvc.core.exceptions import Entity, PreprocessedAudioNotFoundError
from ultimate_rvc.typing_extra import (
    EmbedderModel,
    RVCVersion,
    TrainingF0Method,
    TrainingSampleRate,
)

if TYPE_CHECKING:
    import gradio as gr

    from ultimate_rvc.rvc.train.extract import preparing_files
else:
    preparing_files = lazy_import("ultimate_rvc.rvc.train.extract.preparing_files")


def extract_features(
    model_name: str,
    rvc_version: RVCVersion = RVCVersion.V2,
    f0_method: TrainingF0Method = TrainingF0Method.RMVPE,
    hop_length: int = 128,
    cpu_cores: int = cpu_count(),
    gpus: set[int] | None = None,
    sample_rate: TrainingSampleRate = TrainingSampleRate.HZ_40K,
    embedder_model: EmbedderModel = EmbedderModel.CONTENTVEC,
    custom_embedder_model: str | None = None,
    include_mutes: int = 2,
    progress_bar: gr.Progress | None = None,
    percentage: tuple[float, float] = (0.0, 0.5),
) -> None:
    """
    Extract features from the preprocessed dataset associated with a
    model to be trained.

    Parameters
    ----------
    model_name : str
        The name of the model to be trained.
    rvc_version : RVCVersion, default=RVCVersion.V2
        Version of RVC to use for training the model.
    f0_method : TrainingF0Method, default=TrainingF0Method.RMVPE
        The method to use for extracting pitch features.
    hop_length : int, default=128
        The hop length to use for extracting pitch features. Only used
        with the CREPE pitch extraction method.
    cpu_cores : int, default=cpu_count()
        The number of CPU cores to use for feature extraction.
    gpus : set[int], optional
        The device ids of the GPUs to use for feature extraction.
        If None, only CPU will be used.
    sample_rate : TrainingSampleRate, default=TrainingSampleRate.HZ_40K
        The sample rate of the audio files in the preprocessed
        dataset associated with the model to be trained.
    embedder_model : EmbedderModel, default=EmbedderModel.CONTENTVEC
        The model to use for extracting audio embeddings.
    custom_embedder_model : StrPath, optional
        The name of the custom embedder model to use for extracting
        audio embeddings.
    include_mutes : int, default=2
        The number of mute audio files to include in the generated
        training file list. Adding silent files enables the model to
        handle pure silence in inferred audio files. If the preprocessed
        audio dataset already contains segments of pure silence, set
        this to 0.
    progress_bar : gr.Progress, optional
        The progress bar to update as the features are extracted.
    percentage : float, optional
        The percentage to display on the progress bar.

    Raises
    ------
    PreprocessedAudioNotFoundError
        If no preprocessed dataset audio files are associated with the
        model identified by the provided name.

    """
    model_path = validate_model_exists(model_name, Entity.TRAINING_MODEL)
    sliced_audios16k_path = model_path / "sliced_audios_16k"
    if not sliced_audios16k_path.is_dir() or not any(sliced_audios16k_path.iterdir()):
        raise PreprocessedAudioNotFoundError(model_name)

    custom_embedder_model_path = None
    chosen_embedder_model, embedder_model_id = [embedder_model] * 2
    if embedder_model == EmbedderModel.CUSTOM:
        custom_embedder_model_path = validate_model_exists(
            custom_embedder_model,
            Entity.CUSTOM_EMBEDDER_MODEL,
        )
        json_file = custom_embedder_model_path / "config.json"
        bin_path = custom_embedder_model_path / "pytorch_model.bin"

        combined_file_hash = get_combined_file_hash([json_file, bin_path])
        chosen_embedder_model = str(custom_embedder_model_path)
        embedder_model_id = f"custom_{combined_file_hash}"

    f0_method_id = f0_method
    if f0_method in {TrainingF0Method.CREPE, TrainingF0Method.CREPE_TINY}:
        f0_method_id = f"{f0_method}_{hop_length}"

    devices = [f"cuda:{gpu}" for gpu in gpus] if gpus else ["cpu"]

    # NOTE The lazy_import function does not work with the package below
    # so we import it here manually
    from ultimate_rvc.rvc.train.extract import extract  # noqa: PLC0415

    file_infos = extract.initialize_extraction(
        str(model_path),
        rvc_version,
        f0_method_id,
        embedder_model_id,
    )
    extract.update_model_info(str(model_path), chosen_embedder_model)
    display_progress("[~] Extracting pitch features...", percentage[0], progress_bar)
    extract.run_pitch_extraction(file_infos, devices, f0_method, hop_length, cpu_cores)
    display_progress("[~] Extracting audio embeddings...", percentage[1], progress_bar)
    extract.run_embedding_extraction(
        file_infos,
        devices,
        rvc_version,
        embedder_model,
        (
            str(custom_embedder_model_path)
            if custom_embedder_model_path is not None
            else None
        ),
        cpu_cores,
    )
    preparing_files.generate_config(rvc_version, int(sample_rate), str(model_path))
    preparing_files.generate_filelist(
        str(model_path),
        rvc_version,
        int(sample_rate),
        include_mutes,
        f0_method_id,
        embedder_model_id,
    )
