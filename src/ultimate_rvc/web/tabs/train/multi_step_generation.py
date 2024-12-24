"""
Module which defines the code for the
"Train models - multi-step generation" tab.
"""

from __future__ import annotations

from functools import partial
from multiprocessing import cpu_count

import gradio as gr

from ultimate_rvc.core.manage.audio import get_audio_datasets, get_named_audio_datasets
from ultimate_rvc.core.manage.models import get_training_model_names
from ultimate_rvc.core.train.common import get_gpu_info
from ultimate_rvc.core.train.extract import extract_features
from ultimate_rvc.core.train.prepare import (
    populate_dataset,
    preprocess_dataset,
)
from ultimate_rvc.typing_extra import (
    AudioExt,
    EmbedderModel,
    RVCVersion,
    TrainingF0Method,
    TrainingSampleRate,
)
from ultimate_rvc.web.common import (
    PROGRESS_BAR,
    exception_harness,
    render_msg,
    toggle_visibility,
    update_dropdowns,
    update_values,
)


def render(
    dataset: gr.Dropdown,
    preprocess_model: gr.Dropdown,
    custom_embedder_model: gr.Dropdown,
    extract_model: gr.Dropdown,
    training_model_delete: gr.Dropdown,
    dataset_audio: gr.Dropdown,
) -> None:
    """
    Render the "Train models - multi-step generation" tab.

    Parameters
    ----------
    dataset : gr.Dropdown
        Dropdown to display available datasets in the "Train models -
        multi-step generation" tab.
    preprocess_model : gr.Dropdown
        Dropdown for selecting a voice model to preprocess a
        dataset for in the "Train models - multi-step generation" tab.
    custom_embedder_model : gr.Dropdown
        Dropdown for selecting a custom embedder model to use for
        extracting audio embeddings in the "Train models - multi-step
        generation" tab.
    extract_model : gr.Dropdown
        Dropdown for selecting a voice model with an associated
        preprocessed dataset to extract features from in the
        "Train models - multi-step generation" tab
    training_model_delete : gr.Dropdown
        Dropdown for selecting training models to delete in the
        "Delete models" tab.
    dataset_audio : gr.Dropdown
        Dropdown for selecting dataset audio files to delete in the
        "Delete audio" tab.

    """
    sample_rate_preprocess, sample_rate_extract = [
        gr.Radio(
            choices=list(TrainingSampleRate),
            label="Sampling rate",
            info=info,
            value=TrainingSampleRate.HZ_40K,
            render=False,
        )
        for info in [
            (
                "Target sample rate for the audio files in the dataset to"
                " preprocess.<br><br><br>"
            ),
            (
                "The sample rate of the audio files in the preprocessed dataset"
                " associated with the model. When a new dataset is preprocessed,"
                " its target sample rate is selected by default."
            ),
        ]
    ]
    current_dataset = gr.State()
    with gr.Tab("Multi-step generation"):
        with gr.Accordion("Step 0: dataset population", open=True):
            with gr.Row():
                dataset_name = gr.Textbox(
                    label="Dataset name",
                    info=(
                        "The name of the dataset to populate. If the dataset does not"
                        " exist, it will be created."
                    ),
                    value="My dataset",
                )
                audio_files = gr.File(
                    file_count="multiple",
                    label="Audio files to populate the given dataset with.",
                    file_types=[f".{e.value}" for e in AudioExt],
                )
            with gr.Row(equal_height=True):
                populate_btn = gr.Button("Populate dataset", variant="primary")
                populate_msg = gr.Textbox(label="Output message", interactive=False)
                populate_btn.click(
                    partial(
                        exception_harness(populate_dataset),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[dataset_name, audio_files],
                    outputs=current_dataset,
                ).success(
                    partial(
                        render_msg,
                        "[+] Dataset successfully populated with the provided audio"
                        " files !",
                    ),
                    outputs=populate_msg,
                ).then(
                    partial(update_dropdowns, get_audio_datasets, 1, value_indices=[0]),
                    inputs=current_dataset,
                    outputs=dataset,
                    show_progress="hidden",
                ).then(
                    partial(update_dropdowns, get_named_audio_datasets, 1, [], [0]),
                    outputs=dataset_audio,
                    show_progress="hidden",
                )
        with gr.Accordion("Step 1: dataset preprocessing", open=True):
            with gr.Row():
                preprocess_model.render()
                dataset.render()
                sample_rate_preprocess.render()
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    cpu_cores_preprocess = gr.Slider(
                        1,
                        cpu_count(),
                        cpu_count(),
                        step=1,
                        label="CPU cores",
                        info="The number of CPU cores to use for preprocessing.",
                    )
                with gr.Row():
                    with gr.Column():
                        split_audio = gr.Checkbox(
                            value=True,
                            label="Split audio",
                            info=(
                                "Whether to split the audio files in the provided"
                                " dataset into smaller segments before pre-processing."
                                " This help can improve the pre-processing speed for"
                                " large audio files."
                            ),
                        )
                    with gr.Column():
                        filter_audio = gr.Checkbox(
                            value=True,
                            label="Filter audio",
                            info=(
                                "Whether to remove low-frequency sounds from the audio"
                                " files in the provided dataset by applying a high-pass"
                                " butterworth filter."
                            ),
                        )
                    with gr.Column():
                        clean_audio = gr.Checkbox(
                            label="Clean audio",
                            info=(
                                "Whether to clean the audio files in the provided"
                                " dataset using noise reduction algorithms."
                            ),
                        )
                        clean_strength = gr.Slider(
                            0.0,
                            1.0,
                            0.7,
                            step=0.1,
                            label="Clean strength",
                            info="The strength of the audio cleaning process.",
                            visible=False,
                        )
                        clean_audio.change(
                            partial(toggle_visibility, targets={True}),
                            inputs=clean_audio,
                            outputs=clean_strength,
                            show_progress="hidden",
                        )
            with gr.Row(equal_height=True):
                preprocess_btn = gr.Button("Preprocess dataset", variant="primary")
                preprocess_msg = gr.Textbox(label="Output message", interactive=False)
                preprocess_btn.click(
                    partial(
                        exception_harness(preprocess_dataset),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[
                        preprocess_model,
                        dataset,
                        sample_rate_preprocess,
                        cpu_cores_preprocess,
                        split_audio,
                        filter_audio,
                        clean_audio,
                        clean_strength,
                    ],
                    outputs=preprocess_msg,
                ).success(
                    partial(render_msg, "[+] Dataset successfully preprocessed!"),
                    outputs=preprocess_msg,
                    show_progress="hidden",
                ).then(
                    partial(update_dropdowns, get_training_model_names, 3, [], [2]),
                    outputs=[
                        preprocess_model,
                        extract_model,
                        training_model_delete,
                    ],
                    show_progress="hidden",
                ).then(
                    update_values,
                    inputs=[preprocess_model, sample_rate_preprocess],
                    outputs=[extract_model, sample_rate_extract],
                    show_progress="hidden",
                )
        with gr.Accordion("Step 2: feature extraction", open=True):
            with gr.Row():
                extract_model.render()
                rvc_version = gr.Radio(
                    choices=list(RVCVersion),
                    label="RVC version",
                    info="The version of RVC to use for training the model.<br><br>",
                    value=RVCVersion.V2,
                )
                sample_rate_extract.render()
            with gr.Accordion("Advanced Settings", open=False):

                with gr.Row():
                    cpu_cores_extract = gr.Slider(
                        1,
                        cpu_count(),
                        cpu_count(),
                        step=1,
                        label="CPU cores",
                        info=(
                            "The number of CPU cores to use for feature"
                            " extraction.<br><br>"
                        ),
                    )
                    gpu_choices = get_gpu_info()
                    gpus = gr.Dropdown(
                        choices=gpu_choices,
                        value=gpu_choices[0][1] if gpu_choices else None,
                        label="GPU(s)",
                        info="The GPU(s) to use for feature extraction.",
                        multiselect=True,
                    )
                with gr.Row():
                    with gr.Column():
                        f0_method = gr.Dropdown(
                            choices=list(TrainingF0Method),
                            label="F0 method",
                            info="The method to use for extracting pitch features.",
                            value=TrainingF0Method.RMVPE,
                        )
                        hop_length = gr.Slider(
                            1,
                            512,
                            128,
                            step=1,
                            label="Hop length",
                            info=(
                                "The hop length to use for extracting pitch"
                                " features. Only used with the CREPE pitch extraction"
                                " method.<br><br>"
                            ),
                            visible=False,
                        )
                    f0_method.change(
                        partial(
                            toggle_visibility,
                            targets={
                                TrainingF0Method.CREPE,
                                TrainingF0Method.CREPE_TINY,
                            },
                        ),
                        inputs=f0_method,
                        outputs=hop_length,
                        show_progress="hidden",
                    )
                    with gr.Column():
                        embedder_model = gr.Dropdown(
                            choices=list(EmbedderModel),
                            label="Embedder model",
                            info="The model to use for extracting audio embeddings.",
                            value=EmbedderModel.CONTENTVEC,
                        )
                        custom_embedder_model.render()

                    embedder_model.change(
                        partial(toggle_visibility, targets={EmbedderModel.CUSTOM}),
                        inputs=embedder_model,
                        outputs=custom_embedder_model,
                        show_progress="hidden",
                    )
            with gr.Row(equal_height=True):
                extract_btn = gr.Button("Extract features", variant="primary")
                extract_msg = gr.Textbox(label="Output message", interactive=False)
                extract_btn.click(
                    partial(
                        exception_harness(extract_features),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[
                        extract_model,
                        rvc_version,
                        f0_method,
                        hop_length,
                        cpu_cores_extract,
                        gpus,
                        sample_rate_extract,
                        embedder_model,
                        custom_embedder_model,
                    ],
                    outputs=extract_msg,
                ).success(
                    partial(render_msg, "[+] Features successfully extracted!"),
                    outputs=extract_msg,
                    show_progress="hidden",
                )
