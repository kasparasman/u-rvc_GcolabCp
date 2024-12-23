"""
Module which defines the code for the
"Train models - multi-step generation" tab.
"""

from __future__ import annotations

from functools import partial
from multiprocessing import cpu_count

import gradio as gr

from ultimate_rvc.core.train.prepare import (
    populate_dataset,
    preprocess_dataset,
)
from ultimate_rvc.typing_extra import AudioExt, TrainingSampleRate
from ultimate_rvc.web.common import (
    PROGRESS_BAR,
    exception_harness,
    render_msg,
    toggle_visibility,
    update_audio_datasets,
    update_named_audio_datasets,
    update_training_models,
)


def render(
    dataset: gr.Dropdown,
    dataset_audio: gr.Dropdown,
    training_model_multi: gr.Dropdown,
    training_model_delete: gr.Dropdown,
) -> None:
    """
    Render the "Train models - multi-step generation" tab.

    Parameters
    ----------
    dataset : gr.Dropdown
        Dropdown to display available datasets in the "Train models -
        multi-step generation" tab.
    dataset_audio : gr.Dropdown
        Dropdown for selecting dataset audio files to delete in the
        "Delete audio" tab.
    training_model_multi: gr.Dropdown
        Dropdown to select the name of the model to train in the "Train
        models - multi-step generation" tab.
    training_model_delete : gr.Dropdown
        Dropdown for selecting training models to delete in the
        "Delete models" tab.

    """
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
                populate_msg = gr.Textbox(
                    label="Output message",
                    interactive=False,
                )
                populate_btn.click(
                    partial(
                        exception_harness(
                            populate_dataset,
                        ),
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
                    partial(update_audio_datasets, 1, value_indices=[0]),
                    inputs=current_dataset,
                    outputs=dataset,
                    show_progress="hidden",
                ).then(
                    partial(update_named_audio_datasets, 1, [], [0]),
                    outputs=dataset_audio,
                    show_progress="hidden",
                )
        with gr.Accordion("Step 1: dataset preprocessing", open=True):
            with gr.Row():
                training_model_multi.render()
                dataset.render()
                sampling_rate = gr.Radio(
                    choices=list(TrainingSampleRate),
                    label="Sampling rate",
                    info=(
                        "The sampling rate of the audio files in the dataset to"
                        " preprocess."
                    ),
                    value=TrainingSampleRate.HZ_40000,
                )
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
                            partial(toggle_visibility, target=True),
                            inputs=clean_audio,
                            outputs=clean_strength,
                            show_progress="hidden",
                        )
            with gr.Row(equal_height=True):
                preprocess_btn = gr.Button("Preprocess dataset", variant="primary")
                preprocess_msg = gr.Textbox(
                    label="Output message",
                    interactive=False,
                )
                preprocess_btn.click(
                    partial(
                        exception_harness(
                            preprocess_dataset,
                        ),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[
                        training_model_multi,
                        dataset,
                        sampling_rate,
                        cpu_cores_preprocess,
                        split_audio,
                        filter_audio,
                        clean_audio,
                        clean_strength,
                    ],
                    outputs=preprocess_msg,
                ).success(
                    partial(
                        render_msg,
                        "[+] Dataset successfully preprocessed!",
                    ),
                    outputs=preprocess_msg,
                    show_progress="hidden",
                ).then(
                    partial(update_training_models, 2, [], [1]),
                    outputs=[training_model_multi, training_model_delete],
                    show_progress="hidden",
                )
