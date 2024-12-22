"""Module which defines the code for the "Manage models" tab."""

from __future__ import annotations

from typing import TYPE_CHECKING

from collections.abc import Sequence
from functools import partial

import gradio as gr

from ultimate_rvc.common import lazy_import
from ultimate_rvc.core.manage.models import (
    delete_all_models,
    delete_all_training_models,
    delete_training_models,
    delete_voice_models,
    download_voice_model,
    filter_public_models_table,
    get_public_model_tags,
    get_voice_model_names,
    upload_voice_model,
)
from ultimate_rvc.web.common import (
    PROGRESS_BAR,
    confirm_box_js,
    confirmation_harness,
    exception_harness,
    render_msg,
    update_dropdowns,
    update_training_models,
)

if TYPE_CHECKING:

    import pandas as pd

    from ultimate_rvc.web.typing_extra import DropdownValue

else:
    pd = lazy_import("pandas")


def _update_voice_models(
    num_components: int,
    value: DropdownValue = None,
    value_indices: Sequence[int] = [],
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    """
    Update the choices of one or more dropdown components to the set of
    currently saved voice models.

    Optionally updates the default value of one or more of these
    components.

    Parameters
    ----------
    num_components : int
        Number of dropdown components to update.
    value : DropdownValue, optional
        New value for dropdown components.
    value_indices : Sequence[int], default=[]
        Indices of dropdown components to update the value for.

    Returns
    -------
    gr.Dropdown | tuple[gr.Dropdown, ...]
        Updated dropdown component or components.

    """
    return update_dropdowns(get_voice_model_names, num_components, value, value_indices)


def _filter_public_models_table(tags: Sequence[str], query: str) -> gr.Dataframe:
    """
    Filter table containing metadata of public voice models by tags and
    a search query.

    Parameters
    ----------
    tags : Sequence[str]
        Tags to filter the metadata table by.
    query : str
        Search query to filter the metadata table by.

    Returns
    -------
    gr.Dataframe
        The filtered table rendered in a Gradio dataframe.

    """
    models_table = filter_public_models_table(tags, query)
    return gr.Dataframe(value=models_table)


def _autofill_model_name_and_url(
    public_models_table: pd.DataFrame,
    select_event: gr.SelectData,
) -> tuple[gr.Textbox, gr.Textbox]:
    """
    Autofill two textboxes with respectively the name and URL that is
    saved in the currently selected row of the public models table.

    Parameters
    ----------
    public_models_table : pd.DataFrame
        The public models table saved in a Pandas dataframe.
    select_event : gr.SelectData
        Event containing the index of the currently selected row in the
        public models table.

    Returns
    -------
    name : gr.Textbox
        The textbox containing the model name.

    url : gr.Textbox
        The textbox containing the model URL.

    Raises
    ------
    TypeError
        If the index in the provided event is not a sequence.

    """
    event_index = select_event.index
    if not isinstance(event_index, Sequence):
        err_msg = (
            f"Expected a sequence of indices but got {type(event_index)} from the"
            " provided event."
        )
        raise TypeError(err_msg)
    event_index = event_index[0]
    url = public_models_table.loc[event_index, "URL"]
    name = public_models_table.loc[event_index, "Name"]
    if isinstance(url, str) and isinstance(name, str):
        return gr.Textbox(value=name), gr.Textbox(value=url)
    err_msg = (
        "Expected model name and URL to be strings but got"
        f" {type(name)} and {type(url)} respectively."
    )
    raise TypeError(err_msg)


def render(
    voice_model_delete: gr.Dropdown,
    song_cover_model_1click: gr.Dropdown,
    song_cover_model_multi: gr.Dropdown,
    speech_model_1click: gr.Dropdown,
    speech_model_multi: gr.Dropdown,
    training_model_delete: gr.Dropdown,
    training_model_multi: gr.Dropdown,
) -> None:
    """

    Render "Manage models" tab.

    Parameters
    ----------
    voice_model_delete : gr.Dropdown
        Dropdown for selecting voice models to delete in the
        "Delete models" tab.
    song_cover_model_1click : gr.Dropdown
        Dropdown for selecting a voice model to use in the
        "Generate song covers - One-click generation" tab.
    song_cover_model_multi : gr.Dropdown
        Dropdown for selecting a voice model to use in the
        "Generate song covers - Multi-step generation" tab.
    speech_model_1click : gr.Dropdown
        Dropdown for selecting a voice model to use in the
        "Generate speech - One Click Generation" tab.
    speech_model_multi : gr.Dropdown
        Dropdown for selecting a voice model to use in the
        "Generate speech - Multi-step Generation" tab.
    training_model_delete : gr.Dropdown
        Dropdown for selecting training models to delete in the
        "Delete models" tab.
    training_model_multi : gr.Dropdown
        Dropdown for selecting a voice model to use in the
        "Train models - multi-step generation" tab.

    """
    # Download tab

    dummy_checkbox = gr.Checkbox(visible=False)
    with gr.Tab("Download voice model"):
        with gr.Accordion("View public models table", open=False):
            gr.Markdown("")
            gr.Markdown("*HOW TO USE*")
            gr.Markdown(
                "- Filter voice models by selecting one or more tags and/or providing a"
                " search query.",
            )
            gr.Markdown(
                "- Select a row in the table to autofill the name and"
                " URL for the given voice model in the form fields below.",
            )
            gr.Markdown("")
            with gr.Row():
                search_query = gr.Textbox(label="Search query")
                tags = gr.CheckboxGroup(
                    value=[],
                    label="Tags",
                    choices=get_public_model_tags(),
                )
            with gr.Row():
                public_models_table = gr.Dataframe(
                    value=_filter_public_models_table,
                    inputs=[tags, search_query],
                    headers=["Name", "Description", "Tags", "Credit", "Added", "URL"],
                    label="Public models table",
                    interactive=False,
                )

        with gr.Row():
            voice_model_url = gr.Textbox(
                label="Voice model URL",
                info=(
                    "Should point to a zip file containing a .pth model file and"
                    " optionally also an .index file."
                ),
            )
            voice_model_name = gr.Textbox(
                label="Voice model name",
                info="Enter a unique name for the voice model.",
            )

        with gr.Row(equal_height=True):
            download_btn = gr.Button("Download 🌐", variant="primary", scale=19)
            download_msg = gr.Textbox(
                label="Output message",
                interactive=False,
                scale=20,
            )

        public_models_table.select(
            _autofill_model_name_and_url,
            inputs=public_models_table,
            outputs=[voice_model_name, voice_model_url],
            show_progress="hidden",
        )

        download_btn_click = download_btn.click(
            partial(
                exception_harness(download_voice_model),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=[voice_model_url, voice_model_name],
            outputs=download_msg,
        ).success(
            partial(
                render_msg,
                "[+] Succesfully downloaded voice model!",
            ),
            outputs=download_msg,
            show_progress="hidden",
        )

    # Upload tab
    with gr.Tab("Upload voice model"):
        with gr.Accordion("HOW TO USE"):
            gr.Markdown("")
            gr.Markdown(
                "1. Find the .pth file for a locally trained RVC model (e.g. in your"
                " local weights folder) and optionally also a corresponding .index file"
                " (e.g. in your logs/[name] folder)",
            )
            gr.Markdown(
                "2. Upload the files directly or save them to a folder, then compress"
                " that folder and upload the resulting .zip file",
            )
            gr.Markdown("3. Enter a unique name for the uploaded model")
            gr.Markdown("4. Click 'Upload'")

        with gr.Row():
            voice_model_files = gr.File(
                label="Files",
                file_count="multiple",
                file_types=[".zip", ".pth", ".index"],
            )

            local_voice_model_name = gr.Textbox(label="Voice model name")

        with gr.Row(equal_height=True):
            upload_btn = gr.Button("Upload", variant="primary", scale=19)
            upload_msg = gr.Textbox(
                label="Output message",
                interactive=False,
                scale=20,
            )
            upload_btn_click = upload_btn.click(
                partial(
                    exception_harness(upload_voice_model),
                    progress_bar=PROGRESS_BAR,
                ),
                inputs=[voice_model_files, local_voice_model_name],
                outputs=upload_msg,
            ).success(
                partial(
                    render_msg,
                    "[+] Successfully uploaded voice model!",
                ),
                outputs=upload_msg,
                show_progress="hidden",
            )

    with gr.Tab("Delete models"):
        with gr.Accordion("Voice models", open=False), gr.Row():
            with gr.Column():
                voice_model_delete.render()
                delete_voice_btn = gr.Button("Delete selected", variant="secondary")
                delete_all_voice_btn = gr.Button("Delete all", variant="primary")
            with gr.Column():
                delete_voice_msg = gr.Textbox(label="Output message", interactive=False)

        with gr.Accordion("Training models", open=False), gr.Row():
            with gr.Column():
                training_model_delete.render()
                delete_train_btn = gr.Button("Delete selected", variant="secondary")
                delete_all_train_btn = gr.Button("Delete all", variant="primary")
            with gr.Column():
                delete_train_msg = gr.Textbox(label="Output message", interactive=False)

        with gr.Accordion("All models"), gr.Row(equal_height=True):
            delete_all_btn = gr.Button("Delete", variant="primary")
            delete_all_msg = gr.Textbox(label="Output message", interactive=False)

        delete_voice_btn_click = delete_voice_btn.click(
            partial(
                confirmation_harness(delete_voice_models),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=[dummy_checkbox, voice_model_delete],
            outputs=delete_voice_msg,
            js=confirm_box_js(
                "Are you sure you want to delete the selected voice models?",
            ),
        ).success(
            partial(render_msg, "[-] Successfully deleted selected voice models!"),
            outputs=delete_voice_msg,
            show_progress="hidden",
        )

        delete_all_voice_btn_click = delete_all_voice_btn.click(
            partial(
                confirmation_harness(delete_all_models),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=dummy_checkbox,
            outputs=delete_voice_msg,
            js=confirm_box_js("Are you sure you want to delete all voice models?"),
        ).success(
            partial(render_msg, "[-] Successfully deleted all voice models!"),
            outputs=delete_voice_msg,
            show_progress="hidden",
        )

        delete_train_btn_click = delete_train_btn.click(
            partial(
                confirmation_harness(delete_training_models),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=[dummy_checkbox, training_model_delete],
            outputs=delete_train_msg,
            js=confirm_box_js(
                "Are you sure you want to delete the selected training models?",
            ),
        ).success(
            partial(render_msg, "[-] Successfully deleted selected training models!"),
            outputs=delete_train_msg,
            show_progress="hidden",
        )

        delete_all_train_btn_click = delete_all_train_btn.click(
            partial(
                confirmation_harness(delete_all_training_models),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=dummy_checkbox,
            outputs=delete_train_msg,
            js=confirm_box_js("Are you sure you want to delete all training models?"),
        ).success(
            partial(render_msg, "[-] Successfully deleted all training models!"),
            outputs=delete_train_msg,
            show_progress="hidden",
        )

        delete_all_click = delete_all_btn.click(
            partial(
                confirmation_harness(delete_all_models),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=dummy_checkbox,
            outputs=delete_all_msg,
            js=confirm_box_js("Are you sure you want to delete all models?"),
        ).success(
            partial(render_msg, "[-] Successfully deleted all models!"),
            outputs=delete_all_msg,
            show_progress="hidden",
        )

    *_, all_model_update = [
        click_event.success(
            partial(_update_voice_models, 5, [], [4]),
            outputs=[
                song_cover_model_1click,
                song_cover_model_multi,
                speech_model_1click,
                speech_model_multi,
                voice_model_delete,
            ],
            show_progress="hidden",
        )
        for click_event in [
            download_btn_click,
            upload_btn_click,
            delete_voice_btn_click,
            delete_all_voice_btn_click,
            delete_all_click,
        ]
    ]

    for click_event in [
        delete_train_btn_click,
        delete_all_train_btn_click,
        all_model_update,
    ]:
        click_event.success(
            partial(update_training_models, 2, [], [0, 1]),
            outputs=[training_model_multi, training_model_delete],
            show_progress="hidden",
        )
