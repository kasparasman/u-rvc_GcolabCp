"""
Module which defines the code for the "Generate speech - one-click
generation" tab.
"""

from functools import partial

import gradio as gr

from ultimate_rvc.core.generate.speech import get_mixed_speech_track_name, run_pipeline
from ultimate_rvc.core.manage.audio import (
    get_saved_output_audio,
    get_saved_speech_audio,
)
from ultimate_rvc.typing_extra import AudioExt, EmbedderModel, F0Method, SampleRate
from ultimate_rvc.web.common import (
    PROGRESS_BAR,
    exception_harness,
    toggle_intermediate_audio,
    toggle_visibility,
    toggle_visible_component,
    update_dropdowns,
    update_output_name,
    update_value,
)
from ultimate_rvc.web.typing_extra import ConcurrencyId, SpeechSourceType


def render(
    edge_tts_voice: gr.Dropdown,
    voice_model: gr.Dropdown,
    custom_embedder_model: gr.Dropdown,
    speech_audio: gr.Dropdown,
    output_audio: gr.Dropdown,
) -> None:
    """
    Render "Generate speech - one-click generation" tab.

    Parameters
    ----------
    edge_tts_voice : gr.Dropdown
        Dropdown component for selecting an Edge TTS voice in the
        "Generate speech - one-click generation" tab.
    voice_model : gr.Dropdown
        Dropdown component for selecting a voice model in the
        "Generate speech - one-click generation" tab.
    custom_embedder_model : gr.Dropdown
        Dropdown component for selecting a custom embedder model in the
        "Generate speech - one-click generation" tab.
    speech_audio : gr.Dropdown
        Dropdown component for speech audio files to delete in the
        "Delete audio" tab.
    output_audio : gr.Dropdown
        Dropdown for selecting output audio files to delete in the
        "Delete audio" tab.

    """
    with gr.Tab("One-click generation"):
        with gr.Accordion("Main options"):
            with gr.Row():
                with gr.Column():
                    source_type = gr.Dropdown(
                        list(SpeechSourceType),
                        value=SpeechSourceType.TEXT,
                        label="Source type",
                        type="index",
                        info="The type of source to generate speech from.",
                    )
                with gr.Column():
                    source = gr.Textbox(
                        label="Source",
                        info="Text to generate speech from",
                    )
                    local_file = gr.File(
                        label="Source",
                        file_types=[".txt"],
                        file_count="single",
                        type="filepath",
                        visible=False,
                    )
                source_type.input(
                    partial(toggle_visible_component, 2),
                    inputs=source_type,
                    outputs=[source, local_file],
                    show_progress="hidden",
                )
                local_file.change(
                    update_value,
                    inputs=local_file,
                    outputs=source,
                    show_progress="hidden",
                )
            with gr.Row():
                edge_tts_voice.render()
                voice_model.render()
        with gr.Accordion("Edge TTS options", open=False), gr.Row():
            tts_pitch_shift = gr.Slider(
                -100,
                100,
                value=0,
                step=1,
                label="Edge TTS pitch shift",
                info=(
                    "The number of hertz to shift the pitch of the speech generated"
                    " by Edge TTS."
                ),
            )
            tts_speed_change = gr.Slider(
                -50,
                100,
                value=0,
                step=1,
                label="TTS speed change",
                info=(
                    "The percentual change to the speed of the speech generated by"
                    " Edge TTS."
                ),
            )
            tts_volume_change = gr.Slider(
                -100,
                100,
                value=0,
                step=1,
                label="TTS volume change",
                info=(
                    "The percentual change to the volume of the speech generated by"
                    " Edge TTS."
                ),
            )

        with gr.Accordion("Speech conversion options", open=False):
            gr.Markdown("")
            with gr.Accordion("Main settings", open=True), gr.Row():
                n_octaves = gr.Slider(
                    -3,
                    3,
                    value=0,
                    step=1,
                    label="Octave shift",
                    info=(
                        "The number of octaves to pitch-shift the converted speech by."
                        " Use 1 for male-to-female and -1 for vice-versa."
                    ),
                )
                n_semitones = gr.Slider(
                    -12,
                    12,
                    value=0,
                    step=1,
                    label="Semitone shift",
                    info=(
                        "The number of semi-tones to pitch-shift the converted"
                        " speech by."
                    ),
                )
            with gr.Accordion("Voice synthesis settings", open=False):
                with gr.Row():
                    f0_methods = gr.Dropdown(
                        list(F0Method),
                        value=F0Method.RMVPE,
                        label="Pitch extraction algorithm(s)",
                        info=(
                            "If more than one method is selected, then the median of"
                            " the pitch values extracted by each method is used. RMVPE"
                            " is recommended for most cases and is the default when no"
                            " method is selected."
                        ),
                        multiselect=True,
                    )
                    index_rate = gr.Slider(
                        0,
                        1,
                        value=0.5,
                        label="Index rate",
                        info=(
                            "Increase to bias the conversion towards the accent of the"
                            " voice model. Decrease to potentially reduce artifacts"
                            " coming from the voice model.<br><br>"
                        ),
                    )
                    filter_radius = gr.Slider(
                        0,
                        7,
                        value=3,
                        step=1,
                        label="Filter radius",
                        info=(
                            "If >=3: apply median filtering to extracted pitch values."
                            " Can help reduce breathiness in the converted"
                            " speech.<br><br>"
                        ),
                    )
                with gr.Row():
                    rms_mix_rate = gr.Slider(
                        0,
                        1,
                        value=0.25,
                        label="RMS mix rate",
                        info=(
                            "How much to mimic the loudness (0) of the input speech or"
                            " a fixed loudness (1).<br><br><br>"
                        ),
                    )
                    protect_rate = gr.Slider(
                        0,
                        0.5,
                        value=0.33,
                        label="Protect rate",
                        info=(
                            "Controls the extent to which consonants and breathing"
                            " sounds are protected from artifacts. A higher value"
                            " offers more protection but may worsen the indexing"
                            " effect.<br><br>"
                        ),
                    )
                    hop_length = gr.Slider(
                        1,
                        512,
                        value=128,
                        step=1,
                        label="Hop length",
                        info=(
                            "How often the CREPE-based pitch extraction method checks"
                            " for pitch changes measured in milliseconds. Lower values"
                            " lead to longer conversion times and a higher risk of"
                            " voice cracks, but better pitch accuracy."
                        ),
                    )
            with gr.Accordion("Speech enrichment settings", open=False), gr.Row():
                with gr.Column():
                    split_speech = gr.Checkbox(
                        value=True,
                        label="Split speech track",
                        info=(
                            "Whether to split the Edge TTS speech intosmaller segments"
                            " before converting it. This can improve output quality for"
                            " longer speech."
                        ),
                    )
                with gr.Column():
                    autotune_speech = gr.Checkbox(
                        label="Autotune converted speech",
                        info=(
                            "Whether to apply autotune to the converted speech.<br><br>"
                        ),
                    )
                    autotune_strength = gr.Slider(
                        0,
                        1,
                        value=1.0,
                        label="Autotune intensity",
                        info=(
                            "Higher values result in stronger snapping to the"
                            " chromatic grid and artifacting."
                        ),
                        visible=False,
                    )
                with gr.Column():
                    clean_speech = gr.Checkbox(
                        value=True,
                        label="Clean converted speech",
                        info=(
                            "Whether to clean the converted speech using noise"
                            " reduction algorithms.<br><br>"
                        ),
                    )
                    clean_strength = gr.Slider(
                        0,
                        1,
                        value=0.7,
                        label="Cleaning intensity",
                        info=(
                            "Higher values result in stronger cleaning, but may"
                            " lead to a more compressed sound."
                        ),
                        visible=True,
                    )
            autotune_speech.change(
                partial(toggle_visibility, targets={True}),
                inputs=autotune_speech,
                outputs=autotune_strength,
                show_progress="hidden",
            )
            clean_speech.change(
                partial(toggle_visibility, targets={True}),
                inputs=clean_speech,
                outputs=clean_strength,
                show_progress="hidden",
            )
            with gr.Accordion("Speaker embedding settings", open=False), gr.Row():
                with gr.Column():
                    embedder_model = gr.Dropdown(
                        list(EmbedderModel),
                        value=EmbedderModel.CONTENTVEC,
                        label="Embedder model",
                        info="The model to use for generating speaker embeddings.",
                    )
                    custom_embedder_model.render()
                sid = gr.Number(
                    label="Speaker ID",
                    info="Speaker ID for multi-speaker-models.",
                    precision=0,
                )
            embedder_model.change(
                partial(toggle_visibility, targets={EmbedderModel.CUSTOM}),
                inputs=embedder_model,
                outputs=custom_embedder_model,
                show_progress="hidden",
            )
        with gr.Accordion("Audio output options", open=False):
            with gr.Row():
                output_gain = gr.Slider(
                    -20,
                    20,
                    value=0,
                    step=1,
                    label="Output gain",
                    info="The gain to apply to the converted speech during mixing.",
                )
                output_sr = gr.Dropdown(
                    choices=list(SampleRate),
                    value=SampleRate.HZ_44100,
                    label="Output sample rate",
                    info="The sample rate of the mixed speech track.",
                )
            with gr.Row():
                output_name = gr.Textbox(
                    value=partial(
                        update_output_name,
                        get_mixed_speech_track_name,
                        True,  # noqa: FBT003,,
                    ),
                    inputs=[source, voice_model],
                    label="Output name",
                    info=(
                        "If no name is provided, a suitable name will be generated"
                        " automatically."
                    ),
                    placeholder="Ultimate RVC speech",
                )

                output_format = gr.Dropdown(
                    list(AudioExt),
                    value=AudioExt.MP3,
                    label="Output format",
                    info="The format of mixed speech track.",
                )
            with gr.Row():
                show_intermediate_audio = gr.Checkbox(
                    label="Show intermediate audio",
                    value=False,
                    info=(
                        "Show intermediate audio tracks generated during speech"
                        " generation."
                    ),
                )
        intermediate_audio_accordions = [
            gr.Accordion(label, open=False, render=False)
            for label in [
                "Step 1: text-to-speech conversion",
                "Step 2: speech conversion",
            ]
        ]
        tts_accordion, speech_conversion_accordion = intermediate_audio_accordions
        intermediate_audio_tracks = [
            gr.Audio(label=label, type="filepath", interactive=False, render=False)
            for label in [
                "Speech",
                "Converted speech",
            ]
        ]
        speech, converted_speech = intermediate_audio_tracks
        with gr.Accordion(
            "Intermediate audio tracks",
            open=False,
            visible=False,
        ) as intermediate_audio_accordion:
            tts_accordion.render()
            with tts_accordion:
                speech.render()
            speech_conversion_accordion.render()
            with speech_conversion_accordion:
                converted_speech.render()
        show_intermediate_audio.change(
            partial(toggle_intermediate_audio, num_components=2),
            inputs=show_intermediate_audio,
            outputs=[intermediate_audio_accordion, *intermediate_audio_accordions],
            show_progress="hidden",
        )
        with gr.Row(equal_height=True):
            reset_btn = gr.Button(value="Reeset settings", scale=2)
            generate_btn = gr.Button(value="Generate", scale=2, variant="primary")
            mixed_speech = gr.Audio(label="Mixed speech", scale=3)
        generate_btn.click(
            partial(
                exception_harness(
                    run_pipeline,
                    info_msg="Speech generated successfully!",
                ),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=[
                source,
                voice_model,
                edge_tts_voice,
                tts_pitch_shift,
                tts_speed_change,
                tts_volume_change,
                n_octaves,
                n_semitones,
                f0_methods,
                index_rate,
                filter_radius,
                rms_mix_rate,
                protect_rate,
                hop_length,
                split_speech,
                autotune_speech,
                autotune_strength,
                clean_speech,
                clean_strength,
                embedder_model,
                custom_embedder_model,
                sid,
                output_gain,
                output_sr,
                output_format,
                output_name,
            ],
            outputs=[mixed_speech, *intermediate_audio_tracks],
            concurrency_limit=1,
            concurrency_id=ConcurrencyId.GPU,
        ).success(
            partial(update_dropdowns, get_saved_speech_audio, 1),
            outputs=speech_audio,
            show_progress="hidden",
        ).then(
            partial(update_dropdowns, get_saved_output_audio, 1),
            outputs=output_audio,
            show_progress="hidden",
        )
        reset_btn.click(
            lambda: [
                0,
                0,
                0,
                0,
                0,
                F0Method.RMVPE,
                0.5,
                3,
                0.25,
                0.33,
                128,
                True,
                False,
                1.0,
                True,
                0.7,
                EmbedderModel.CONTENTVEC,
                None,
                0,
                0,
                SampleRate.HZ_44100,
                AudioExt.MP3,
                False,
            ],
            outputs=[
                tts_pitch_shift,
                tts_speed_change,
                tts_volume_change,
                n_octaves,
                n_semitones,
                f0_methods,
                index_rate,
                filter_radius,
                rms_mix_rate,
                protect_rate,
                hop_length,
                split_speech,
                autotune_speech,
                autotune_strength,
                clean_speech,
                clean_strength,
                embedder_model,
                custom_embedder_model,
                sid,
                output_gain,
                output_sr,
                output_format,
                show_intermediate_audio,
            ],
            show_progress="hidden",
        )
