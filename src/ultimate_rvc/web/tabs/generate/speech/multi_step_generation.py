"""
Module which defines the code for the "Generate Speech - Multi-step
Generation" tab.
"""

import gradio as gr


def render(
    edge_tts_voice: gr.Dropdown,
    model_multi: gr.Dropdown,
    speech_audio: gr.Dropdown,
    output_audio: gr.Dropdown,
) -> None:
    """Render "Generate Speech - Multi-step Generation" tab."""
    with gr.Tab("Multi-step generation", visible=False):
        model_multi.render()
