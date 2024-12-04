"""
Extra type definitions for the audio generation commands in the
CLI of the Ultimate RVC project.
"""

from __future__ import annotations

from enum import StrEnum


class PanelName(StrEnum):
    """
    Valid panel names for audio generation commands in the CLI of
    the Ultimate RVC project.
    """

    MAIN_OPTIONS = "Main Options"
    VOICE_SYNTHESIS_OPTIONS = "Voice Synthesis Options"
    VOCAL_ENRICHMENT_OPTIONS = "Vocal Enrichment Options"
    SPEAKER_EMBEDDINGS_OPTIONS = "Speaker Embeddings Options"
    VOCAL_POST_PROCESSING_OPTIONS = "Vocal Post-processing Options"
    AUDIO_MIXING_OPTIONS = "Audio Mixing Options"
