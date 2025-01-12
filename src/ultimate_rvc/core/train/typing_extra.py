"""
Module which defines extra types used by modules in the
ultimate_rvc.core.train package.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """
    Information about a voice model to be trained.

    Attributes
    ----------
    sample_rate : PretrainedSampleRate
        The sample rate of the post-processed audio to train the model
        on.

    """

    sample_rate: int
    # TODO add more attributes later
