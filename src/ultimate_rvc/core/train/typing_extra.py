"""
Module which defines extra types used by modules in the
ultimate_rvc.core.train package.
"""

from __future__ import annotations

from pydantic import BaseModel

# NOTE these types are used at runtime by pydantic so cannot be
# relegated to a IF TYPE_CHECKING block
from ultimate_rvc.typing_extra import RVCVersion  # noqa: TC002


class ModelInfo(BaseModel):
    """
    Information about a voice model to be trained.

    Attributes
    ----------
    sample_rate : PretrainedSampleRate
        The sample rate of the post-processed audio to train the model
        on.
    rvc_version : RVCVersion
        The version of the RVC package used to train the model.

    """

    sample_rate: int
    rvc_version: RVCVersion
    # TODO add more attributes later
