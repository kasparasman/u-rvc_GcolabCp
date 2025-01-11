"""
Module which defines extra types used by modules in the
ultimate_rvc.core.manage package.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, RootModel

from ultimate_rvc.typing_extra import PretrainedSampleRate


class VoiceModelTagName(StrEnum):
    """Names of valid voice model tags."""

    ENGLISH = "English"
    JAPANESE = "Japanese"
    OTHER_LANGUAGE = "Other Language"
    ANIME = "Anime"
    VTUBER = "Vtuber"
    REAL_PERSON = "Real person"
    GAME_CHARACTER = "Game character"


class VoiceModelTagMetaData(BaseModel):
    """
    Metadata for a voice model tag.

    Attributes
    ----------
    name : ModelTagName
        The name of the tag.
    description : str
        The description of the tag.

    """

    name: VoiceModelTagName
    description: str


class VoiceModelMetaData(BaseModel):
    """
    Metadata for a voice model.

    Attributes
    ----------
    name : str
        The name of the voice model.
    description : str
        A description of the voice model.
    tags : list[ModelTagName]
        The tags associated with the voice model.
    credit : str
        Who created the voice model.
    added : str
        The date the voice model was created.
    url : str
        An URL pointing to a location where the voice model can be
        downloaded.

    """

    name: str
    description: str
    tags: list[VoiceModelTagName]
    credit: str
    added: str
    url: str


class VoiceModelMetaDataTable(BaseModel):
    """
    Table with metadata for a set of voice models.

    Attributes
    ----------
    tags : list[ModelTagMetaData]
        Metadata for the tags associated with the given set of voice
        models.
    models : list[ModelMetaData]
        Metadata for the given set of voice models.

    """

    tags: list[VoiceModelTagMetaData]
    models: list[VoiceModelMetaData]


VoiceModelMetaDataPredicate = Callable[[VoiceModelMetaData], bool]

VoiceModelMetaDataList = list[list[str | list[VoiceModelTagName]]]


class PretrainedPaths(BaseModel):
    """
    Paths to the generator and discriminator for a pretrained
    model with a given name and sample rate.
    """

    G: str
    D: str


class PretrainedModelMetaData(RootModel[dict[PretrainedSampleRate, PretrainedPaths]]):
    """
    Metadata for a pretrained model with a given name.

    Attributes
    ----------
    root : dict[PretrainedSampleRate, PretrainedPaths]
        Mapping from sample rate to paths to the generator and
        discriminator for the pretrained model with the given name
        at the given sample rate.

    """

    root: dict[PretrainedSampleRate, PretrainedPaths]

    def __getitem__(self, item: PretrainedSampleRate) -> PretrainedPaths:
        """
        Get the paths to the generator and discriminator for the
        pretrained model at the given sample rate.

        Parameters
        ----------
        item : PretrainedSampleRate
            The sample rate for which to get paths to the generator
            and discriminator for the pretrained model.

        Returns
        -------
        PretrainedPaths
            The paths to the generator and discriminator for the
            pretrained model at the given sample rate.

        """
        return self.root[item]

    def keys(self) -> list[PretrainedSampleRate]:
        """
        Get the sample rates for which generator and discriminator
        paths are available for the pretrained model.

        Returns
        -------
        list[PretrainedSampleRate]
            The sample rates for which paths are available for the
            pretrained model.

        """
        return sorted(self.root.keys())


class PretrainedModelMetaDataTable(RootModel[dict[str, PretrainedModelMetaData]]):
    """
    Table with metadata for pretrained models available online.

    Attributes
    ----------
    root : dict[str, PretrainedSampleRates]
        Mapping from the names of pretrained models to metadata for
        those models.

    """

    root: dict[str, PretrainedModelMetaData]

    def __getitem__(self, item: str) -> PretrainedModelMetaData:
        """
        Get the metadata for the pretrained model with the given name.

        Parameters
        ----------
        item : str
            The name of the pretrained model for which to get metadata.

        Returns
        -------
        PretrainedSampleRates
            The metadata for the pretrained model with the given name.

        """
        return self.root[item]

    def keys(self) -> list[str]:
        """
        Get the names of all pretrained models available online.

        Returns
        -------
        list[str]
            The names of all pretrained models available online.

        """
        return sorted(self.root.keys())
