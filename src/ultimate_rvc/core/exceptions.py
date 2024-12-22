"""
Module which defines custom exception and enumerations used when
instiating and re-raising those exceptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from enum import StrEnum

if TYPE_CHECKING:
    from ultimate_rvc.typing_extra import StrPath


class Entity(StrEnum):
    """Enumeration of entities that can be provided."""

    DIRECTORY = "directory"
    DIRECTORIES = "directories"
    DATASET = "dataset"
    DATASETS = "datasets"
    DATASET_NAME = "dataset name"
    FILE = "file"
    FILES = "files"
    URL = "URL"
    MODEL_NAME = "model name"
    MODEL_NAMES = "model names"
    MODEL_FILE = "model file"
    SOURCE = "source"
    SONG_DIR = "song directory"
    AUDIO_TRACK = "audio track"
    AUDIO_TRACK_GAIN_PAIRS = "pairs of audio track and gain"
    VOICE_TRACK = "voice track"
    SPEECH_TRACK = "speech track"
    SONG = "song"
    VOCALS_TRACK = "vocals track"
    INSTRUMENTALS_TRACK = "instrumentals track"
    BACKUP_VOCALS_TRACK = "backup vocals track"
    MAIN_VOCALS_TRACK = "main vocals track"
    EMBEDDER_MODEL_CUSTOM = "custom embedder model"


class Location(StrEnum):
    """Enumeration of locations where entities can be found."""

    INTERMEDIATE_AUDIO_ROOT = "the root of the intermediate audio base directory"
    TRAINING_AUDIO_ROOT = "the root of the training audio directory"
    SPEECH_AUDIO_ROOT = "the root of the speech audio directory"
    OUTPUT_AUDIO_ROOT = "the root of the output audio directory"
    EXTRACTED_ZIP_FILE = "extracted zip file"


class UIMessage(StrEnum):
    """
    Enumeration of messages that can be displayed in the UI
    in place of core exception messages.
    """

    NO_AUDIO_TRACK = "No audio tracks provided."
    NO_SONG_DIR = "No song directory selected."
    NO_SONG_DIRS = (
        "No song directories selected. Please select one or more song directories"
        " containing intermediate audio files to delete."
    )
    NO_DATASETS = (
        "No datasets selected. Please select one or more datasets containing audio"
        " files to delete."
    )
    NO_OUTPUT_AUDIO_FILES = (
        "No files selected. Please select one or more output audio files to delete."
    )
    NO_SPEECH_AUDIO_FILES = (
        "No files selected. Please select one or more speech audio files to delete."
    )
    NO_UPLOADED_FILES = "No files selected."
    NO_VOICE_MODEL = "No voice model selected."
    NO_VOICE_MODELS = "No voice models selected."
    NO_TRAINING_MODELS = "No training models selected."
    NO_AUDIO_SOURCE = (
        "No source provided. Please provide a valid Youtube URL, local audio file"
        " or song directory."
    )
    NO_TEXT_SOURCE = (
        "No source provided. Please provide a valid text string or path to a text file."
    )


class NotProvidedError(ValueError):
    """Raised when an entity is not provided."""

    def __init__(self, entity: Entity, ui_msg: UIMessage | None = None) -> None:
        """
        Initialize a NotProvidedError instance.

        Exception message will be formatted as:

        "No `<entity>` provided."

        Parameters
        ----------
        entity : Entity
            The entity that was not provided.
        ui_msg : UIMessage, default=None
            Message which, if provided, is displayed in the UI
            instead of the default exception message.

        """
        super().__init__(f"No {entity} provided.")
        self.ui_msg = ui_msg


class NotFoundError(OSError):
    """Raised when an entity is not found."""

    def __init__(
        self,
        entity: Entity,
        location: StrPath | Location,
        is_path: bool = True,
    ) -> None:
        """
        Initialize a NotFoundError instance.

        Exception message will be formatted as:

        "`<entity>` not found `(`in `|` at:`)` `<location>`."

        Parameters
        ----------
        entity : Entity
            The entity that was not found.
        location : StrPath | Location
            The location where the entity was not found.
        is_path : bool, default=True
            Whether the location is a path to the entity.

        """
        proposition = "at:" if is_path else "in"
        entity_cap = entity.capitalize() if not entity.isupper() else entity
        super().__init__(
            f"{entity_cap} not found {proposition} {location}",
        )


class ModelNotFoundError(OSError):
    """Raised when a model is not found."""

    def __init__(self, type_: str, name: str) -> None:
        r"""
        Initialize a ModelNotFoundError instance.

        Exception message will be formatted as:

        '`<type_>` with name "`<name>`" not found.'

        Parameters
        ----------
        type_ : str
            The type of model that was not found.
        name : str
            The name of the model that was not found.

        """
        super().__init__(f"{type_} with name '{name}' not found.")


class VoiceModelNotFoundError(ModelNotFoundError):
    """Raised when a voice model is not found."""

    def __init__(self, name: str) -> None:
        r"""
        Initialize a VoiceModelNotFoundError instance.

        Exception message will be formatted as:

        'Voice model with name "`<name>`" not found.'

        Parameters
        ----------
        name : str
            The name of the voice model that was not found.

        """
        super().__init__("Voice model", name)


class TrainingModelNotFoundError(ModelNotFoundError):
    """Raised when a training model is not found."""

    def __init__(self, name: str) -> None:
        r"""
        Initialize a TrainingModelNotFoundError instance.

        Exception message will be formatted as:

        'Training model with name "`<name>`" not found.'

        Parameters
        ----------
        name : str
            The name of the training model that was not found.

        """
        super().__init__("Training model", name)


class ProtectedModelError(OSError):
    """Raised when a protected model is attempted to be deleted."""

    def __init__(self, name: str) -> None:
        r"""
        Initialize a ProtectedModelError instance.

        Exception message will be formatted as:

        'Model with name "`<name>`" is protected and cannot be deleted.'

        Parameters
        ----------
        name : str
            The name of the protected model.

        """
        super().__init__(
            f"Model with name '{name}' is protected and cannot be deleted.",
        )


class VoiceModelExistsError(OSError):
    """Raised when a voice model already exists."""

    def __init__(self, name: str) -> None:
        r"""
        Initialize a VoiceModelExistsError instance.

        Exception message will be formatted as:

        "Voice model with name '`<name>`' already exists. Please provide
        a different name for your voice model."

        Parameters
        ----------
        name : str
            The name of the voice model that already exists.

        """
        super().__init__(
            f'Voice model with name "{name}" already exists. Please provide a different'
            " name for your voice model.",
        )


class InvalidLocationError(OSError):
    """Raised when an entity is in a wrong location."""

    def __init__(self, entity: Entity, location: Location, path: StrPath) -> None:
        r"""
        Initialize an InvalidLocationError instance.

        Exception message will be formatted as:

        "`<entity>` should be located in `<location>` but found at:
        `<path>`"

        Parameters
        ----------
        entity : Entity
            The entity that is in a wrong location.
        location : Location
            The correct location for the entity.
        path : StrPath
            The path to the entity.

        """
        entity_cap = entity.capitalize() if not entity.isupper() else entity
        super().__init__(
            f"{entity_cap} should be located in {location} but found at: {path}",
        )


class HttpUrlError(OSError):
    """Raised when a HTTP-based URL is invalid."""

    def __init__(self, url: str) -> None:
        """
        Initialize a HttpUrlError instance.

        Exception message will be formatted as:

        "Invalid HTTP-based URL: `<url>`"

        Parameters
        ----------
        url : str
            The invalid HTTP-based URL.

        """
        super().__init__(
            f"Invalid HTTP-based URL: {url}",
        )


class YoutubeUrlError(OSError):
    """
    Raised when an URL does not point to a YouTube video or
    , potentially, a Youtube playlist.
    """

    def __init__(self, url: str, playlist: bool) -> None:
        """
        Initialize a YoutubeURlError instance.

        Exception message will be formatted as:

        "URL does not point to a YouTube video `[`or playlist`]`:
         `<url>`"

        Parameters
        ----------
        url : str
            The URL that does not point to a YouTube video or playlist.
        playlist : bool
            Whether the URL might point to a YouTube playlist.

        """
        suffix = "or playlist" if playlist else ""
        super().__init__(
            f"Not able to access Youtube video {suffix} at: {url}",
        )


class UploadLimitError(ValueError):
    """Raised when the upload limit for an entity is exceeded."""

    def __init__(self, entity: Entity, limit: str | float) -> None:
        """
        Initialize an UploadLimitError instance.

        Exception message will be formatted as:

        "At most `<limit>` `<entity>` can be uploaded."

        Parameters
        ----------
        entity : Entity
            The entity for which the upload limit was exceeded.
        limit : str
            The upload limit.

        """
        super().__init__(f"At most {limit} {entity} can be uploaded.")


class UploadFormatError(ValueError):
    """
    Raised when one or more uploaded entities have an invalid format
    .
    """

    def __init__(self, entity: Entity, formats: list[str], multiple: bool) -> None:
        """
        Initialize an UploadFileFormatError instance.


        Exception message will be formatted as:

        "Only `<entity>` with the following formats can be uploaded
        `(`by themselves | together`)`: `<formats>`."

        Parameters
        ----------
        entity : Entity
            The entity that was uploaded with an invalid format.
        formats : list[str]
            Valid formats.
        multiple : bool
            Whether multiple entities are uploaded.

        """
        suffix = "by themselves" if not multiple else "together (at most one of each)"
        super().__init__(
            f"Only {entity} with the following formats can be uploaded {suffix}:"
            f" {', '.join(formats)}.",
        )


class InvalidAudioFormatError(ValueError):
    """Raised when an audio file has an invalid format."""

    def __init__(self, path: StrPath, formats: list[str]) -> None:
        """
        Initialize an InvalidAudioFormatError instance.

        Exception message will be formatted as:

        "Invalid audio file format: `<path>`. Supported formats are:
        `<formats>`."

        Parameters
        ----------
        path : StrPath
            The path to the audio file with an invalid format.
        formats : list[str]
            Supported audio formats.

        """
        super().__init__(
            f"Invalid audio file format: {path}. Supported formats are:"
            f" {', '.join(formats)}.",
        )
