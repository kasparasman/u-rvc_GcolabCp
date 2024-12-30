from ultimate_rvc.typing_extra import (
    RVCVersion,
    StrPath,
    TrainingIndexAlgorithm,
    TrainingSampleRate,
    TrainingVocoder,
)


def train(
    model_name: str,
    rvc_version: RVCVersion = RVCVersion.V2,
    vocoder: TrainingVocoder = TrainingVocoder.HIFI_GAN,
    index_algorithm: TrainingIndexAlgorithm = TrainingIndexAlgorithm.AUTO,
    sample_rate: TrainingSampleRate = TrainingSampleRate.HZ_40K,
    num_epochs: int = 500,
    batch_size: int = 8,
    save_interval: int = 10,
    save_all_checkpoints: bool = True,
    save_all_weights: bool = True,
    enable_checkpointing: bool = False,
    use_pretrained: bool = True,
    use_custom_pretrained: bool = False,
    pretrained_generator: StrPath | None = None,
    pretrained_discriminator: StrPath | None = None,
    detect_overtraining: bool = False,
    overtraining_threshold: int = 50,
    reset_training_data: bool = False,
    preload_dataset: bool = False,
    gpus: set[int] | None = None,
):
    """

    Train a voice model using its associated set of preprocessed dataset audio files and extracted features.

    Parameters
    ----------
    model_name : str
        The name of the model to train.
    rvc_version : RVCVersion, default=RVCVersion.V2
        The version of the RVC model architecture to train.
    vocoder : TrainingVocoder, default=TrainingVocoder.HIFI_GAN
        The vocoder to use for audio synthesis during training.
    index_algorithm : TrainingIndexAlgorithm, default=TrainingIndexAlgorithm.AUTO
        The indexing algorithm to use for training the model.

    """
