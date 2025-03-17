import ultimate_rvc.rvc.infer.logger_config
import logging
logger = logging.getLogger(__name__)
import os
import re
import sys
import unicodedata
import warnings

import soxr

import wget

import numpy as np

from torch import nn
from transformers import HubertModel

import librosa
import soundfile as sf

from ultimate_rvc.common import RVC_MODELS_DIR

# Remove this to see warnings about transformers models
warnings.filterwarnings("ignore")


now_dir = os.getcwd()
sys.path.append(now_dir)

base_path = os.path.join(str(RVC_MODELS_DIR), "formant", "stftpitchshift")
stft = base_path + ".exe" if sys.platform == "win32" else base_path


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=sample_rate,
                res_type="soxr_vhq",
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio_infer(
    file,
    sample_rate,
    **kwargs,
):
    formant_shifting = kwargs.get("formant_shifting", False)
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File not found: {file}")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=sample_rate,
                res_type="soxr_vhq",
            )
        if formant_shifting:
            formant_qfrency = kwargs.get("formant_qfrency", 0.8)
            formant_timbre = kwargs.get("formant_timbre", 0.8)

            from stftpitchshift import StftPitchShift

            pitchshifter = StftPitchShift(1024, 32, sample_rate)
            audio = pitchshifter.shiftpitch(
                audio,
                factors=1,
                quefrency=formant_qfrency * 1e-3,
                distortion=formant_timbre,
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")
    return np.array(audio).flatten()


def format_title(title):
    formatted_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("utf-8")
    )
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def load_embedding(embedder_model, custom_embedder=None):
    import os
    
    # If embedder_model is "contentvec", use the custom folder path.
    if embedder_model.lower() == "contentvec":
        model_path = r"C:\Users\Kasparas\argos_tts\Main_RVC\u-rvc_GcolabCp\src\ultimate_rvc\rvc\infer\models\rvc\embedders\contentvec"
    else:
        raise ValueError(f"Embedder model '{embedder_model}' not supported. Only 'contentvec' is supported.")
    
    bin_file = os.path.join(model_path, "pytorch_model.bin")
    json_file = os.path.join(model_path, "config.json")
    os.makedirs(model_path, exist_ok=True)
    if not os.path.exists(bin_file) or not os.path.exists(json_file):
        raise FileNotFoundError(f"Required embedder files not found in {model_path}")

    models = HubertModelWithFinalProj.from_pretrained(model_path)
    return models
