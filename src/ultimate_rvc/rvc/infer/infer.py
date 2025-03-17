import logger_config  # This will configure logging for the entire application
import logging
logger = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Unpack

import os
import sys
import time
import traceback

import soxr
import json
import numpy as np
print("Numpy version:", np.__version__)
import torch
print("Torch version:", torch.__version__)
import librosa
print("Librosa version:", librosa.__version__)
import soundfile as sf
print("Soundfile version:", sf.__version__)
from pedalboard import (
    Bitcrush,
    Chorus,
    Clipping,
    Compressor,
    Delay,
    Distortion,
    Gain,
    Limiter,
    Pedalboard,
    PitchShift,
    Reverb,
)
now_dir = os.getcwd()
sys.path.append(now_dir)

from ultimate_rvc.common import lazy_import
from ultimate_rvc.rvc.configs.config import Config
from ultimate_rvc.rvc.infer.pipeline import Pipeline as VC
from ultimate_rvc.rvc.infer.typing_extra import ConvertAudioKwArgs
from ultimate_rvc.rvc.lib.algorithm.synthesizers import Synthesizer
from ultimate_rvc.rvc.lib.tools.split_audio import merge_audio, process_audio
from ultimate_rvc.rvc.lib.utils import load_audio_infer, load_embedding
from ultimate_rvc.typing_extra import F0Method

if TYPE_CHECKING:
    import noisereduce as nr
else:
    nr = lazy_import("noisereduce")

# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)
# logging.getLogger("faiss").setLevel(logging.WARNING)
# logging.getLogger("faiss.loader").setLevel(logging.WARNING)


class VoiceConverter:
    """
    A class for performing voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self):
        """
        Initializes the VoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load configuration
        self.hubert_model = (
            None  # Initialize the Hubert model (for embedding extraction)
        )
        self.last_embedder_model = None  # Last used embedder model
        self.tgt_sr = None  # Target sampling rate for the output audio
        self.net_g = None  # Generator network for voice conversion
        self.vc = None  # Voice conversion pipeline instance
        self.cpt = None  # Checkpoint for loading model weights
        self.version = None  # Model version
        self.n_spk = None  # Number of speakers in the model
        self.use_f0 = None  # Whether the model uses F0
        self.loaded_model = None

    def load_hubert(self, embedder_model: str, embedder_model_custom: str = None):
        """
        Loads the HuBERT model for speaker embedding extraction.

        Args:
            embedder_model (str): Path to the pre-trained HuBERT model.
            embedder_model_custom (str): Path to the custom HuBERT model.

        """
        self.hubert_model = load_embedding(embedder_model, embedder_model_custom)
        self.hubert_model = self.hubert_model.to(self.config.device).float()
        self.hubert_model.eval()

    @staticmethod
    def remove_audio_noise(data, sr, reduction_strength=0.7):
        """
        Removes noise from an audio file using the NoiseReduce library.

        Args:
            data (numpy.ndarray): The audio data as a NumPy array.
            sr (int): The sample rate of the audio data.
            reduction_strength (float): Strength of the noise reduction. Default is 0.7.

        """
        try:

            reduced_noise = nr.reduce_noise(
                y=data,
                sr=sr,
                prop_decrease=reduction_strength,
            )
            return reduced_noise
        except Exception as error:
            print(f"An error occurred removing audio noise: {error}")
            return None

    @staticmethod
    def convert_audio_format(input_path, output_path, output_format):
        """
        Converts an audio file to a specified output format.

        Args:
            input_path (str): Path to the input audio file.
            output_path (str): Path to the output audio file.
            output_format (str): Desired audio format (e.g., "WAV", "MP3").

        """
        try:
            if output_format != "WAV":
                print(f"Saving audio as {output_format}...")
                audio, sample_rate = librosa.load(input_path, sr=None)
                common_sample_rates = [
                    8000,
                    11025,
                    12000,
                    16000,
                    22050,
                    24000,
                    32000,
                    44100,
                    48000,
                ]
                target_sr = min(common_sample_rates, key=lambda x: abs(x - sample_rate))
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=target_sr,
                    res_type="soxr_vhq",
                )
                sf.write(output_path, audio, target_sr, format=output_format.lower())
            return output_path
        except Exception as error:
            print(f"An error occurred converting the audio format: {error}")

    @staticmethod
    def post_process_audio(
        audio_input,
        sample_rate,
        **kwargs,
    ):
        board = Pedalboard()
        if kwargs.get("reverb"):
            reverb = Reverb(
                room_size=kwargs.get("reverb_room_size", 0.5),
                damping=kwargs.get("reverb_damping", 0.5),
                wet_level=kwargs.get("reverb_wet_level", 0.33),
                dry_level=kwargs.get("reverb_dry_level", 0.4),
                width=kwargs.get("reverb_width", 1.0),
                freeze_mode=kwargs.get("reverb_freeze_mode", 0),
            )
            board.append(reverb)
        if kwargs.get("pitch_shift"):
            pitch_shift = PitchShift(semitones=kwargs.get("pitch_shift_semitones", 0))
            board.append(pitch_shift)
        if kwargs.get("limiter"):
            limiter = Limiter(
                threshold_db=kwargs.get("limiter_threshold", -6),
                release_ms=kwargs.get("limiter_release", 0.05),
            )
            board.append(limiter)
        if kwargs.get("gain"):
            gain = Gain(gain_db=kwargs.get("gain_db", 0))
            board.append(gain)
        if kwargs.get("distortion"):
            distortion = Distortion(drive_db=kwargs.get("distortion_gain", 25))
            board.append(distortion)
        if kwargs.get("chorus"):
            chorus = Chorus(
                rate_hz=kwargs.get("chorus_rate", 1.0),
                depth=kwargs.get("chorus_depth", 0.25),
                centre_delay_ms=kwargs.get("chorus_delay", 7),
                feedback=kwargs.get("chorus_feedback", 0.0),
                mix=kwargs.get("chorus_mix", 0.5),
            )
            board.append(chorus)
        if kwargs.get("bitcrush"):
            bitcrush = Bitcrush(bit_depth=kwargs.get("bitcrush_bit_depth", 8))
            board.append(bitcrush)
        if kwargs.get("clipping"):
            clipping = Clipping(threshold_db=kwargs.get("clipping_threshold", 0))
            board.append(clipping)
        if kwargs.get("compressor"):
            compressor = Compressor(
                threshold_db=kwargs.get("compressor_threshold", 0),
                ratio=kwargs.get("compressor_ratio", 1),
                attack_ms=kwargs.get("compressor_attack", 1.0),
                release_ms=kwargs.get("compressor_release", 100),
            )
            board.append(compressor)
        if kwargs.get("delay"):
            delay = Delay(
                delay_seconds=kwargs.get("delay_seconds", 0.5),
                feedback=kwargs.get("delay_feedback", 0.0),
                mix=kwargs.get("delay_mix", 0.5),
            )
            board.append(delay)
        return board(audio_input, sample_rate)
    def convert_audio(
        self,
        audio_input_path: str,
        audio_output_path: str,
        model_path: str,
        index_path: str,
        pitch: int = 0,
        f0_file: str | None = None,
        f0_methods: set[F0Method] | None = None,
        index_rate: float = 1.0,
        volume_envelope: float = 1,
        protect: float = 0.5,
        hop_length: int = 128,
        split_audio: bool = False,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        filter_radius: int = 3,
        embedder_model: str = "contentvec",
        embedder_model_custom: str | None = None,
        clean_audio: bool = False,
        clean_strength: float = 0.5,
        export_format: str = "WAV",
        upscale_audio: bool = False,
        post_process: bool = False,
        resample_sr: int = 0,
        sid: int = 0,
        **kwargs: Unpack[ConvertAudioKwArgs],
    ) -> str:
        """
        Performs voice conversion on the input audio.
        Args:
            pitch (int): Key for F0 up-sampling.
            filter_radius (int): Radius for filtering.
            index_rate (float): Rate for index matching.
            volume_envelope (int): RMS mix rate.
            protect (float): Protection rate for certain audio segments.
            hop_length (int): Hop length for audio processing.
            f0_methods (set[F0Method], optional): Methods for F0 extraction.
            audio_input_path (str): Path to the input audio file.
            audio_output_path (str): Path to the output audio file.
            model_path (str): Path to the voice conversion model.
            index_path (str): Path to the index file.
            split_audio (bool): Whether to split the audio for processing.
            f0_autotune (bool): Whether to use F0 autotune.
            clean_audio (bool): Whether to clean the audio.
            clean_strength (float): Strength of the audio cleaning.
            export_format (str): Format for exporting the audio.
            upscale_audio (bool): Whether to upscale the audio.
            f0_file (str): Path to the F0 file.
            embedder_model (str): Path to the embedder model.
            embedder_model_custom (str): Path to the custom embedder model.
            resample_sr (int, optional): Resample sampling rate. Default is 0.
            sid (int, optional): Speaker ID. Default is 0.
            **kwargs: Additional keyword arguments.
        Returns:
            str: The path to the converted audio file.
        """
        logger.debug("Starting convert_audio with parameters:")
        logger.debug("  audio_input_path: %s", audio_input_path)
        logger.debug("  audio_output_path: %s", audio_output_path)
        logger.debug("  model_path: %s", model_path)
        logger.debug("  index_path: %s", index_path)
        logger.debug("  pitch: %d, f0_methods: %s, index_rate: %.2f", pitch, f0_methods, index_rate)
        logger.debug("  split_audio: %s, f0_autotune: %s, clean_audio: %s, post_process: %s",
                     split_audio, f0_autotune, clean_audio, post_process)
        logger.debug("  Additional kwargs: %s", kwargs)
        self.get_vc(model_path, sid)
        start_time = time.time()
        logger.info("Converting audio '%s'...", audio_input_path)
        try:
            audio = load_audio_infer(
                audio_input_path,
                16000,
                **kwargs,
            )
            logger.info("Audio loaded. Shape: %s, max amplitude: %f", audio.shape, np.max(np.abs(audio)))
        except Exception as e:
            logger.exception("Failed to load audio from '%s': %s", audio_input_path, e)
            raise
        logger.info("Audio loaded. Shape: %s, max amplitude: %f", audio.shape, np.max(np.abs(audio)))
        audio_max = np.abs(audio).max() / 0.95
        logger.info("Computed normalization factor: %f", audio_max)
        if audio_max > 1:
            audio /= audio_max
            logger.info("Audio normalized.")
        else:
            logger.info("Audio normalization not required.")
        if not self.hubert_model or embedder_model != self.last_embedder_model:
            logger.info("Loading embedder model: %s", embedder_model)
            self.load_hubert(embedder_model)
            self.last_embedder_model = embedder_model
            logger.info("Embedder model loaded.")

        file_index = (
            index_path.strip()
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip()
            .replace("trained", "added")
        )
        logger.info("Processed file index: %s", file_index)

        if self.tgt_sr != resample_sr >= 16000:
            self.tgt_sr = resample_sr
            logger.info("Updated target sampling rate: %s", self.tgt_sr)

        if split_audio:
            chunks, intervals = process_audio(audio, 16000)
            logger.info("Audio split into %d chunks for processing.", len(chunks))
        else:
            chunks = []
            chunks.append(audio)
            logger.info("Processing audio as a single chunk.")

        converted_chunks = []
        for i, c in enumerate(chunks):  # Use enumerate to get an index
            logger.info("Processing chunk %d with shape: %s", i, c.shape)
            logger.debug("vc attribute: %s", self.vc)
            audio_opt = self.vc.pipeline(
                model=self.hubert_model,
                net_g=self.net_g,
                sid=sid,
                audio=c,
                pitch=pitch,
                f0_methods=f0_methods or {F0Method.RMVPE},
                file_index=file_index,
                index_rate=index_rate,
                pitch_guidance=self.use_f0,
                filter_radius=filter_radius,
                volume_envelope=volume_envelope,
                version=self.version,
                protect=protect,
                hop_length=hop_length,
                f0_autotune=f0_autotune,
                f0_autotune_strength=f0_autotune_strength,
                f0_file=f0_file,
            )
            logger.info("Chunk %d conversion complete. Output shape: %s", i, audio_opt.shape)
            converted_chunks.append(audio_opt)
            if split_audio:
                logger.info("Converted audio chunk %d", len(converted_chunks))
        if split_audio:
            audio_opt = merge_audio(
                chunks,
                converted_chunks,
                intervals,
                16000,
                self.tgt_sr,
            )
        else:
            audio_opt = converted_chunks[0]
        if clean_audio:
            logger.info("Cleaning audio with strength: %f", clean_strength)
            cleaned_audio = self.remove_audio_noise(
                audio_opt,
                self.tgt_sr,
                clean_strength,
            )
            if cleaned_audio is not None:
                audio_opt = cleaned_audio
                logger.info("Audio cleaning applied.")
        if post_process:
            logger.info("Post-processing audio...")
            audio_opt = self.post_process_audio(
                audio_input=audio_opt,
                sample_rate=self.tgt_sr,
                **kwargs,
            )
            logger.info("Post-processing complete.")

        sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")
        logger.info("Output audio written to: %s", audio_output_path)
        output_path_format = audio_output_path.replace(
            ".wav",
            f".{export_format.lower()}",
        )
        audio_output_path = self.convert_audio_format(
            audio_output_path,
            output_path_format,
            export_format,
        )
        elapsed_time = time.time() - start_time
        logger.info(
            "Conversion completed at '%s' in %.2f seconds.",
            audio_output_path,
            elapsed_time,
        )
        return audio_output_path

    def convert_audio_batch(
        self,
        audio_input_paths: str,
        audio_output_path: str,
        **kwargs,
    ):
        """
        Performs voice conversion on a batch of input audio files.

        Args:
            audio_input_paths (str): List of paths to the input audio files.
            audio_output_path (str): Path to the output audio file.
            resample_sr (int, optional): Resample sampling rate. Default is 0.
            sid (int, optional): Speaker ID. Default is 0.
            **kwargs: Additional keyword arguments.

        """
        pid = os.getpid()
        try:
            with open(
                os.path.join(now_dir, "assets", "infer_pid.txt"),
                "w",
            ) as pid_file:
                pid_file.write(str(pid))
            start_time = time.time()
            print(f"Converting audio batch '{audio_input_paths}'...")
            audio_files = [
                f
                for f in os.listdir(audio_input_paths)
                if f.endswith(
                    (
                        "wav",
                        "mp3",
                        "flac",
                        "ogg",
                        "opus",
                        "m4a",
                        "mp4",
                        "aac",
                        "alac",
                        "wma",
                        "aiff",
                        "webm",
                        "ac3",
                    ),
                )
            ]
            print(f"Detected {len(audio_files)} audio files for inference.")
            for a in audio_files:
                new_input = os.path.join(audio_input_paths, a)
                new_output = os.path.splitext(a)[0] + "_output.wav"
                new_output = os.path.join(audio_output_path, new_output)
                if os.path.exists(new_output):
                    continue
                self.convert_audio(
                    audio_input_path=new_input,
                    audio_output_path=new_output,
                    **kwargs,
                )
            print(f"Conversion completed at '{audio_input_paths}'.")
            elapsed_time = time.time() - start_time
            print(f"Batch conversion completed in {elapsed_time:.2f} seconds.")
        except Exception as error:
            print(f"An error occurred during audio batch conversion: {error}")
            print(traceback.format_exc())
        finally:
            os.remove(os.path.join(now_dir, "assets", "infer_pid.txt"))

    def get_vc(self, weight_root, sid):
        """
        Loads the voice conversion model and sets up the pipeline.

        Args:
            weight_root (str): Path to the model weights.
            sid (int): Speaker ID.

        """
        if sid == "" or sid == []:
            self.cleanup_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("Loading model...", weight_root, sid)
        if not self.loaded_model or self.loaded_model != weight_root:
            self.load_model(weight_root)
            if self.cpt is not None:
                self.setup_network()
                self.setup_vc_instance()
            self.loaded_model = weight_root

    def cleanup_model(self):
        """
        Cleans up the model and releases resources.
        """
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del self.net_g, self.cpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cpt = None

    def load_model(self, weight_root):
        """
        Loads the model weights from the specified path.

        Args:
            weight_root (str): Path to the model weights.

        """
        self.cpt = (
            torch.load(weight_root, map_location="cpu", weights_only=False)
            if os.path.isfile(weight_root)
            else None
        )



    def setup_network(self):
        """
        Sets up the network configuration using the correct configuration from config.json,
        merging it with dynamic values from the loaded checkpoint.
        """
        if self.cpt is None:
            logger.error("Checkpoint (cpt) is None. Cannot setup network.")
            return

        logger.info("Starting network setup...")

        try:
            # Load the original configuration from config.json
            config_path = r"C:\Users\Kasparas\argos_tts\Main_RVC\u-rvc_GcolabCp\src\ultimate_rvc\rvc\infer\config.json"  # Update this path as needed.
            with open(config_path, "r") as f:
                full_config = json.load(f)
            data_config = full_config["data"]
            model_config = full_config["model"]
            train_config = full_config["train"]

            logger.debug("Loaded original config.json:")
            logger.debug("Data config: %s", data_config)
            logger.debug("Model config: %s", model_config)
            logger.debug("Train config: %s", train_config)

            # Map training parameters to the network's expected parameters.
            spec_channels = data_config["n_mel_channels"]             # e.g., 125
            segment_size = train_config["segment_size"]                # e.g., 12800
            inter_channels = model_config["inter_channels"]            # e.g., 192
            hidden_channels = model_config["hidden_channels"]          # e.g., 192
            filter_channels = model_config["filter_channels"]          # e.g., 768
            n_heads = model_config["n_heads"]                          # e.g., 2
            n_layers = model_config["n_layers"]                        # e.g., 6
            kernel_size = model_config["kernel_size"]                  # e.g., 3
            p_dropout = model_config["p_dropout"]                      # e.g., 0
            resblock = model_config["resblock"]                        # e.g., "1"
            resblock_kernel_sizes = model_config["resblock_kernel_sizes"]          # e.g., [3, 7, 11]
            resblock_dilation_sizes = model_config["resblock_dilation_sizes"]      # e.g., [[1,3,5],[1,3,5],[1,3,5]]
            upsample_rates = model_config["upsample_rates"]            # e.g., [10, 10, 2, 2]
            upsample_initial_channel = model_config["upsample_initial_channel"]  # e.g., 512
            upsample_kernel_sizes = model_config["upsample_kernel_sizes"]            # e.g., [16, 16, 4, 4]

            # Update speaker embedding dimension dynamically from the checkpoint weights
            spk_embed_dim = self.cpt["weight"]["emb_g.weight"].shape[0]
            gin_channels = model_config["gin_channels"]                # e.g., 256
            sr = data_config["sampling_rate"]                          # e.g., 40000

            logger.debug("Dynamic spk_embed_dim from weights: %d", spk_embed_dim)
            logger.debug("Using sampling rate (sr): %d", sr)

            # Set additional dynamic parameters from the checkpoint
            self.tgt_sr = sr
            self.use_f0 = bool(self.cpt.get("f0", 1))
            self.version = self.cpt.get("version", "v1")
            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")

            logger.info("Dynamic parameters: use_f0=%s, version=%s, text_enc_hidden_dim=%d, vocoder=%s",
                        self.use_f0, self.version, self.text_enc_hidden_dim, self.vocoder)

            # Construct the full configuration list in the expected order:
            # Expected order for Synthesizer:
            # spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
            # n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes,
            # resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
            # spk_embed_dim, gin_channels, sr, use_f0
            synthesizer_params = [
                spec_channels,         # spec_channels (n_mel_channels)
                segment_size,          # segment_size
                inter_channels,        # inter_channels
                hidden_channels,       # hidden_channels
                filter_channels,       # filter_channels
                n_heads,               # n_heads
                n_layers,              # n_layers
                kernel_size,           # kernel_size
                p_dropout,             # p_dropout
                resblock,              # resblock
                resblock_kernel_sizes, # resblock_kernel_sizes
                resblock_dilation_sizes,  # resblock_dilation_sizes
                upsample_rates,        # upsample_rates
                upsample_initial_channel,  # upsample_initial_channel
                upsample_kernel_sizes,     # upsample_kernel_sizes
                spk_embed_dim,         # spk_embed_dim (from weights)
                gin_channels,          # gin_channels
                sr                     # sampling rate
            ]

            logger.debug("Synthesizer parameters: %s", synthesizer_params)

            # Initialize the Synthesizer with the full set of parameters and additional keyword arguments.
            self.net_g = Synthesizer(
                *synthesizer_params,
                use_f0=self.use_f0,
                text_enc_hidden_dim=self.text_enc_hidden_dim,
                vocoder=self.vocoder
            )
            logger.info("Synthesizer network initialized successfully.")

            # Remove unused components to free memory.
            if hasattr(self.net_g, "enc_q"):
                del self.net_g.enc_q
                logger.debug("Removed enc_q from the network.")

            # Load the pre-trained weights
            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            logger.info("Model weights loaded successfully.")

            # Move the model to the configured device and set to eval mode.
            self.net_g = self.net_g.to(self.config.device).float()
            self.net_g.eval()
            logger.info("Synthesizer moved to device: %s and set to evaluation mode.", self.config.device)

        except Exception as e:
            logger.exception("Exception occurred during network setup: %s", e)

    def setup_vc_instance(self):
        """
        Sets up the voice conversion pipeline instance based on the target sampling rate and configuration.
        """
        if self.cpt is None:
            logger.error("Checkpoint (cpt) is None. Cannot setup voice conversion instance.")
            return

        try:
            logger.info("Initializing voice conversion pipeline instance (VC).")
            # Create the VC pipeline using the target sampling rate and overall configuration.
            self.vc = VC(self.tgt_sr, self.config)
            logger.debug("VC instance created with target sampling rate: %d and config: %s", self.tgt_sr, self.config)

            # Set the number of speakers based on the configuration (dynamic value from weights).
            self.n_spk = self.cpt["config"][-3]
            logger.info("Number of speakers set to: %d", self.n_spk)
        except Exception as e:
            logger.exception("Exception occurred during VC instance setup: %s", e)
