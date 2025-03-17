import gc
import os
import re
import sys
import numpy as np
from scipy import signal
import faiss
import torch
import torch.nn.functional as F
from torch import Tensor
import librosa
import logging

now_dir = os.getcwd()
sys.path.append(now_dir)

from ultimate_rvc.common import RVC_MODELS_DIR, lazy_import

# Lazy-import torchcrepe if available.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torchcrepe
else:
    torchcrepe = lazy_import("torchcrepe")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure debug messages are output.

# Constants for high-pass filter
FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48  # Hz
SAMPLE_RATE = 16000  # Hz
bh, ah = signal.butter(
    N=FILTER_ORDER,
    Wn=CUTOFF_FREQUENCY,
    btype="high",
    fs=SAMPLE_RATE,
)

input_audio_path2wav = {}

class AudioProcessor:
    """
    A class for processing audio signals, specifically for adjusting RMS levels.
    """
    @staticmethod
    def change_rms(source_audio: np.ndarray, source_rate: int, target_audio: np.ndarray, target_rate: int, rate: float) -> np.ndarray:
        logger.debug("Changing RMS: source_rate=%d, target_rate=%d, rate=%.3f", source_rate, target_rate, rate)
        # Calculate RMS of both audio data
        rms1 = librosa.feature.rms(
            y=source_audio,
            frame_length=source_rate // 2 * 2,
            hop_length=source_rate // 2,
        )
        rms2 = librosa.feature.rms(
            y=target_audio,
            frame_length=target_rate // 2 * 2,
            hop_length=target_rate // 2,
        )
        logger.debug("Calculated RMS shapes: rms1=%s, rms2=%s", rms1.shape, rms2.shape)
        # Interpolate RMS to match target audio length
        rms1 = F.interpolate(
            torch.from_numpy(rms1).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = F.interpolate(
            torch.from_numpy(rms2).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)
        logger.debug("Interpolated RMS computed.")
        # Adjust target audio RMS based on the source audio RMS
        adjusted_audio = target_audio * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        logger.debug("RMS adjustment done.")
        return adjusted_audio

class Autotune:
    """
    A class for applying autotune to a given fundamental frequency (F0) contour.
    """
    def __init__(self, ref_freqs):
        self.ref_freqs = ref_freqs
        self.note_dict = self.ref_freqs  # No interpolation needed
        logger.debug("Autotune initialized with %d reference frequencies.", len(ref_freqs))

    def autotune_f0(self, f0, f0_autotune_strength):
        logger.debug("Autotuning F0 with strength %.3f", f0_autotune_strength)
        autotuned_f0 = np.zeros_like(f0)
        for i, freq in enumerate(f0):
            closest_note = min(self.note_dict, key=lambda x: abs(x - freq))
            autotuned_f0[i] = freq + (closest_note - freq) * f0_autotune_strength
        return autotuned_f0

class Pipeline:
    """
    The main pipeline class for performing voice conversion, including preprocessing, F0 estimation,
    voice conversion using a model, and post-processing.
    """
    def __init__(self, tgt_sr, config):
        logger.info("Initializing Pipeline with target sampling rate %d", tgt_sr)
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = config.device
        logger.debug("Pipeline device set to: %s", self.device)
        self.ref_freqs = [  # List of reference frequencies.
            49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78,
            82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81,
            138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65,
            220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63,
            349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25,
            554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61,
            880.00, 932.33, 987.77, 1046.50,
        ]
        self.autotune = Autotune(self.ref_freqs)
        self.note_dict = self.autotune.note_dict
        logger.info("Loading RMVPE model for F0 estimation...")
        from ultimate_rvc.rvc.lib.predictors.RMVPE import RMVPE0Predictor  # local import
        rmvpe_model_path = os.path.join(str(RVC_MODELS_DIR), "predictors", "rmvpe.pt")
        self.model_rmvpe = RMVPE0Predictor(rmvpe_model_path, device=self.device)
        logger.info("RMVPE model loaded.")

    def get_f0_crepe(self, x, f0_min, f0_max, p_len, hop_length, model="full"):
        logger.debug("Running Crepe-based F0 estimation: model=%s", model)
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        audio = torch.from_numpy(x).to(self.device, copy=True).unsqueeze(0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        pitch: Tensor = torchcrepe.predict(
            audio,
            self.sample_rate,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=hop_length * 2,
            device=self.device,
            pad=True,
        )
        p_len = p_len or x.shape[0] // hop_length
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0 = np.nan_to_num(target)
        logger.debug("Crepe F0 estimated with shape: %s", f0.shape)
        return f0

    def get_f0_hybrid(self, methods, x, f0_min, f0_max, p_len, hop_length):
        logger.info("Calculating F0 with hybrid methods: %s", ", ".join(methods))
        f0_computation_stack = []
        for method in methods:
            f0 = None
            if method == "crepe":
                f0 = self.get_f0_crepe(x, f0_min, f0_max, p_len, int(hop_length))
            elif method == "crepe-tiny":
                f0 = self.get_f0_crepe(x, self.f0_min, self.f0_max, p_len, int(hop_length), "tiny")
            elif method == "rmvpe":
                f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)[1:]
            elif method == "fcpe":
                from ultimate_rvc.rvc.lib.predictors.FCPE import FCPEF0Predictor
                self.model_fcpe = FCPEF0Predictor(
                    os.path.join(str(RVC_MODELS_DIR), "predictors", "fcpe.pt"),
                    f0_min=int(f0_min),
                    f0_max=int(f0_max),
                    dtype=torch.float32,
                    device=self.device,
                    sample_rate=self.sample_rate,
                    threshold=0.03,
                )
                f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
                del self.model_fcpe
                gc.collect()
            if f0 is not None:
                f0_computation_stack.append(f0)
        if len(f0_computation_stack) == 0:
            logger.error("No valid F0 estimation method returned results.")
            return None
        f0_median_hybrid = f0_computation_stack[0] if len(f0_computation_stack) == 1 else np.nanmedian(f0_computation_stack, axis=0)
        logger.debug("Hybrid F0 computed with shape: %s", f0_median_hybrid.shape)
        return f0_median_hybrid

    def get_f0(self, input_audio_path, x, p_len, pitch, f0_methods, filter_radius, hop_length, f0_autotune, f0_autotune_strength, inp_f0=None):
        logger.info("Estimating F0 for input: %s", input_audio_path)
        f0 = self.get_f0_hybrid(f0_methods, x, self.f0_min, self.f0_max, p_len, hop_length)
        if f0_autotune:
            logger.debug("Applying autotune to F0.")
            f0 = self.autotune.autotune_f0(f0, f0_autotune_strength)
        f0 *= pow(2, pitch / 12)
        tf0 = self.sample_rate // self.window
        if inp_f0 is not None:
            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)
        logger.debug("F0 estimation complete: coarse shape %s, original shape %s", f0_coarse.shape, f0bak.shape)
        return f0_coarse, f0bak

    def voice_conversion(self, model, net_g, sid, audio0, pitch, pitchf, index, big_npy, index_rate, version, protect):
        logger.info("Starting voice conversion for segment, protect=%.2f, index_rate=%.2f", protect, index_rate)
        with torch.no_grad():
            pitch_guidance = (pitch is not None) and (pitchf is not None)
            feats = torch.from_numpy(audio0).float()
            if feats.dim() == 2:
                feats = feats.mean(-1)
            assert feats.dim() == 1, f"Expected feats dim 1 but got {feats.dim()}"
            feats = feats.view(1, -1).to(self.device)
            logger.debug("Extracting features from audio segment: feats shape %s", feats.shape)
            feats = model(feats)["last_hidden_state"]
            if version == "v1":
                feats = model.final_proj(feats[0]).unsqueeze(0)
            feats0 = feats.clone() if pitch_guidance else None
            if index:
                feats = self._retrieve_speaker_embeddings(feats, index, big_npy, index_rate)
                logger.debug("Retrieved speaker embeddings, new feats shape: %s", feats.shape)
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            p_len = min(audio0.shape[0] // self.window, feats.shape[1])
            logger.debug("After upsampling: feats shape %s, p_len=%d", feats.shape, p_len)
            if pitch_guidance:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
                pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
                if protect < 0.5:
                    pitchff = pitchf.clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    feats = feats * pitchff.unsqueeze(-1) + feats0 * (1 - pitchff.unsqueeze(-1))
                    feats = feats.to(feats0.dtype)
            else:
                pitch, pitchf = None, None
            p_len = torch.tensor([p_len], device=self.device).long()
            logger.debug("Calling synthesizer net_g.infer with p_len=%s", p_len)
            audio1 = (net_g.infer(feats.float(), p_len, pitch, pitchf.float() if pitchf is not None else None, sid)[0][0, 0]).data.cpu().float().numpy()
            logger.info("Voice conversion for segment complete, output audio shape: %s", audio1.shape)
            del feats, feats0, p_len
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return audio1

    def _retrieve_speaker_embeddings(self, feats, index, big_npy, index_rate):
        logger.debug("Retrieving speaker embeddings from FAISS index...")
        npy = feats[0].cpu().numpy()
        score, ix = index.search(npy, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats
        logger.debug("Speaker embeddings retrieved, feats shape now: %s", feats.shape)
        return feats

    def pipeline(self, model, net_g, sid, audio, pitch, f0_methods, file_index, index_rate, pitch_guidance, filter_radius, volume_envelope, version, protect, hop_length, f0_autotune, f0_autotune_strength, f0_file):
        logger.info("Starting full voice conversion pipeline.")
        if file_index != "" and os.path.exists(file_index) and index_rate > 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
                logger.debug("FAISS index loaded, index.ntotal=%d", index.ntotal)
            except Exception as error:
                logger.error("Error reading FAISS index: %s", error)
                index = big_npy = None
        else:
            index = big_npy = None
            logger.debug("No FAISS index provided or index_rate=0.")
        audio = signal.filtfilt(bh, ah, audio)
        logger.debug("Applied high-pass filtering to audio.")
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        logger.debug("Padded audio, new shape: %s", audio_pad.shape)
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i: i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                local_region = audio_sum[t - self.t_query: t + self.t_query]
                min_idx = np.where(np.abs(local_region) == np.abs(local_region).min())[0][0]
                opt_ts.append(t - self.t_query + min_idx)
            logger.debug("Calculated optimal time segments: %s", opt_ts)
        s = 0
        audio_opt = []
        t = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name) as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
                logger.debug("Loaded external F0 file with shape: %s", inp_f0.shape)
            except Exception as error:
                logger.error("Error reading F0 file: %s", error)
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        if pitch_guidance:
            pitch, pitchf = self.get_f0("input_audio_path", audio_pad, p_len, pitch, f0_methods, filter_radius, hop_length, f0_autotune, f0_autotune_strength, inp_f0)
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
            logger.debug("Pitch guidance obtained, pitch shape: %s, pitchf shape: %s", pitch.shape, pitchf.shape)
        for t in opt_ts:
            t = t // self.window * self.window
            if pitch_guidance:
                seg_audio = audio_pad[s: t + self.t_pad2 + self.window]
                seg_pitch = pitch[:, s // self.window: (t + self.t_pad2) // self.window]
                seg_pitchf = pitchf[:, s // self.window: (t + self.t_pad2) // self.window]
                segment = self.voice_conversion(model, net_g, sid, seg_audio, seg_pitch, seg_pitchf, index, big_npy, index_rate, version, protect)[self.t_pad_tgt: -self.t_pad_tgt]
                logger.debug("Processed segment with pitch guidance, segment shape: %s", segment.shape)
                audio_opt.append(segment)
            else:
                segment = self.voice_conversion(model, net_g, sid, audio_pad[s: t + self.t_pad2 + self.window], None, None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt: -self.t_pad_tgt]
                logger.debug("Processed segment without pitch guidance, segment shape: %s", segment.shape)
                audio_opt.append(segment)
            s = t
        if pitch_guidance:
            final_segment = self.voice_conversion(model, net_g, sid, audio_pad[t:], pitch[:, t // self.window:] if t is not None else pitch, pitchf[:, t // self.window:] if t is not None else pitchf, index, big_npy, index_rate, version, protect)[self.t_pad_tgt: -self.t_pad_tgt]
            audio_opt.append(final_segment)
        else:
            final_segment = self.voice_conversion(model, net_g, sid, audio_pad[t:], None, None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt: -self.t_pad_tgt]
            audio_opt.append(final_segment)
        audio_opt = np.concatenate(audio_opt)
        logger.info("Concatenated output audio shape: %s", audio_opt.shape)
        if volume_envelope != 1:
            audio_opt = AudioProcessor.change_rms(audio, self.sample_rate, audio_opt, self.sample_rate, volume_envelope)
            logger.debug("Applied volume envelope adjustment.")
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1:
            audio_opt /= audio_max
            logger.debug("Normalized output audio.")
        if pitch_guidance:
            del pitch, pitchf
        del sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Pipeline processing complete.")
        return audio_opt
