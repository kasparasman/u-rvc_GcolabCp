from typing import TYPE_CHECKING

import concurrent.futures
import glob
import json
import logging
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import tqdm

import torch
import torchcrepe

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
import ultimate_rvc.rvc.lib.zluda
from ultimate_rvc.common import RVC_MODELS_DIR, lazy_import
from ultimate_rvc.rvc.configs.config import Config
from ultimate_rvc.rvc.lib.predictors.RMVPE import RMVPE0Predictor
from ultimate_rvc.rvc.lib.utils import load_audio, load_embedding

if TYPE_CHECKING:
    import static_sox.run as static_sox_run
else:
    static_sox_run = lazy_import("static_sox.run")


logger = logging.getLogger(__name__)

# Load config
config = Config()
mp.set_start_method("spawn", force=True)


class FeatureInput:
    """Class for F0 extraction."""

    def __init__(self, sample_rate=16000, hop_size=160, device="cpu"):
        self.fs = sample_rate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        self.model_rmvpe = None

    def compute_f0(self, np_arr, f0_method, hop_length):
        """Extract F0 using the specified method."""
        if f0_method == "crepe":
            return self.get_crepe(np_arr, hop_length)
        if f0_method == "crepe-tiny":
            return self.get_crepe(np_arr, hop_length, model="tiny")
        if f0_method == "rmvpe":
            return self.model_rmvpe.infer_from_audio(np_arr, thred=0.03)
        raise ValueError(f"Unknown F0 method: {f0_method}")

    def get_crepe(self, x, hop_length, model="full"):
        """Extract F0 using CREPE."""
        audio = torch.from_numpy(x.astype(np.float32)).to(self.device)
        audio /= torch.quantile(torch.abs(audio), 0.999)
        audio = audio.unsqueeze(0)
        pitch = torchcrepe.predict(
            audio,
            self.fs,
            hop_length,
            self.f0_min,
            self.f0_max,
            model,
            batch_size=hop_length * 2,
            device=audio.device,
            pad=True,
        )
        source = pitch.squeeze(0).cpu().float().numpy()
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * (x.size // self.hop), len(source))
            / (x.size // self.hop),
            np.arange(0, len(source)),
            source,
        )
        return np.nan_to_num(target)

    def coarse_f0(self, f0):
        """Convert F0 to coarse F0."""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel = np.clip(
            (f0_mel - self.f0_mel_min)
            * (self.f0_bin - 2)
            / (self.f0_mel_max - self.f0_mel_min)
            + 1,
            1,
            self.f0_bin - 1,
        )
        return np.rint(f0_mel).astype(int)

    def process_file(self, file_info, f0_method, hop_length):
        """Process a single audio file for F0 extraction."""
        inp_path, opt_path1, opt_path2, _ = file_info

        if os.path.exists(opt_path1) and os.path.exists(opt_path2):
            return

        try:
            np_arr = load_audio(inp_path, 16000)
            feature_pit = self.compute_f0(np_arr, f0_method, hop_length)
            np.save(opt_path2, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path1, coarse_pit, allow_pickle=False)
        except Exception as error:
            logger.error(  # noqa: TRY400
                "An error occurred extracting file %s on %s: %s",
                inp_path,
                self.device,
                error,
            )

    def process_files(
        self,
        files,
        f0_method,
        hop_length,
        device_num,
        device,
        n_threads,
    ):
        """Process multiple files."""
        self.device = device
        if f0_method == "rmvpe":
            self.model_rmvpe = RMVPE0Predictor(
                os.path.join(str(RVC_MODELS_DIR), "predictors", "rmvpe.pt"),
                is_half=False,
                device=device,
            )
        else:
            n_threads = 1

        n_threads = 1 if n_threads == 0 else n_threads

        def process_file_wrapper(file_info):
            self.process_file(file_info, f0_method, hop_length)

        with tqdm.tqdm(total=len(files), leave=True, position=device_num) as pbar:
            # using multi-threading
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_threads,
            ) as executor:
                futures = [
                    executor.submit(process_file_wrapper, file_info)
                    for file_info in files
                ]
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)


def remove_from_ld_preload(prefix: str) -> None:
    """
    Remove entries from the LD_PRELOAD environment variable that start
    with the given prefix.

    Parameters
    ----------
    prefix : str
        The prefix to match entries in LD_PRELOAD.

    """
    # Get the current LD_PRELOAD value
    ld_preload = os.environ.get("LD_PRELOAD", "")

    # Split the LD_PRELOAD into a list of entries
    preload_entries = ld_preload.split(os.pathsep)

    # Remove the entries that start with the given prefix
    preload_entries = [
        entry for entry in preload_entries if not entry.startswith(prefix)
    ]

    # Join the list back into a string and update LD_PRELOAD
    os.environ["LD_PRELOAD"] = os.pathsep.join(preload_entries)


def run_pitch_extraction(
    files: list[list[str]],
    devices: list[str],
    f0_method: str,
    hop_length: int,
    num_processes: int,
) -> None:
    devices_str = ", ".join(devices)
    logger.info(
        "Starting pitch extraction with %d cores on %s using %s...",
        num_processes,
        devices_str,
        f0_method,
    )
    start_time = time.time()
    fe = FeatureInput()
    # split the task between devices
    ps = []
    num_devices = len(devices)

    # NOTE On ubuntu 24.04 the static_sox module does not work with
    # multiprocessing using the spawn method due to a "version
    # `GLIBC_2.38' not found" error. This is a workaround, which removes
    # the path to the libm.so.6 library from the LD_PRELOAD environment
    # variable.
    sox_exe = static_sox_run.get_or_fetch_platform_executables_else_raise()
    remove_from_ld_preload(os.path.join(os.path.dirname(sox_exe), "libm.so.6"))

    for i, device in enumerate(devices):
        p = mp.Process(
            target=fe.process_files,
            args=(
                files[i::num_devices],
                f0_method,
                hop_length,
                i,
                device,
                num_processes // num_devices,
            ),
        )
        ps.append(p)
        p.start()
    for i, device in enumerate(devices):
        ps[i].join()

    elapsed_time = time.time() - start_time
    logger.info("Pitch extraction completed in %.2f seconds.", elapsed_time)


def process_file_embedding(
    files,
    version,
    embedder_model,
    embedder_model_custom,
    device_num,
    device,
    n_threads,
):
    dtype = torch.float16 if config.is_half and "cuda" in device else torch.float32
    model = load_embedding(embedder_model, embedder_model_custom).to(dtype).to(device)
    n_threads = 1 if n_threads == 0 else n_threads

    def process_file_embedding_wrapper(file_info):
        wav_file_path, _, _, out_file_path = file_info
        if os.path.exists(out_file_path):
            return
        feats = torch.from_numpy(load_audio(wav_file_path, 16000)).to(dtype).to(device)
        feats = feats.view(1, -1)
        with torch.no_grad():
            feats = model(feats)["last_hidden_state"]
            feats = (
                model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
            )
        feats = feats.squeeze(0).float().cpu().numpy()
        if not np.isnan(feats).any():
            np.save(out_file_path, feats, allow_pickle=False)
        else:
            logger.error("%s contains NaN values and will be skipped.", wav_file_path)

    with tqdm.tqdm(total=len(files), leave=True, position=device_num) as pbar:
        # using multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(process_file_embedding_wrapper, file_info)
                for file_info in files
            ]
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)


def run_embedding_extraction(
    files: list[list[str]],
    devices: list[str],
    version: str,
    embedder_model: str,
    embedder_model_custom: str | None,
    num_processes: int,
) -> None:
    start_time = time.time()
    devices_str = ", ".join(devices)
    logger.info(
        "Starting embedding extraction with %d cores on %s...",
        num_processes,
        devices_str,
    )
    # split the task between devices
    ps = []
    num_devices = len(devices)
    for i, device in enumerate(devices):
        p = mp.Process(
            target=process_file_embedding,
            args=(
                files[i::num_devices],
                version,
                embedder_model,
                embedder_model_custom,
                i,
                device,
                num_processes // num_devices,
            ),
        )
        ps.append(p)
        p.start()
    for i, device in enumerate(devices):
        ps[i].join()
    elapsed_time = time.time() - start_time
    logger.info("Embedding extraction completed in %.2f seconds.", elapsed_time)


def initialize_extraction(
    exp_dir: str,
    version: str,
    f0_method: str,
    embedder_model: str,
) -> list[list[str]]:
    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    os.makedirs(os.path.join(exp_dir, f"f0_{f0_method}"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, f"f0_{f0_method}_voiced"), exist_ok=True)
    os.makedirs(
        os.path.join(exp_dir, f"{version}_{embedder_model}_extracted"),
        exist_ok=True,
    )

    files: list[list[str]] = []
    for file in glob.glob(os.path.join(wav_path, "*.wav")):
        file_name = os.path.basename(file)
        file_info = [
            file,  # full path to sliced 16k wav
            os.path.join(exp_dir, f"f0_{f0_method}_voiced", file_name + ".npy"),
            os.path.join(exp_dir, f"f0_{f0_method}", file_name + ".npy"),
            os.path.join(
                exp_dir,
                f"{version}_{embedder_model}_extracted",
                file_name.replace("wav", "npy"),
            ),
        ]
        files.append(file_info)

    return files


def update_model_info(exp_dir: str, embedder_model: str) -> None:
    file_path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path) as f:
            data = json.load(f)
    else:
        data = {}
    data.update(
        {
            "embedder_model": embedder_model,
        },
    )
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
