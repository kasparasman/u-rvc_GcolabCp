import os

from ultimate_rvc.common import PRETRAINED_MODELS_DIR


def pretrained_selector(
    version: str,
    pitch_guidance: bool,
    sample_rate: int,
) -> tuple[str, str]:
    base_path = os.path.join(PRETRAINED_MODELS_DIR, f"pretrained_{version}")
    f0 = "f0" if pitch_guidance else ""

    path_g = os.path.join(base_path, f"{f0}G{str(sample_rate)[:2]}k.pth")
    path_d = os.path.join(base_path, f"{f0}D{str(sample_rate)[:2]}k.pth")

    if os.path.exists(path_g) and os.path.exists(path_d):
        return path_g, path_d
    return "", ""
