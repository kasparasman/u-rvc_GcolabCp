#!/usr/bin/env python
"""
Simplified Inference Script for RVC Voice Conversion

This script performs voice conversion using hardcoded settings.
It assumes the following files are in the same directory as this script:
  - Main model weights: "argos.pth"
  - Model index file: "argos.index"
  - RMVPE model for F0 extraction: "rmvpe.pt"
  - ContentVec embedder model: placed in folder "contentvec" (the custom embedder)
The configuration used for training/inference is loaded from config.json.
"""

import ultimate_rvc.rvc.infer.logger_config  # This configures logging for the entire application
import logging

logger = logging.getLogger(__name__)

import os
import time
import argparse
import torch

print("Torch version:", torch.__version__)

# Import configuration and common validation utilities
from ultimate_rvc.rvc.configs.config import Config         # From config.py
from ultimate_rvc.core.common import validate_model_exists   # Utility to check model existence
# Import the conversion function from the generate module.
# Adjust the import path if necessary.
from ultimate_rvc.core.generate.common import convert

# Import F0Method and other type definitions
from ultimate_rvc.typing_extra import F0Method, RVCContentType, EmbedderModel

# Set up directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENTVEC_DIR = os.path.join(BASE_DIR, "contentvec")

def run_inference(
    input_audio: str,
    output_audio: str,
    pitch: int = 0
) -> str:
    """
    Runs voice conversion inference by first calling the common convert function.
    
    Args:
        input_audio (str): Path to the input audio file.
        output_audio (str): Path where the converted audio will be saved.
        pitch (int, optional): Pitch shift (in semitones). Default is 0.
    
    Returns:
        str: Path to the converted audio file.
    """
    # Load global configuration (device, etc.)
    config = Config()
    print("Using device:", config.device)
    
    # Here, model_name is used as an identifier for the voice model.
    # Make sure that model_name corresponds to a directory or identifier expected by validate_model_exists.
    model_name = "argos"
    model_path = "C:\\Users\\Kasparas\\argos_tts\\Main_RVC\\u-rvc_GcolabCp\\src\\ultimate_rvc\\rvc\\infer\\argos.pth"
    print("Model exists:", model_path, "name:", model_name)
    # Use the directory of the output_audio as the target directory
    output_dir = os.path.dirname(os.path.abspath(output_audio))
    
    # Set up additional parameters
    embedder_model = "contentvec"   # Use contentvec for speaker embedding
    f0_methods = {F0Method.RMVPE}
    
    # Hardcode conversion parameters.
    # Note: In this new workflow the convert() function handles audio file conversion (wavify),
    # metadata, and then calls the internal convert_audio. Resample_sr is not needed here.
    conversion_params = {
        "n_octaves": 0,
        "n_semitones": -20,  # Total pitch shift in semitones.
        "f0_methods": f0_methods,
        "index_rate": 1.0,
        "filter_radius": 5,
        "rms_mix_rate": 1,     # Volume envelope (set to 1 as before)
        "protect_rate": 0.5,
        "hop_length": 1024,
        "split_audio": True,
        "autotune_audio": True,
        "autotune_strength": 1.0,
        "clean_audio": False,
        "clean_strength": 1.0,
        "embedder_model": embedder_model,
        "custom_embedder_model": None,
        "sid": 0,
        "content_type": RVCContentType.SPEECH,
        "progress_bar": None,  # You can pass a gr.Progress instance if using Gradio.
        "percentage": 0.5,
    }
    
    start_time = time.time()
    
    # Call the convert function to perform all necessary pre-processing and conversion.
    converted_audio_path = convert(
        audio_track=input_audio,
        directory=output_dir,
        model_name=model_name,
        **conversion_params
    )
    
    if not converted_audio_path:
        print("Conversion returned None. Check the conversion pipeline for errors.")
    else:
        print("Converted audio saved at:", converted_audio_path)
    
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds.")
    return str(converted_audio_path)


def main():
    parser = argparse.ArgumentParser(
        description="Simplified RVC Inference Script (Using Conversion Function)"
    )
    parser.add_argument("input_audio", type=str, help="Path to the input audio file (e.g., input.wav)")
    parser.add_argument("output_audio", type=str, help="Path to save the converted audio (e.g., output.wav)")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch shift in semitones (default: 0)")
    args = parser.parse_args()
    
    run_inference(
        input_audio=args.input_audio,
        output_audio=args.output_audio,
        pitch=args.pitch,
    )


if __name__ == "__main__":
    main()
