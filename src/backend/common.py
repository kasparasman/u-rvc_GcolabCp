import os
import shutil
import json
import hashlib
from common import BASE_DIR, RVC_MODELS_DIR
from backend.exceptions import PathNotFoundError

SONGS_DIR = os.path.join(BASE_DIR, "songs")
TEMP_AUDIO_DIR = os.path.join(SONGS_DIR, "temp")


def display_progress(message, percentage=None, progress_bar=None):
    if progress_bar is None:
        print(message)
    else:
        progress_bar(percentage, desc=message)


def remove_suffix_after(text: str, occurrence: str):
    location = text.rfind(occurrence)
    if location == -1:
        return text
    else:
        return text[: location + len(occurrence)]


def copy_files_to_new_folder(file_paths, folder_path):
    os.makedirs(folder_path)
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise PathNotFoundError(f"File not found: {file_path}")
        shutil.copyfile(
            file_path, os.path.join(folder_path, os.path.basename(file_path))
        )


def get_path_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def json_dumps(thing):
    return json.dumps(
        thing,
        ensure_ascii=False,
        sort_keys=True,
        indent=4,
        separators=(",", ": "),
    )


def json_dump(thing, path):
    with open(path, "w", encoding="utf-8") as file:
        return json.dump(
            thing,
            file,
            ensure_ascii=False,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
        )


def json_load(path, encoding="utf-8"):
    with open(path, encoding=encoding) as file:
        return json.load(file)


def get_hash(thing, size=5):
    return hashlib.blake2b(
        json_dumps(thing).encode("utf-8"), digest_size=size
    ).hexdigest()


# TODO consider increasing size to 16
# otherwise we might have problems with hash collisions
# when using app as CLI
# TODO use dedicated file_digest function once we upgradeto python 3.11
# for better speedups
def get_file_hash(filepath, digest_size=5, chunk_size=655360):
    with open(filepath, "rb") as f:
        file_hash = hashlib.blake2b(digest_size=digest_size)
        while chunk := f.read(chunk_size):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def get_rvc_model(voice_model):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(RVC_MODELS_DIR, voice_model)
    if not os.path.exists(model_dir):
        raise PathNotFoundError(
            f"Voice model directory '{voice_model}' does not exist."
        )
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == ".pth":
            rvc_model_filename = file
        if ext == ".index":
            rvc_index_filename = file

    if rvc_model_filename is None:
        raise PathNotFoundError(f"No model file exists in {model_dir}.")

    return os.path.join(model_dir, rvc_model_filename), (
        os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ""
    )
