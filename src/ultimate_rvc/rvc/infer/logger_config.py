import logging
import os

# Determine the directory where the main script is run.
# For example, if the main script is in the "infer" directory:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Assumes this file is in the infer dir
LOG_FILE = os.path.join(BASE_DIR, "conversion.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILE,
    filemode='w'  # 'w' to overwrite each run, or 'a' to append
)

# Optionally, also configure a console handler if you want output on the screen.
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
