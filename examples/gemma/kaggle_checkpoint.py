import os
import subprocess
from pathlib import Path

import kagglehub

# Update the following with your Kaggle username and key
os.environ["KAGGLE_USERNAME"] = "klue980"
os.environ["KAGGLE_KEY"] = "9643a971d4fd332b3072343e4c5b37b5"

torch_weights_dir = kagglehub.model_download(f"google/gemma/pyTorch/2b-it")
print(torch_weights_dir)

# Create the directory where we will save the TRT engine. This is the file we will be deploying.
model_dir = Path(os.getcwd() + "/trt-engine").expanduser()
model_dir.mkdir(parents=True, exist_ok=True)
print(model_dir)

# Convert the checkpoint to TRT engine
part1 = f"""
python3 ./convert_checkpoint.py \
    --ckpt-type torch \
    --model-dir {str(torch_weights_dir)} \
    --dtype float16 \
    --world-size 1 \
    --output-model-dir {os.getcwd()}
"""

print("Building checkpoint...")
subprocess.run(part1, shell=True, check=True)
print("\nDownload success PATH : ", torch_weights_dir)
print("\nConvert Checkpoint success PATH : ", os.getcwd())