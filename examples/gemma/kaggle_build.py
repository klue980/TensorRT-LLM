import os
import subprocess
from pathlib import Path

import kagglehub
import subprocess

# Update the following with your Kaggle username and key
os.environ["KAGGLE_USERNAME"] = "klue980"
os.environ["KAGGLE_KEY"] = "9643a971d4fd332b3072343e4c5b37b5"

torch_weights_dir = kagglehub.model_download(f"google/gemma/pyTorch/2b-it")
print(torch_weights_dir)

# Create the directory where we will save the TRT engine. This is the file we will be deploying.
model_dir = Path(os.getcwd() + "/trt-engine").expanduser()
model_dir.mkdir(parents=True, exist_ok=True)
print(model_dir)

# Build the TRT engine
# part1 = f"""
# python3 ./convert_checkpoint.py \
#     --ckpt-type torch \
#     --model-dir {str(torch_weights_dir)} \
#     --dtype bfloat16 \
#     --world-size 1 \
#     --output-model-dir {os.getcwd()}
# """

# part2 = f"""
# trtllm-build --checkpoint_dir {os.getcwd()} \
#              --gemm_plugin float16 \
#              --gpt_attention_plugin float16 \
#              --max_batch_size 1 \
#              --max_input_len 384 \
#              --max_output_len 2 \
#              --context_fmha enable \
#              --output_dir {str(model_dir)}
# """

part_sum = f"""
python3 ../summarize.py --test_trt_llm \
                        --tokenizer_dir {str(f"{torch_weights_dir}/")} \
                        --engine_dir {str(f"{model_dir}/")} \
                        --batch_size 1 \
                        --max_ite 1
"""

# print("Building checkpoint...")
# subprocess.run(part1, shell=True, check=True)
# print("\nDownload success PATH : ", torch_weights_dir)
# print("\nConvert Checkpoint success PATH : ", os.getcwd())

# print("Building engine...")
# subprocess.run(part2, shell=True, check=True)
# print("\nEngine built success PATH : ", model_dir)

subprocess.run(part_sum, shell=True, check=True)