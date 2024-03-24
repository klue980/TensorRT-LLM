# original requirements.txt
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
--extra-index-url https://pypi.nvidia.com
tensorrt_llm==0.9.0.dev2024031900
flax~=0.8.0
jax[cuda12_pip]~=0.4.19; platform_system != "Windows"
jax~=0.4.19; platform_system == "Windows"
safetensors~=0.4.1
sentencepiece~=0.1.99
h5py~=3.10.0
easydict~=1.11
rouge_score
nltk

# modified requirements.txt
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
--extra-index-url https://pypi.nvidia.com
tensorrt_llm==0.9.0.dev2024031900
flax~=0.8.0
jax
safetensors~=0.4.1
sentencepiece~=0.1.99
h5py~=3.10.0
easydict~=1.11
rouge_score
nltk
mpmath==1.3.0
kagglehub

mpmath 1.4.0a does not work with the current version of tensorrt_llm

$ python3 -c "import torch; print(torch.__version__)"
$ python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
$ python3 -c "import mpmath; print(mpmath.__version__)"
$ python3 -c "import jax; print(jax.__version__)"

# setting
part_hf_convert = f"""
mkdir -p ./check/hf/2b-it/fp16 && \
    mkdir -p ./trt-engine/hf/2b-it/fp16 && \
    mkdir -p ./trt-engine/hf/2b-it-context-disable/fp16 && \
    mkdir -p ./NSYS && \
    mkdir -p ./NCU
"""


# under SM80, bf is not working
part_hf_convert = f"""
python3 ./convert_checkpoint.py \
    --ckpt-type hf \
    --model-dir ./gemma-2b-it \
    --dtype float16 \
    --world-size 1 \
    --output-model-dir ./check/hf/2b-it/fp16
"""

part_build = f"""
trtllm-build --checkpoint_dir ./check/hf/2b-it/fp16 \
             --gemm_plugin float16 \
             --gpt_attention_plugin float16 \
             --max_batch_size 1 \
             --max_input_len 384 \
             --max_output_len 2 \
             --context_fmha enable \
             --output_dir ./trt-engine/hf/2b-it/fp16
"""

part_unbuild = f"""
trtllm-build --checkpoint_dir ./check/hf/2b-it/fp16 \
             --gemm_plugin float16 \
             --gpt_attention_plugin float16 \
             --max_batch_size 1 \
             --max_input_len 384 \
             --max_output_len 2 \
             --context_fmha disable \
             --output_dir ./trt-engine/hf/2b-it-context-disable/fp16
"""

part_summarize = f"""
nsys profile --wait all -t cuda,nvtx,cudnn,cublas -f true --stats true -w true -o ./NSYS/gemmaite1ba1in384o2.nsys-rep \
                        python3 ../summarize.py --test_trt_llm \
                        --hf_model_dir ./gemma-2b-it \
                        --data_type fp16 \
                        --engine_dir ./trt-engine/hf/2b-it/fp16 \
                        --batch_size 1 \
                        --max_input_length 384 \
                        --output_len 2 \
                        --max_ite 1
"""