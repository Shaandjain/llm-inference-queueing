"""vLLM OpenAI-compatible server on a Modal L4.

Deploy:  modal deploy serving/vllm_modal.py
Stop:    modal app stop vllm-l4-qwen7b        (don't leave it idling)

The endpoint requires `Authorization: Bearer $VLLM_API_KEY` (set below) since
Modal web endpoints are public URLs. Container scales to zero after 5 idle
minutes; weights persist in the `hf-cache` volume so reboots are fast.
"""

import subprocess

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
API_KEY = "inference-summer-validation"  # not a secret that matters; gates a transient bench endpoint
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

app = modal.App("vllm-l4-qwen7b")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=3600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=256)
@modal.web_server(port=PORT, startup_timeout=1800)
def serve():
    subprocess.Popen(
        f"vllm serve {MODEL} --host 0.0.0.0 --port {PORT} "
        f"--api-key {API_KEY} "
        "--max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.92",
        shell=True,
    )
