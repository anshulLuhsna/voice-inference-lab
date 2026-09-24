"""Smallest Modal GPU environment check for voice-inference-lab.

Plumbing certification only: proves CUDA is reachable and that a real tensor
operation runs from Python on the remote GPU. It does not certify that this
GPU class can serve any particular voice model.
"""

import modal

app = modal.App("voice-inference-lab-gpu-check")

image = modal.Image.debian_slim().uv_pip_install("torch", "numpy")


@app.function(gpu="A10G", image=image)
def inspect_gpu() -> str:
    import json
    import platform
    import shutil
    import subprocess
    import time

    import torch

    report = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }

    if shutil.which("nvidia-smi"):
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        report["nvidia_smi"] = smi.stdout.strip()

    if report["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        start = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        report.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "vram_total_gb": round(props.total_memory / 1e9, 1),
                "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                "matmul_ok": tuple(c.shape) == (512, 512),
                "matmul_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        )

    return json.dumps(report, indent=2)


@app.local_entrypoint()
def main() -> None:
    print(inspect_gpu.remote())
