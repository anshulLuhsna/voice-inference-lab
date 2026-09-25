"""AWS EC2 runner for the voice-inference-lab Moshi experiments.

This script provisions nothing. It assumes shell access to a machine you
created, with an NVIDIA driver and a visible GPU, and it only makes the existing
experiment harness runnable there and then runs it. There is no boto3, no
Terraform, and no instance, network, or quota management anywhere in this file.

    python3 aws/run_experiment.py collect-env --out artifacts --label pre-install
    # install the pinned runtime, see aws/README.md
    python3 aws/run_experiment.py collect-env --out artifacts --label post-install
    python3 aws/run_experiment.py suite --root /mnt/<your-nvme>/voice-lab --out artifacts

Every experiment runs in a fresh process, because a managed platform runs every
experiment in a fresh container. Matching that keeps the comparison honest: each
step pays its own model load and its own CUDA context setup, exactly as it did
on Modal.

Provider differences are recorded rather than hidden. Raw nvidia-smi output is
saved verbatim next to the parsed values, and GPU memory is reported in bytes
because GB and GiB renderings invite a false comparison.

The phases are kept separate on purpose. Instance boot is not measured here,
because nothing inside the instance can see it. Download, model load, warm-up,
first frame, and steady state are separate numbers, and none of them is a single
figure called "cold start".
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# The suite, in order, with the overrides each step needs.
#
# The stream steps pin the frame count and the warm-up count so they reproduce
# the three Modal runs exactly: the no-warmup and warm-up runs used 162 frames,
# which is the 8 second clip plus the 5 second tail, and the sustained run used
# 3400 frames to cross the context window. The overrides are module attributes,
# which the experiment reads at call time.
STEPS = {
    "inspect_gpu": {},
    "cache_weights": {"online": True},
    "load_model": {},
    "stream_no_warmup": {"warmup": 0, "frames": 162},
    "stream_warmup": {"warmup": 5, "frames": 162},
    "stream_sustained": {"warmup": 5, "frames": 3400},
}

SUITE_ORDER = [
    "inspect_gpu",
    "cache_weights",
    "load_model",
    "stream_no_warmup",
    "stream_warmup",
    "stream_sustained",
]


def _sh(command, timeout=60):
    """Run a command and return its stdout, or None when the tool is absent."""
    if shutil.which(command[0]) is None:
        return None
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # a missing tool or a timeout is data, not a crash
        return f"<failed: {exc!r}>"
    return (done.stdout or done.stderr or "").strip() or None


def _read(path):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def _meminfo():
    text = _read("/proc/meminfo") or ""
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            out[key] = rest.strip()
    return out


def collect_environment(out_dir: Path, label: str) -> dict:
    """Capture the machine as it is, before and after installing our runtime."""
    out_dir.mkdir(parents=True, exist_ok=True)

    smi_raw = _sh(["nvidia-smi"], timeout=120)
    (out_dir / f"nvidia-smi.{label}.txt").write_text(smi_raw or "nvidia-smi not found\n")

    smi_query = _sh(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap,uuid",
            "--format=csv,noheader",
        ]
    )

    report = {
        "label": label,
        "provider": "aws-ec2",
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "os": {
            "platform": platform.platform(),
            "uname": " ".join(platform.uname()),
            "os_release": _read("/etc/os-release"),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "cpu": {
            "count": os.cpu_count(),
            "model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in (_read("/proc/cpuinfo") or "").splitlines()
                    if line.startswith("model name")
                ),
                None,
            ),
        },
        "system_memory": _meminfo(),
        "gpu": {
            "nvidia_smi_query": smi_query,
            "raw_saved_to": f"nvidia-smi.{label}.txt",
            "cuda_version_reported_by_driver": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in (smi_raw or "").splitlines()
                    if "CUDA Version" in line
                ),
                None,
            ),
        },
        "cuda_toolkit": {
            "nvcc": _sh(["nvcc", "--version"]),
            "cuda_version_json": _read("/usr/local/cuda/version.json"),
        },
        "storage": {
            "lsblk": _sh(["lsblk", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,MODEL"]),
            "df_h": _sh(["df", "-h"]),
            "nvme_devices": sorted(str(p) for p in Path("/dev").glob("nvme*")),
            "root_fstype": _sh(["findmnt", "-no", "FSTYPE", "/"]),
        },
        "pinned_runtime_present": {
            "torch": _sh([sys.executable, "-c", "import torch; print(torch.__version__)"]),
            "torch_cuda": _sh(
                [sys.executable, "-c", "import torch; print(torch.version.cuda)"]
            ),
            "moshi": _sh([sys.executable, "-c", "import moshi; print(moshi.__version__)"]),
            "sphn": _sh([sys.executable, "-c", "import sphn; print('present')"]),
            "hf_offline_set": os.environ.get("HF_HUB_OFFLINE"),
            "voice_lab_root": os.environ.get("VOICE_LAB_ROOT"),
        },
    }

    path = out_dir / f"environment.{label}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path}")
    return report


def run_step(name: str, root: str, out_dir: Path) -> int:
    """Run one experiment. This process is fresh, so the GPU state is fresh."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # The storage root is read when the experiment module is imported, so it has
    # to be in the environment before that import happens.
    os.environ["VOICE_LAB_ROOT"] = root

    spec = STEPS[name]
    if not spec.get("online"):
        # The download step needs the network. Everything else must resolve from
        # the local copy, so a miss becomes a loud error instead of a fetch.
        os.environ["HF_HUB_OFFLINE"] = "1"

    import moshi_experiments as exp

    if "warmup" in spec:
        exp.EXPLICIT_WARMUP_FRAMES = spec["warmup"]
    if "frames" in spec:
        exp.SUSTAINED_FRAMES = spec["frames"]

    started = time.time()
    if name == "inspect_gpu":
        text = exp.inspect_gpu()
    elif name == "cache_weights":
        # No commit callable. On a plain filesystem the default no-op is right,
        # and that difference from the Modal runner is deliberate.
        text = exp.cache_weights()
    elif name == "load_model":
        text = exp.load_model()
    else:
        text = exp.stream_session()

    elapsed_ms = round((time.time() - started) * 1000)
    path = out_dir / f"{name}.json"
    try:
        report = json.loads(text)
    except Exception:
        report = {"raw": text}
    report["aws_step_process_ms"] = elapsed_ms
    report["aws_step_name"] = name
    report["aws_step_overrides"] = spec
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path} in {elapsed_ms} ms")
    return 0


def _step_env(root: str, online: bool) -> dict:
    env = dict(os.environ)
    env["VOICE_LAB_ROOT"] = root
    if online:
        env.pop("HF_HUB_OFFLINE", None)
    else:
        env["HF_HUB_OFFLINE"] = "1"
    return env


def run_suite(root: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "provider": "aws-ec2",
        "root": root,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": (
            "Each step is a separate process, matching one fresh container per "
            "experiment on the managed platform. Instance boot time is not "
            "measurable from inside and is not included."
        ),
        "steps": [],
    }

    for name in SUITE_ORDER:
        online = STEPS[name].get("online", False)
        started = time.time()
        proc = subprocess.run(
            [sys.executable, __file__, "step", name, "--root", root, "--out", str(out_dir)],
            env=_step_env(root, online),
            capture_output=True,
            text=True,
        )
        elapsed_ms = round((time.time() - started) * 1000)
        (out_dir / f"{name}.log").write_text(
            f"$ aws/run_experiment.py step {name}\n"
            f"exit_code={proc.returncode} wall_ms={elapsed_ms}\n\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
        )
        manifest["steps"].append(
            {
                "step": name,
                "exit_code": proc.returncode,
                "wall_ms": elapsed_ms,
                "online": online,
                "overrides": STEPS[name],
            }
        )
        print(f"{name}: exit={proc.returncode} wall={elapsed_ms} ms")
        if proc.returncode != 0:
            # A Spot interruption lands here. Report it, do not hide it, and do
            # not retry silently.
            manifest["stopped_at"] = name
            break

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out_dir / 'manifest.json'}")
    print(
        "\nCopy the artifacts directory off this instance before shutdown. The "
        "local disk is ephemeral and a Spot instance can be reclaimed."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("collect-env", help="capture the machine as it is")
    env.add_argument("--out", default="artifacts")
    env.add_argument("--label", default="unlabelled")

    step = sub.add_parser("step", help="run one experiment in this process")
    step.add_argument("name", choices=sorted(STEPS))
    step.add_argument("--root", required=True)
    step.add_argument("--out", default="artifacts")

    suite = sub.add_parser("suite", help="run every experiment, one process each")
    suite.add_argument("--root", required=True)
    suite.add_argument("--out", default="artifacts")

    args = parser.parse_args()

    if args.command == "collect-env":
        collect_environment(Path(args.out), args.label)
        return 0
    if args.command == "step":
        return run_step(args.name, args.root, Path(args.out))
    return run_suite(args.root, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
