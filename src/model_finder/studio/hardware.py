"""What is this computer? Used when a pipeline should run locally: the server runs on the user's machine, so
its hardware is the target. Apple Silicon shares memory between CPU and GPU (unified); an NVIDIA card's VRAM is
the limit for GPU models; otherwise system RAM (CPU inference)."""
from __future__ import annotations

import platform
import re
import subprocess
from functools import lru_cache

from .. import query


def _run(cmd: list[str], timeout: float = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _ram_gb() -> float | None:
    if platform.system() == "Darwin":
        v = _run(["sysctl", "-n", "hw.memsize"])
        return round(int(v) / 2**30) if v.isdigit() else None
    try:
        with open("/proc/meminfo") as f:
            kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
        return round(kb / 2**20)
    except (OSError, AttributeError):
        return None


def _nvidia() -> tuple[str, float] | None:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return None
    name, mib = out.splitlines()[0].rsplit(",", 1)
    return name.strip(), round(float(mib) / 1024)


@lru_cache(maxsize=1)
def detect() -> dict:
    system = platform.system()
    ram = _ram_gb()
    chip = gpu = None
    vram = None
    unified = False
    if system == "Darwin":
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Mac"
        unified = _run(["sysctl", "-n", "hw.optional.arm64"]) == "1"
        cores = re.search(r"Total Number of Cores:\s*(\d+)", _run(["system_profiler", "SPDisplaysDataType"], 20))
        gpu = f"{chip} ({cores.group(1)}-core GPU)" if cores else chip
        kind = "mac"
    else:
        chip = platform.processor() or platform.machine()
        nv = _nvidia()
        if nv:
            gpu, vram = nv
        kind = "gpu" if vram and vram >= 4 else "laptop"
    memory = vram if kind == "gpu" else ram
    p = query.apply_device(query.Plan(question="", semantic_query="", device=kind, memory_gb=memory))
    label = f"{chip} · {ram:g} GB" if ram else chip
    if vram:
        label += f" · {gpu} {vram:g} GB"
    return {"label": label, "kind": kind, "chip": chip, "ram_gb": ram, "gpu": gpu, "vram_gb": vram,
            "unified": unified, "memory_gb": memory, "budget_gb": p.ram_cap_gb, "os": system}
