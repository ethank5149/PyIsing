"""
Hardware capability detection for PyIsing.

Detects available CPU and GPU resources and recommends the best backend.

Example::

    from pyising.hardware import detect, print_summary, recommended_backend
    info = detect()
    print_summary(info)
    backend = recommended_backend(info)   # 'gpu' or 'cpu'
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CUDADeviceInfo:
    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    compute_capability: str
    cuda_version: str


@dataclass
class HardwareInfo:
    platform: str
    python_version: str
    cpu_count: int
    cpu_model: str

    # GPU
    cuda_available: bool
    cuda_version: Optional[str]
    cupy_available: bool
    cupy_version: Optional[str]
    devices: List[CUDADeviceInfo] = field(default_factory=list)

    # Recommendations
    recommended_backend: str = 'cpu'
    recommended_batch_size: int = 1


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _cpu_model() -> str:
    """Best-effort CPU model string."""
    try:
        if platform.system() == 'Linux':
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
        elif platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=2)
            return result.stdout.strip()
        elif platform.system() == 'Windows':
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True, text=True, timeout=2)
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip() and l.strip() != 'Name']
            return lines[0] if lines else 'Unknown'
    except Exception:
        pass
    return platform.processor() or 'Unknown'


def _cuda_version_from_nvcc() -> Optional[str]:
    """Try to get CUDA version from nvcc."""
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if 'release' in line.lower():
                # e.g. "Cuda compilation tools, release 12.1, V12.1.105"
                parts = line.split('release')
                if len(parts) > 1:
                    return parts[1].split(',')[0].strip()
    except Exception:
        pass
    return None


def _cuda_version_from_nvidia_smi() -> Optional[str]:
    """Try to get CUDA version from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # nvidia-smi header line often contains CUDA version
            pass
        # Try the full output for CUDA version
        result2 = subprocess.run(
            ['nvidia-smi'],
            capture_output=True, text=True, timeout=5)
        for line in result2.stdout.splitlines():
            if 'CUDA Version' in line:
                parts = line.split('CUDA Version:')
                if len(parts) > 1:
                    return parts[1].strip().split()[0]
    except Exception:
        pass
    return None


def _probe_cupy_devices() -> tuple[bool, Optional[str], list[CUDADeviceInfo]]:
    """Return (cupy_available, cupy_version, device_list)."""
    try:
        import cupy as cp  # type: ignore
        version = cp.__version__
        devices = []
        n = cp.cuda.runtime.getDeviceCount()
        for i in range(n):
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                mem = cp.cuda.Device(i).mem_info  # (free, total) in bytes
                free_gb = mem[0] / 1024 ** 3
                total_gb = mem[1] / 1024 ** 3
                cc = f"{props['major']}.{props['minor']}"
                cuda_ver = _cuda_version_from_nvidia_smi() or _cuda_version_from_nvcc() or 'unknown'
                devices.append(CUDADeviceInfo(
                    index=i,
                    name=props['name'].decode() if isinstance(props['name'], bytes) else props['name'],
                    total_memory_gb=round(total_gb, 2),
                    free_memory_gb=round(free_gb, 2),
                    compute_capability=cc,
                    cuda_version=cuda_ver,
                ))
        return True, version, devices
    except Exception:
        return False, None, []


def _recommend(info: HardwareInfo) -> tuple[str, int]:
    """Return (backend, batch_size) recommendation."""
    if not info.cupy_available or not info.devices:
        return 'cpu', 1

    # Use the device with the most free VRAM
    best = max(info.devices, key=lambda d: d.free_memory_gb)

    # Rough heuristic: each 10×10 int8 lattice frame costs ~100 bytes;
    # for a batch of sample_size=30 with frames=150: 30*150*100 = ~450 KB
    # We can fit thousands of batches in 24 GB — cap at a sensible default
    vram_gb = best.free_memory_gb
    if vram_gb >= 20:
        batch = 256
    elif vram_gb >= 8:
        batch = 128
    elif vram_gb >= 4:
        batch = 64
    else:
        batch = 32

    return 'gpu', batch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect() -> HardwareInfo:
    """
    Probe the current system and return a :class:`HardwareInfo` snapshot.

    This function never raises — all errors are caught and reflected as
    ``cuda_available=False`` / ``cupy_available=False``.
    """
    import sys

    cupy_ok, cupy_ver, devices = _probe_cupy_devices()
    cuda_ver = None
    if devices:
        cuda_ver = devices[0].cuda_version
    else:
        cuda_ver = _cuda_version_from_nvidia_smi() or _cuda_version_from_nvcc()

    info = HardwareInfo(
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        cpu_count=os.cpu_count() or 1,
        cpu_model=_cpu_model(),
        cuda_available=bool(devices),
        cuda_version=cuda_ver,
        cupy_available=cupy_ok,
        cupy_version=cupy_ver,
        devices=devices,
    )
    info.recommended_backend, info.recommended_batch_size = _recommend(info)
    return info


def print_summary(info: Optional[HardwareInfo] = None) -> None:
    """
    Pretty-print a hardware summary table.

    If *info* is ``None``, :func:`detect` is called automatically.
    """
    if info is None:
        info = detect()

    sep = '─' * 52
    print(sep)
    print('  PyIsing Hardware Summary')
    print(sep)
    print(f'  Platform   : {info.platform}')
    print(f'  Python     : {info.python_version}')
    print(f'  CPU cores  : {info.cpu_count}')
    print(f'  CPU model  : {info.cpu_model}')
    print(sep)
    if info.cupy_available and info.devices:
        print(f'  CuPy       : {info.cupy_version}')
        print(f'  CUDA       : {info.cuda_version}')
        for d in info.devices:
            print(f'  GPU [{d.index}]   : {d.name}')
            print(f'             VRAM  {d.free_memory_gb:.1f} GB free / {d.total_memory_gb:.1f} GB total')
            print(f'             Compute capability {d.compute_capability}')
    else:
        print('  GPU        : not available (CuPy not installed or no CUDA device)')
        if not info.cupy_available:
            print('  Hint       : pip install cupy-cuda12x')
    print(sep)
    print(f'  Recommended backend    : {info.recommended_backend.upper()}')
    print(f'  Recommended batch size : {info.recommended_batch_size}')
    print(sep)


def recommended_backend(info: Optional[HardwareInfo] = None) -> str:
    """Return ``'gpu'`` or ``'cpu'`` based on detected hardware."""
    if info is None:
        info = detect()
    return info.recommended_backend
