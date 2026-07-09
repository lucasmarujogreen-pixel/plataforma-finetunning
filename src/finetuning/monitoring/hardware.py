"""Hardware detection and capability resolution."""

import importlib.util
import platform
from dataclasses import dataclass, field

import psutil
import torch
from loguru import logger

from finetuning.core.config.schemas import HardwareConfig
from finetuning.core.enums import AttentionImplementation, DeviceType, Precision
from finetuning.core.exceptions import HardwareError


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_vram_mb: int
    free_vram_mb: int
    compute_capability: tuple[int, int]


@dataclass(frozen=True)
class HardwareProfile:
    cuda_available: bool
    cuda_version: str | None
    driver_version: str | None
    gpus: list[GPUInfo] = field(default_factory=list)
    supports_bf16: bool = False
    supports_fp16: bool = False
    flash_attention_available: bool = False
    xformers_available: bool = False
    torch_compile_supported: bool = False
    cpu_count: int = 0
    total_ram_mb: int = 0
    os_description: str = ""


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _detect_driver_version() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            version = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
        return version.decode() if isinstance(version, bytes) else str(version)
    except Exception:
        return None


def _detect_gpus() -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        gpus.append(
            GPUInfo(
                index=index,
                name=properties.name,
                total_vram_mb=total_bytes // (1024 * 1024),
                free_vram_mb=free_bytes // (1024 * 1024),
                compute_capability=(properties.major, properties.minor),
            )
        )
    return gpus


def detect_hardware() -> HardwareProfile:
    """Probe CUDA, GPUs, precision support and optional acceleration libraries."""
    cuda_available = torch.cuda.is_available()
    return HardwareProfile(
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda if cuda_available else None,
        driver_version=_detect_driver_version() if cuda_available else None,
        gpus=_detect_gpus() if cuda_available else [],
        supports_bf16=cuda_available and torch.cuda.is_bf16_supported(),
        supports_fp16=cuda_available,
        flash_attention_available=cuda_available and _module_available("flash_attn"),
        xformers_available=cuda_available and _module_available("xformers"),
        torch_compile_supported=cuda_available and _module_available("triton"),
        cpu_count=psutil.cpu_count(logical=True) or 0,
        total_ram_mb=psutil.virtual_memory().total // (1024 * 1024),
        os_description=f"{platform.system()} {platform.release()}",
    )


def resolve_device(profile: HardwareProfile, config: HardwareConfig) -> DeviceType:
    """Pick the training device, allowing CPU only as an explicit debug fallback."""
    if config.device is DeviceType.CPU:
        return DeviceType.CPU
    if profile.cuda_available:
        return DeviceType.CUDA
    if config.device is DeviceType.CUDA:
        raise HardwareError("CUDA was requested but is not available")
    if config.allow_cpu_fallback:
        return DeviceType.CPU
    raise HardwareError("CUDA is not available and CPU fallback is disabled")


def resolve_precision(profile: HardwareProfile, requested: Precision) -> Precision:
    """Resolve ``auto`` precision to the best precision the hardware supports."""
    if requested is not Precision.AUTO:
        return requested
    if profile.supports_bf16:
        return Precision.BF16
    if profile.supports_fp16:
        return Precision.FP16
    return Precision.FP32


def limit_vram_usage(device: DeviceType, config: HardwareConfig) -> None:
    """Cap the CUDA allocator so this process leaves VRAM headroom for other GPU work.

    ``torch.cuda.set_per_process_memory_fraction`` bounds how much the caching
    allocator can ever reserve from the driver — training raises OOM instead
    of creeping past the cap, rather than silently squeezing out whatever else
    needs the GPU (the desktop compositor, a browser, etc. on a shared laptop
    GPU under WSL2). No-op when ``max_vram_fraction`` is unset or the device
    isn't CUDA, so existing configs/behavior are unaffected by default.
    """
    if device is not DeviceType.CUDA or config.max_vram_fraction is None:
        return
    torch.cuda.set_per_process_memory_fraction(config.max_vram_fraction)
    logger.info(
        "Capped CUDA memory fraction at {:.0%} — leaving >= {:.0%} of VRAM free for other tasks",
        config.max_vram_fraction,
        1 - config.max_vram_fraction,
    )


def resolve_attention(
    profile: HardwareProfile, requested: AttentionImplementation
) -> AttentionImplementation:
    """Resolve ``auto`` attention to flash-attention when installed, else SDPA."""
    if requested is not AttentionImplementation.AUTO:
        return requested
    if profile.flash_attention_available:
        return AttentionImplementation.FLASH_ATTENTION_2
    return AttentionImplementation.SDPA
