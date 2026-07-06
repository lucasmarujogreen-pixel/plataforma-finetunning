import pytest

from finetuning.core.config.schemas import HardwareConfig
from finetuning.core.enums import AttentionImplementation, DeviceType, Precision
from finetuning.core.exceptions import HardwareError
from finetuning.monitoring.hardware import (
    HardwareProfile,
    detect_hardware,
    resolve_attention,
    resolve_device,
    resolve_precision,
)


def make_profile(**kwargs: object) -> HardwareProfile:
    defaults: dict[str, object] = {
        "cuda_available": False,
        "cuda_version": None,
        "driver_version": None,
    }
    defaults.update(kwargs)
    return HardwareProfile(**defaults)  # type: ignore[arg-type]


def test_detect_hardware_returns_coherent_profile() -> None:
    profile = detect_hardware()

    assert profile.cpu_count > 0
    assert profile.total_ram_mb > 0
    if profile.cuda_available:
        assert profile.gpus
        assert profile.cuda_version


def test_resolve_device_prefers_cuda() -> None:
    profile = make_profile(cuda_available=True)

    assert resolve_device(profile, HardwareConfig()) is DeviceType.CUDA


def test_resolve_device_falls_back_to_cpu_when_allowed() -> None:
    profile = make_profile(cuda_available=False)

    assert resolve_device(profile, HardwareConfig(allow_cpu_fallback=True)) is DeviceType.CPU


def test_resolve_device_rejects_missing_cuda_without_fallback() -> None:
    profile = make_profile(cuda_available=False)

    with pytest.raises(HardwareError):
        resolve_device(profile, HardwareConfig(allow_cpu_fallback=False))


def test_resolve_device_rejects_explicit_cuda_request_without_gpu() -> None:
    profile = make_profile(cuda_available=False)

    with pytest.raises(HardwareError):
        resolve_device(profile, HardwareConfig(device=DeviceType.CUDA))


def test_resolve_precision_auto_prefers_bf16() -> None:
    profile = make_profile(cuda_available=True, supports_bf16=True, supports_fp16=True)

    assert resolve_precision(profile, Precision.AUTO) is Precision.BF16


def test_resolve_precision_auto_without_bf16_uses_fp16() -> None:
    profile = make_profile(cuda_available=True, supports_bf16=False, supports_fp16=True)

    assert resolve_precision(profile, Precision.AUTO) is Precision.FP16


def test_resolve_precision_auto_on_cpu_uses_fp32() -> None:
    profile = make_profile(cuda_available=False)

    assert resolve_precision(profile, Precision.AUTO) is Precision.FP32


def test_resolve_precision_respects_explicit_request() -> None:
    profile = make_profile(cuda_available=True, supports_bf16=True)

    assert resolve_precision(profile, Precision.FP16) is Precision.FP16


def test_resolve_attention_auto_prefers_flash() -> None:
    profile = make_profile(cuda_available=True, flash_attention_available=True)

    assert (
        resolve_attention(profile, AttentionImplementation.AUTO)
        is AttentionImplementation.FLASH_ATTENTION_2
    )


def test_resolve_attention_auto_falls_back_to_sdpa() -> None:
    profile = make_profile(cuda_available=True, flash_attention_available=False)

    assert resolve_attention(profile, AttentionImplementation.AUTO) is AttentionImplementation.SDPA
