"""System resource metrics collection (GPU, CPU, RAM)."""

import contextlib
from typing import Any

import psutil
from loguru import logger


class SystemMetricsCollector:
    """Collects instantaneous GPU/CPU/RAM metrics for training telemetry."""

    def __init__(self, gpu_index: int = 0) -> None:
        self._gpu_index = gpu_index
        self._nvml: Any = None
        self._handle: Any = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception as error:
            logger.warning("NVML unavailable, GPU metrics disabled: {}", error)

    def snapshot(self) -> dict[str, float]:
        virtual_memory = psutil.virtual_memory()
        metrics: dict[str, float] = {
            "sys_cpu_percent": psutil.cpu_percent(),
            "sys_ram_used_mb": virtual_memory.used / (1024 * 1024),
            "sys_ram_percent": virtual_memory.percent,
        }
        if self._nvml is not None:
            try:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                temperature = self._nvml.nvmlDeviceGetTemperature(
                    self._handle, self._nvml.NVML_TEMPERATURE_GPU
                )
                metrics.update(
                    {
                        "sys_gpu_utilization_percent": float(utilization.gpu),
                        "sys_vram_used_mb": memory.used / (1024 * 1024),
                        "sys_vram_total_mb": memory.total / (1024 * 1024),
                        "sys_gpu_temperature_c": float(temperature),
                    }
                )
            except Exception as error:
                logger.debug("GPU metrics snapshot failed: {}", error)
        return metrics

    def close(self) -> None:
        if self._nvml is not None:
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()
            self._nvml = None
