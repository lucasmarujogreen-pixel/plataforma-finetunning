"""Shared enumerations used across configuration and runtime."""

from enum import StrEnum


class TrainingMethod(StrEnum):
    LORA = "lora"
    QLORA = "qlora"
    FULL_SFT = "full_sft"


class DatasetFileFormat(StrEnum):
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    ARROW = "arrow"
    HF_HUB = "hf_hub"


class DatasetSchema(StrEnum):
    CHAT = "chat"
    ALPACA = "alpaca"
    TEXT = "text"


class Precision(StrEnum):
    AUTO = "auto"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class AttentionImplementation(StrEnum):
    AUTO = "auto"
    FLASH_ATTENTION_2 = "flash_attention_2"
    SDPA = "sdpa"
    EAGER = "eager"


class OptimizerType(StrEnum):
    ADAMW_TORCH = "adamw_torch"
    ADAMW_8BIT = "adamw_8bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_ADAMW_32BIT = "paged_adamw_32bit"
    ADAFACTOR = "adafactor"


class SchedulerType(StrEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class QuantizationType(StrEnum):
    NF4 = "nf4"
    FP4 = "fp4"


class ExportFormat(StrEnum):
    LORA_ADAPTER = "lora_adapter"
    MERGED_SAFETENSORS = "merged_safetensors"
    GGUF = "gguf"


class GGUFQuantization(StrEnum):
    F16 = "f16"
    Q8_0 = "q8_0"
    Q5_K_M = "q5_k_m"
    Q4_K_M = "q4_k_m"


class DeviceType(StrEnum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class PaddingSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class LoraBias(StrEnum):
    NONE = "none"
    ALL = "all"
    LORA_ONLY = "lora_only"
