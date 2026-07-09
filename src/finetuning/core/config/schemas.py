"""Typed configuration schemas validated with Pydantic.

Each schema maps one-to-one to a Hydra config group under ``configs/``.
"""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from finetuning.core.enums import (
    AttentionImplementation,
    DatasetFileFormat,
    DatasetSchema,
    DeviceType,
    ExportFormat,
    GGUFQuantization,
    LoraBias,
    OptimizerType,
    PaddingSide,
    Precision,
    QuantizationType,
    SchedulerType,
    TrainingMethod,
)


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(BaseConfig):
    name: str = Field(min_length=1)
    revision: str = "main"
    cache_dir: Path = Path("models")
    trust_remote_code: bool = False
    precision: Precision = Precision.AUTO
    attention: AttentionImplementation = AttentionImplementation.AUTO


class EarlyStoppingConfig(BaseConfig):
    enabled: bool = False
    patience: int = Field(default=3, gt=0)
    threshold: float = Field(default=0.0, ge=0)


class TrainingConfig(BaseConfig):
    method: TrainingMethod = TrainingMethod.QLORA
    seed: int = 42
    num_epochs: float = Field(default=1.0, gt=0)
    max_steps: int = Field(default=-1, ge=-1)
    micro_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    context_length: int = Field(default=1024, gt=0)
    packing: bool = False
    gradient_checkpointing: bool = True
    eval_steps: int = Field(default=50, gt=0)
    save_steps: int = Field(default=50, gt=0)
    logging_steps: int = Field(default=10, gt=0)
    save_total_limit: int = Field(default=3, gt=0)
    resume_from_checkpoint: Path | None = None
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()


class OptimizerConfig(BaseConfig):
    type: OptimizerType = OptimizerType.PAGED_ADAMW_8BIT
    learning_rate: float = Field(default=2e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.999, gt=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0)


class SchedulerConfig(BaseConfig):
    type: SchedulerType = SchedulerType.COSINE
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    warmup_steps: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_warmup(self) -> Self:
        if self.warmup_ratio > 0 and self.warmup_steps > 0:
            raise ValueError("set either warmup_ratio or warmup_steps, not both")
        return self


class CleaningConfig(BaseConfig):
    strip_whitespace: bool = True
    drop_empty: bool = True
    min_chars: int = Field(default=1, ge=0)
    max_chars: int = Field(default=100_000, gt=0)


class SplitConfig(BaseConfig):
    validation: float = Field(default=0.05, ge=0, lt=1)
    test: float = Field(default=0.0, ge=0, lt=1)
    shuffle: bool = True

    @model_validator(mode="after")
    def validate_fractions(self) -> Self:
        if self.validation + self.test >= 1:
            raise ValueError("validation + test fractions must be below 1")
        return self


class DatasetConfig(BaseConfig):
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    format: DatasetFileFormat
    record_schema: DatasetSchema = DatasetSchema.CHAT
    streaming: bool = False
    deduplicate: bool = True
    text_field: str = "text"
    messages_field: str = "messages"
    cleaning: CleaningConfig = CleaningConfig()
    split: SplitConfig = SplitConfig()
    cache_dir: Path = Path("datasets/cache")
    processed_dir: Path = Path("datasets/processed")


class QuantizationConfig(BaseConfig):
    load_in_4bit: bool = False
    quant_type: QuantizationType = QuantizationType.NF4
    use_double_quant: bool = True
    compute_precision: Precision = Precision.BF16


class LoraConfig(BaseConfig):
    r: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] | str = "all-linear"
    bias: LoraBias = LoraBias.NONE
    quantization: QuantizationConfig = QuantizationConfig()


class MLflowConfig(BaseConfig):
    enabled: bool = True
    tracking_uri: str = "sqlite:///artifacts/mlflow.db"
    experiment_name: str = "finetuning"


class TensorBoardConfig(BaseConfig):
    enabled: bool = True
    log_dir: Path = Path("artifacts/tensorboard")


class LoggingConfig(BaseConfig):
    level: str = "INFO"
    log_dir: Path = Path("logs")
    json_metrics: bool = True
    csv_metrics: bool = True
    system_metrics_interval_seconds: float = Field(default=5.0, gt=0)
    mlflow: MLflowConfig = MLflowConfig()
    tensorboard: TensorBoardConfig = TensorBoardConfig()


class BenchmarkConfig(BaseConfig):
    num_prompts: int = Field(default=5, gt=0)
    max_new_tokens: int = Field(default=128, gt=0)


class EvaluationConfig(BaseConfig):
    enabled: bool = True
    perplexity: bool = True
    compare_with_base: bool = True
    save_plots: bool = True
    max_eval_samples: int | None = Field(default=None, gt=0)
    benchmark: BenchmarkConfig = BenchmarkConfig()


class ExperimentConfig(BaseConfig):
    runs_dir: Path = Path("runs")
    name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class HardwareConfig(BaseConfig):
    device: DeviceType = DeviceType.AUTO
    allow_cpu_fallback: bool = True
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    prefetch_factor: int = Field(default=2, gt=0)
    persistent_workers: bool = True
    torch_compile: bool = False
    max_vram_fraction: float | None = Field(default=None, gt=0.0, le=1.0)


class TokenizerConfig(BaseConfig):
    padding_side: PaddingSide = PaddingSide.RIGHT
    add_eos_token: bool = True
    chat_template: str | None = None


class OllamaConfig(BaseConfig):
    model_name: str = Field(default="finetuned-model", min_length=1)
    system_prompt: str | None = None
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    num_ctx: int = Field(default=4096, gt=0)


class ExportConfig(BaseConfig):
    formats: list[ExportFormat] = Field(default_factory=lambda: [ExportFormat.GGUF])
    gguf_quantization: GGUFQuantization = GGUFQuantization.Q4_K_M
    ollama: OllamaConfig = OllamaConfig()


class AppConfig(BaseConfig):
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    dataset: DatasetConfig
    lora: LoraConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    experiment: ExperimentConfig
    hardware: HardwareConfig
    tokenizer: TokenizerConfig
    export: ExportConfig

    @model_validator(mode="after")
    def validate_method_quantization(self) -> Self:
        if self.training.method is TrainingMethod.QLORA and not self.lora.quantization.load_in_4bit:
            raise ValueError("training.method=qlora requires lora.quantization.load_in_4bit=true")
        if self.training.method is TrainingMethod.LORA and self.lora.quantization.load_in_4bit:
            raise ValueError("training.method=lora is incompatible with 4-bit quantization")
        return self
