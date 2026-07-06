"""Platform exception hierarchy."""


class PlatformError(Exception):
    """Base class for all platform errors."""


class ConfigurationError(PlatformError):
    """Invalid, missing or inconsistent configuration."""


class HardwareError(PlatformError):
    """Hardware detection or capability failure."""


class DatasetError(PlatformError):
    """Dataset loading, validation or processing failure."""


class ModelError(PlatformError):
    """Model download, loading or preparation failure."""


class TrainingError(PlatformError):
    """Training execution failure."""


class EvaluationError(PlatformError):
    """Evaluation or benchmark failure."""


class ExportError(PlatformError):
    """Model export or conversion failure."""


class ExperimentError(PlatformError):
    """Experiment tracking or persistence failure."""
