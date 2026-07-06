"""CLI commands for dataset preparation and inspection."""

from pathlib import Path
from typing import cast

import typer
from rich.table import Table

from finetuning.application.prepare_dataset import PrepareDataset
from finetuning.cli.common import CONFIG_DIR_OPTION, OVERRIDES_ARGUMENT, console, load_cli_config
from finetuning.core.exceptions import DatasetError
from finetuning.domain.entities import DatasetMetadata
from finetuning.preprocessing.loaders import load_raw_dataset
from finetuning.preprocessing.schema_adapters import get_schema_adapter
from finetuning.preprocessing.stages import TokenizerLike
from finetuning.tokenization.loader import load_tokenizer


def _metadata_table(metadata: DatasetMetadata, output_dir: Path | None) -> Table:
    table = Table(title=f"Dataset '{metadata.name}' (version {metadata.version})")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("Source", metadata.source_path)
    table.add_row("Source SHA256", metadata.source_sha256[:16] + "…")
    table.add_row("Schema", metadata.record_schema.value)
    table.add_row("Records", str(metadata.num_records))
    table.add_row("Tokens", str(metadata.num_tokens))
    table.add_row("Mean tokens/record", str(metadata.mean_tokens))
    table.add_row("Max tokens/record", str(metadata.max_tokens))
    table.add_row("Language", metadata.language)
    table.add_row("Splits", ", ".join(f"{k}={v}" for k, v in metadata.splits.items()))
    table.add_row("Dropped by cleaning", str(metadata.records_dropped_by_cleaning))
    table.add_row("Duplicates removed", str(metadata.duplicates_removed))
    table.add_row("Created at", metadata.created_at)
    table.add_row("Output", str(output_dir) if output_dir else "-")
    return table


def prepare_dataset(
    overrides: list[str] = OVERRIDES_ARGUMENT,
    config_dir: Path = CONFIG_DIR_OPTION,
    force: bool = typer.Option(False, "--force", help="Rebuild even when a cache exists."),
) -> None:
    """Run the full preparation pipeline and cache the processed dataset."""
    config = load_cli_config(config_dir, overrides)
    tokenizer = cast(TokenizerLike, load_tokenizer(config.model, config.tokenizer))
    try:
        prepared = PrepareDataset(tokenizer).execute(config, force=force)
    except DatasetError as error:
        console.print(f"[red]Dataset error:[/red] {error}")
        raise typer.Exit(code=1) from error
    console.print(_metadata_table(prepared.metadata, prepared.output_dir))


def tokenize(
    overrides: list[str] = OVERRIDES_ARGUMENT,
    config_dir: Path = CONFIG_DIR_OPTION,
) -> None:
    """Run the pipeline without caching and print token statistics."""
    config = load_cli_config(config_dir, overrides)
    tokenizer = cast(TokenizerLike, load_tokenizer(config.model, config.tokenizer))
    try:
        prepared = PrepareDataset(tokenizer).execute(config, save=False)
    except DatasetError as error:
        console.print(f"[red]Dataset error:[/red] {error}")
        raise typer.Exit(code=1) from error
    console.print(_metadata_table(prepared.metadata, None))


def validate(
    overrides: list[str] = OVERRIDES_ARGUMENT,
    config_dir: Path = CONFIG_DIR_OPTION,
) -> None:
    """Validate raw dataset records against the configured schema."""
    config = load_cli_config(config_dir, overrides)
    adapter = get_schema_adapter(config.dataset)
    try:
        dataset = load_raw_dataset(config.dataset)
    except DatasetError as error:
        console.print(f"[red]Dataset error:[/red] {error}")
        raise typer.Exit(code=1) from error

    problems: list[str] = []
    total = 0
    for index, record in enumerate(dataset):
        total += 1
        problem = adapter.validate(dict(record))
        if problem is not None:
            problems.append(f"record {index}: {problem}")
    if problems:
        console.print(f"[red]{len(problems)} invalid records out of {total}:[/red]")
        for problem in problems[:20]:
            console.print(f"  - {problem}")
        raise typer.Exit(code=1)
    console.print(f"[green]All {total} records are valid.[/green]")
