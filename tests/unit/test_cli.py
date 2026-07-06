from typer.testing import CliRunner

from finetuning.cli.app import app

runner = CliRunner()


def test_system_info_runs() -> None:
    result = runner.invoke(app, ["system-info"])

    assert result.exit_code == 0
    assert "CUDA available" in result.output


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "system-info" in result.output
    assert "gpu-info" in result.output
