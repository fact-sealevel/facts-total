from click.testing import CliRunner
from unittest.mock import patch
from facts_total.cli import main

runner = CliRunner()


def test_cli_help_exits_zero():
    """--help runs and exits with 0"""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0


@patch("facts_total.cli.run_facts_total")
def test_cli_calls_run_facts_total(mock_run_facts_total):
    result = runner.invoke(
        main,
        [
            "--name",
            "myname",
            "--item",
            "file1.nc",
            "--item",
            "file2.nc",
            "--item",
            "file3.nc",
            "--pyear-start",
            "2020",
            "--pyear-step",
            "10",
            "--pyear-end",
            "2150",
            "--output-path",
            "test/out/path.nc",
        ],
    )
    assert result.exit_code == 0, (
        f"test failed, Exception: \n{result.exception}"
        f"test failed, Exc info: \n{result.exc_info}"
    )
    mock_run_facts_total.assert_called_once_with(
        name="myname",
        item=("file1.nc", "file2.nc", "file3.nc"),
        output_path="test/out/path.nc",
        pyear_start=2020,
        pyear_end=2150,
        pyear_step=10,
    )
