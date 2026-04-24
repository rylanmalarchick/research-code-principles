"""Tests for `bible report`."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from agentbible.cli.main import cli
from agentbible.cli.report import generate_markdown_report, run_report
from agentbible.provenance import build_provenance_record


def _write_record(path: Path) -> Path:
    record = build_provenance_record(
        language="julia",
        checks_passed=[
            {
                "check_name": "unitary",
                "passed": True,
                "rtol": 1e-10,
                "atol": 1e-12,
                "norm_used": "frobenius",
                "error_message": None,
            },
            {
                "check_name": "positive_definite",
                "passed": False,
                "rtol": 0.0,
                "atol": 0.0,
                "norm_used": "n/a",
                "error_message": "Cholesky factorization failed",
            },
        ],
    )
    path.write_text(record.to_json(), encoding="utf-8")
    return path


class TestReportHelpers:
    """Helper-level report tests."""

    def test_generate_markdown_report(self, tmp_path: Path) -> None:
        path = _write_record(tmp_path / "record.json")
        record = build_provenance_record(
            language="julia",
            checks_passed=[
                {
                    "check_name": "unitary",
                    "passed": True,
                    "rtol": 1e-10,
                    "atol": 1e-12,
                    "norm_used": "frobenius",
                    "error_message": None,
                }
            ],
        ).to_dict()
        output = generate_markdown_report(record)
        assert "AgentBible Provenance Report" in output
        assert "Language:" in output
        assert "Checks:" in output
        assert path.exists()


class TestRunReport:
    """Command execution tests."""

    def test_run_report(self, tmp_path: Path, capsys: object) -> None:
        path = _write_record(tmp_path / "record.json")
        exit_code = run_report(str(path), "markdown")
        captured = capsys.readouterr()  # type: ignore[attr-defined]
        assert exit_code == 0
        assert "AgentBible Provenance Report" in captured.out

    def test_report_cli(self, tmp_path: Path) -> None:
        runner = CliRunner()
        path = _write_record(tmp_path / "record.json")
        result = runner.invoke(cli, ["report", str(path)])
        assert result.exit_code == 0
        assert "AgentBible Provenance Report" in result.output
