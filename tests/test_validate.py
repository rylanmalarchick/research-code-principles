"""Tests for the unified `bible validate` command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from agentbible.cli.main import cli
from agentbible.cli.validate import get_available_checks, run_validate
from agentbible.provenance import build_provenance_record


class TestGetAvailableChecks:
    """Tests for get_available_checks()."""

    def test_returns_spec_checks(self) -> None:
        checks = get_available_checks()
        assert "unitary" in checks
        assert "density_matrix" in checks
        assert "normalized_l1" in checks


class TestRunValidatePython:
    """Python-file validation tests."""

    def test_validates_unitary_matrix(self, tmp_path: Path) -> None:
        filepath = tmp_path / "hadamard.npy"
        matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        np.save(filepath, matrix)

        exit_code = run_validate(str(filepath), ["unitary"], lang="python")
        assert exit_code == 0

    def test_fails_invalid_density_matrix(self, tmp_path: Path) -> None:
        filepath = tmp_path / "bad.npy"
        np.save(filepath, np.eye(2, dtype=complex))

        exit_code = run_validate(str(filepath), ["density_matrix"], lang="python")
        assert exit_code == 1


class TestRunValidateProvenance:
    """Cross-language provenance viewing tests."""

    def test_reads_valid_rust_record(self, tmp_path: Path) -> None:
        filepath = tmp_path / "rust.json"
        record = build_provenance_record(
            language="rust",
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
        )
        filepath.write_text(record.to_json(), encoding="utf-8")

        exit_code = run_validate(str(filepath), ["unitary"], lang="rust")
        assert exit_code == 0

    def test_fails_invalid_schema(self, tmp_path: Path) -> None:
        filepath = tmp_path / "bad.json"
        filepath.write_text(json.dumps({"language": "rust"}), encoding="utf-8")

        exit_code = run_validate(str(filepath), lang="rust")
        assert exit_code == 1


class TestValidateCLI:
    """CLI integration tests."""

    def test_validate_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])

        assert result.exit_code == 0
        assert "Validate Python data or inspect a provenance JSON record" in result.output
        assert "--lang" in result.output

    def test_validate_python_cli(self, tmp_path: Path) -> None:
        runner = CliRunner()
        filepath = tmp_path / "matrix.npy"
        np.save(filepath, np.eye(2, dtype=complex))

        result = runner.invoke(cli, ["validate", str(filepath), "--check", "hermitian"])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_validate_provenance_cli(self, tmp_path: Path) -> None:
        runner = CliRunner()
        filepath = tmp_path / "cpp.json"
        record = build_provenance_record(
            language="cpp",
            checks_passed=[
                {
                    "check_name": "positive_definite",
                    "passed": False,
                    "rtol": 0.0,
                    "atol": 0.0,
                    "norm_used": "n/a",
                    "error_message": "Cholesky factorization failed",
                }
            ],
        )
        filepath.write_text(record.to_json(), encoding="utf-8")

        result = runner.invoke(cli, ["validate", str(filepath), "--lang", "cpp"])
        assert result.exit_code == 1
        assert "positive_definite" in result.output
