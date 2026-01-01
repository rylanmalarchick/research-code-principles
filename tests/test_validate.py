"""Tests for bible validate command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner

from agentbible.cli.main import cli
from agentbible.cli.validate import (
    _check_hermiticity,
    _check_normalization,
    _check_positivity,
    _check_trace,
    _check_unitarity,
    _determine_applicable_checks,
    get_available_checks,
    run_validate,
)


class TestCheckFunctions:
    """Tests for individual check functions."""

    def test_unitarity_pass(self) -> None:
        """Unitary matrix passes unitarity check."""
        hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        passed, msg = _check_unitarity(hadamard, rtol=1e-5, atol=1e-8)
        assert passed
        assert "unitary" in msg.lower()

    def test_unitarity_fail(self) -> None:
        """Non-unitary matrix fails unitarity check."""
        matrix = np.array([[1, 1], [0, 1]], dtype=complex)
        passed, msg = _check_unitarity(matrix, rtol=1e-5, atol=1e-8)
        assert not passed
        assert "not unitary" in msg.lower()

    def test_unitarity_non_square(self) -> None:
        """Non-square matrix fails unitarity check."""
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
        passed, msg = _check_unitarity(matrix, rtol=1e-5, atol=1e-8)
        assert not passed
        assert "square" in msg.lower()

    def test_hermiticity_pass(self) -> None:
        """Hermitian matrix passes hermiticity check."""
        matrix = np.array([[1, 2 + 1j], [2 - 1j, 3]], dtype=complex)
        passed, msg = _check_hermiticity(matrix, rtol=1e-5, atol=1e-8)
        assert passed
        assert "hermitian" in msg.lower()

    def test_hermiticity_fail(self) -> None:
        """Non-Hermitian matrix fails hermiticity check."""
        matrix = np.array([[1, 2], [3, 4]], dtype=complex)
        passed, msg = _check_hermiticity(matrix, rtol=1e-5, atol=1e-8)
        assert not passed
        assert "not hermitian" in msg.lower()

    def test_hermiticity_non_square(self) -> None:
        """Non-square matrix fails hermiticity check."""
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
        passed, msg = _check_hermiticity(matrix, rtol=1e-5, atol=1e-8)
        assert not passed
        assert "square" in msg.lower()

    def test_trace_pass(self) -> None:
        """Unit trace matrix passes trace check."""
        matrix = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        passed, msg = _check_trace(matrix, rtol=1e-5, atol=1e-8)
        assert passed
        assert "trace" in msg.lower()

    def test_trace_fail(self) -> None:
        """Non-unit trace matrix fails trace check."""
        matrix = np.eye(2, dtype=complex)  # trace = 2
        passed, msg = _check_trace(matrix, rtol=1e-5, atol=1e-8)
        assert not passed
        assert "2" in msg

    def test_trace_non_square(self) -> None:
        """Non-square matrix fails trace check."""
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
        passed, msg = _check_trace(matrix, rtol=1e-5, atol=1e-8)
        assert not passed
        assert "square" in msg.lower()

    def test_positivity_pass(self) -> None:
        """Positive semi-definite matrix passes positivity check."""
        matrix = np.array([[1, 0], [0, 0]], dtype=complex)
        passed, msg = _check_positivity(matrix, rtol=1e-5, atol=1e-8)
        assert passed
        assert "positive" in msg.lower()

    def test_positivity_fail(self) -> None:
        """Matrix with negative eigenvalue fails positivity check."""
        matrix = np.array([[0.5, 0.6], [0.6, 0.5]], dtype=complex)
        passed, msg = _check_positivity(matrix, rtol=1e-5, atol=1e-8)
        assert not passed
        assert "negative" in msg.lower()

    def test_positivity_non_square(self) -> None:
        """Non-square matrix fails positivity check."""
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
        passed, msg = _check_positivity(matrix, rtol=1e-5, atol=1e-8)
        assert not passed
        assert "square" in msg.lower()

    def test_normalization_pass(self) -> None:
        """Normalized vector passes normalization check."""
        vector = np.array([1, 0, 0], dtype=complex)
        passed, msg = _check_normalization(vector, rtol=1e-5, atol=1e-8)
        assert passed
        assert "normalized" in msg.lower()

    def test_normalization_fail(self) -> None:
        """Unnormalized vector fails normalization check."""
        vector = np.array([1, 1, 1], dtype=complex)
        passed, msg = _check_normalization(vector, rtol=1e-5, atol=1e-8)
        assert not passed
        assert "not normalized" in msg.lower()


class TestDetermineApplicableChecks:
    """Tests for _determine_applicable_checks()."""

    def test_all_for_square_matrix(self) -> None:
        """'all' returns matrix checks for square matrix."""
        matrix = np.eye(2)
        checks = _determine_applicable_checks(matrix, ["all"])
        assert "unitarity" in checks
        assert "hermiticity" in checks
        assert "trace" in checks
        assert "positivity" in checks

    def test_all_for_vector(self) -> None:
        """'all' returns normalization for vector."""
        vector = np.array([1, 0, 0])
        checks = _determine_applicable_checks(vector, ["all"])
        assert "normalization" in checks

    def test_specific_checks(self) -> None:
        """Specific checks are returned as-is."""
        matrix = np.eye(2)
        checks = _determine_applicable_checks(matrix, ["unitarity", "hermiticity"])
        assert checks == ["unitarity", "hermiticity"]


class TestGetAvailableChecks:
    """Tests for get_available_checks()."""

    def test_returns_dict(self) -> None:
        """Returns dictionary of checks."""
        checks = get_available_checks()
        assert isinstance(checks, dict)
        assert "unitarity" in checks
        assert "all" in checks


class TestRunValidate:
    """Integration tests for run_validate()."""

    def test_validate_npy_file(self, tmp_path: Path) -> None:
        """Validates .npy file."""
        filepath = tmp_path / "test.npy"
        np.save(filepath, np.eye(2, dtype=complex))

        exit_code = run_validate(str(filepath), ["unitarity"])
        assert exit_code == 0

    def test_validate_npy_file_fail(self, tmp_path: Path) -> None:
        """Validation fails for non-unitary matrix."""
        filepath = tmp_path / "test.npy"
        np.save(filepath, np.array([[1, 1], [0, 1]], dtype=complex))

        exit_code = run_validate(str(filepath), ["unitarity"])
        assert exit_code == 1

    def test_validate_npz_file(self, tmp_path: Path) -> None:
        """Validates .npz file."""
        filepath = tmp_path / "test.npz"
        np.savez(filepath, matrix=np.eye(2, dtype=complex), vector=np.array([1, 0]))

        exit_code = run_validate(str(filepath), ["all"])
        # Should have some passes and some fails
        assert exit_code in [0, 1]

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Returns error for missing file."""
        exit_code = run_validate(str(tmp_path / "nonexistent.npy"), ["all"])
        assert exit_code == 1

    def test_unsupported_format(self, tmp_path: Path) -> None:
        """Returns error for unsupported format."""
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")

        exit_code = run_validate(str(filepath), ["all"])
        assert exit_code == 1


class TestValidateCLI:
    """CLI integration tests."""

    def test_validate_help(self) -> None:
        """validate --help shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate physics constraints" in result.output

    def test_validate_npy(self, tmp_path: Path) -> None:
        """CLI validates .npy file."""
        runner = CliRunner()
        filepath = tmp_path / "test.npy"
        np.save(filepath, np.eye(2, dtype=complex))

        result = runner.invoke(cli, ["validate", str(filepath), "--check", "unitarity"])
        assert "PASS" in result.output

    def test_validate_multiple_checks(self, tmp_path: Path) -> None:
        """CLI supports multiple checks."""
        runner = CliRunner()
        filepath = tmp_path / "test.npy"
        np.save(filepath, np.array([[0.5, 0], [0, 0.5]], dtype=complex))

        result = runner.invoke(
            cli,
            [
                "validate",
                str(filepath),
                "-c",
                "hermiticity",
                "-c",
                "trace",
                "-c",
                "positivity",
            ],
        )
        assert result.exit_code == 0
        assert "hermiticity" in result.output.lower()
        assert "trace" in result.output.lower()

    def test_validate_custom_tolerance(self, tmp_path: Path) -> None:
        """CLI supports custom tolerance."""
        runner = CliRunner()
        filepath = tmp_path / "test.npy"
        np.save(filepath, np.eye(2, dtype=complex))

        result = runner.invoke(
            cli,
            [
                "validate",
                str(filepath),
                "--check",
                "all",
                "--rtol",
                "1e-3",
                "--atol",
                "1e-6",
            ],
        )
        assert "1e-03" in result.output or "0.001" in result.output
