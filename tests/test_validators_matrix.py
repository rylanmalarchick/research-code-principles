"""Tests for direct matrix validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from agentbible.errors import DensityMatrixError, SymmetryError, ValidationError
from agentbible.validators import (
    check_density_matrix,
    check_hermitian,
    check_positive_definite,
    check_positive_semidefinite,
    check_symmetric,
    validate_density_matrix,
    validate_positive_definite,
    validate_positive_semidefinite,
    validate_symmetric,
)


class TestDirectMatrixChecks:
    """Tests for direct matrix check helpers."""

    def test_check_symmetric_and_non_square_error(self) -> None:
        symmetric = np.array([[1.0, 2.0], [2.0, 3.0]])
        assert np.array_equal(check_symmetric(symmetric), symmetric)

        with pytest.raises(SymmetryError, match="Square matrix"):
            check_symmetric(np.array([[1.0, 2.0, 3.0]]))

    def test_check_symmetric_warns_when_not_strict(self) -> None:
        bad = np.array([[1.0, 0.0], [1.0, 3.0]])

        with pytest.warns(RuntimeWarning, match="IS NOT SYMMETRIC"):
            result = check_symmetric(bad, strict=False)

        assert np.array_equal(result, bad)

    def test_check_hermitian_warns_when_not_strict(self) -> None:
        bad = np.array([[1.0, 1.0 + 1.0j], [0.0, 1.0]], dtype=complex)

        with pytest.warns(RuntimeWarning, match="IS NOT HERMITIAN"):
            result = check_hermitian(bad, strict=False)

        assert np.array_equal(result, bad)

    def test_check_positive_definite_and_semidefinite(self) -> None:
        positive_definite = np.array([[2.0, 1.0], [1.0, 2.0]])
        positive_semidefinite = np.array([[1.0, 0.0], [0.0, 0.0]])

        assert np.array_equal(
            check_positive_definite(positive_definite),
            positive_definite,
        )
        assert np.array_equal(
            check_positive_semidefinite(positive_semidefinite),
            positive_semidefinite,
        )

    def test_check_positive_definite_warns_when_not_strict(self) -> None:
        bad = np.array([[0.0, 1.0], [1.0, 0.0]])

        with pytest.warns(RuntimeWarning, match="IS NOT POSITIVE DEFINITE"):
            result = check_positive_definite(bad, strict=False)

        assert np.array_equal(result, bad)

    def test_check_positive_semidefinite_warns_when_not_strict(self) -> None:
        bad = np.array([[0.5, 0.6], [0.6, 0.5]], dtype=complex)

        with pytest.warns(RuntimeWarning, match="IS NOT POSITIVE SEMI-DEFINITE"):
            result = check_positive_semidefinite(bad, strict=False)

        assert np.array_equal(result, bad)

    def test_check_density_matrix_warns_on_trace_violation(self) -> None:
        bad_trace = np.eye(2, dtype=complex) / 4

        with pytest.warns(RuntimeWarning, match="DOES NOT HAVE UNIT TRACE"):
            result = check_density_matrix(bad_trace, strict=False)

        assert np.array_equal(result, bad_trace)

    def test_check_density_matrix_non_square_raises(self) -> None:
        with pytest.raises(DensityMatrixError, match="Square matrix"):
            check_density_matrix(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


class TestMatrixDecorators:
    """Tests for decorator factory forms added in matrix validators."""

    def test_validate_symmetric_with_parentheses(self) -> None:
        @validate_symmetric()
        def make_matrix() -> np.ndarray:
            return np.array([[1.0, 2.0], [2.0, 1.0]])

        assert np.array_equal(make_matrix(), np.array([[1.0, 2.0], [2.0, 1.0]]))

    def test_validate_positive_definite_with_parentheses(self) -> None:
        @validate_positive_definite()
        def make_matrix() -> np.ndarray:
            return np.array([[2.0, 1.0], [1.0, 2.0]])

        assert np.array_equal(make_matrix(), np.array([[2.0, 1.0], [1.0, 2.0]]))

    def test_validate_positive_semidefinite_with_parentheses(self) -> None:
        @validate_positive_semidefinite()
        def make_matrix() -> np.ndarray:
            return np.array([[1.0, 0.0], [0.0, 0.0]])

        assert np.array_equal(make_matrix(), np.array([[1.0, 0.0], [0.0, 0.0]]))

    def test_validate_density_matrix_with_parentheses(
        self,
        mixed_state_density_matrix: np.ndarray,
    ) -> None:
        @validate_density_matrix()
        def make_state() -> np.ndarray:
            return mixed_state_density_matrix

        assert np.array_equal(make_state(), mixed_state_density_matrix)

    def test_validate_positive_definite_raises_on_invalid_matrix(self) -> None:
        @validate_positive_definite
        def make_bad_matrix() -> np.ndarray:
            return np.array([[0.0, 1.0], [1.0, 0.0]])

        with pytest.raises(ValidationError, match="IS NOT POSITIVE DEFINITE"):
            make_bad_matrix()
