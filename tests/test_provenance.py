"""Tests for provenance module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agentbible.provenance.hdf5 import get_provenance_metadata

# Only import HDF5-specific functions if h5py is available
h5py = pytest.importorskip("h5py")

from agentbible.provenance import load_with_metadata, save_with_metadata  # noqa: E402


class TestGetProvenanceMetadata:
    """Tests for get_provenance_metadata()."""

    def test_returns_dict(self) -> None:
        """Returns a dictionary."""
        metadata = get_provenance_metadata()
        assert isinstance(metadata, dict)

    def test_has_timestamp(self) -> None:
        """Includes timestamp in ISO 8601 format."""
        metadata = get_provenance_metadata()
        assert "timestamp" in metadata
        assert "T" in metadata["timestamp"]  # ISO 8601 format

    def test_has_git_fields(self) -> None:
        """Includes git-related fields."""
        metadata = get_provenance_metadata()
        assert "git_sha" in metadata
        assert "git_branch" in metadata
        assert "git_dirty" in metadata

    def test_has_system_info(self) -> None:
        """Includes system information."""
        metadata = get_provenance_metadata()
        assert "hostname" in metadata
        assert "platform" in metadata
        assert "python_version" in metadata

    def test_has_package_versions(self) -> None:
        """Includes package versions."""
        metadata = get_provenance_metadata()
        assert "packages" in metadata
        assert "numpy" in metadata["packages"]

    def test_custom_description(self) -> None:
        """Custom description is included."""
        metadata = get_provenance_metadata(description="Test data")
        assert metadata["description"] == "Test data"

    def test_extra_metadata(self) -> None:
        """Extra metadata is included."""
        extra = {"experiment_id": 42, "parameters": {"alpha": 0.5}}
        metadata = get_provenance_metadata(extra=extra)
        assert metadata["extra"] == extra


class TestSaveWithMetadata:
    """Tests for save_with_metadata()."""

    def test_saves_single_array(self, tmp_path: Path) -> None:
        """Saves a single array to HDF5."""
        filepath = tmp_path / "test.h5"
        data = {"array": np.eye(2)}

        save_with_metadata(filepath, data)

        assert filepath.exists()
        with h5py.File(filepath, "r") as f:
            assert "array" in f
            np.testing.assert_array_equal(f["array"][()], np.eye(2))

    def test_saves_multiple_arrays(self, tmp_path: Path) -> None:
        """Saves multiple arrays to HDF5."""
        filepath = tmp_path / "test.h5"
        data = {
            "matrix": np.eye(3),
            "vector": np.array([1, 2, 3]),
            "scalar": np.array(42.0),
        }

        save_with_metadata(filepath, data)

        with h5py.File(filepath, "r") as f:
            assert len(f.keys()) == 3
            np.testing.assert_array_equal(f["matrix"][()], np.eye(3))
            np.testing.assert_array_equal(f["vector"][()], np.array([1, 2, 3]))

    def test_saves_provenance_metadata(self, tmp_path: Path) -> None:
        """Saves provenance metadata as attribute."""
        filepath = tmp_path / "test.h5"
        data = {"array": np.eye(2)}

        save_with_metadata(filepath, data, description="Test description")

        with h5py.File(filepath, "r") as f:
            assert "provenance" in f.attrs
            metadata = json.loads(f.attrs["provenance"])
            assert metadata["description"] == "Test description"
            assert "timestamp" in metadata
            assert "packages" in metadata

    def test_empty_data_raises(self, tmp_path: Path) -> None:
        """Empty data dictionary raises ValueError."""
        filepath = tmp_path / "test.h5"

        with pytest.raises(ValueError, match="empty"):
            save_with_metadata(filepath, {})

    def test_non_array_raises(self, tmp_path: Path) -> None:
        """Non-array value raises ValueError."""
        filepath = tmp_path / "test.h5"

        with pytest.raises(ValueError, match="numpy array"):
            save_with_metadata(filepath, {"bad": [1, 2, 3]})  # type: ignore[dict-item]

    def test_complex_arrays(self, tmp_path: Path) -> None:
        """Saves complex-valued arrays correctly."""
        filepath = tmp_path / "test.h5"
        data = {"complex": np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])}

        save_with_metadata(filepath, data)

        with h5py.File(filepath, "r") as f:
            loaded = f["complex"][()]
            np.testing.assert_array_equal(loaded, data["complex"])


class TestLoadWithMetadata:
    """Tests for load_with_metadata()."""

    def test_loads_data_and_metadata(self, tmp_path: Path) -> None:
        """Loads both data and metadata."""
        filepath = tmp_path / "test.h5"
        original_data = {"matrix": np.eye(2), "vector": np.array([1, 2, 3])}

        save_with_metadata(filepath, original_data, description="Test")

        data, metadata = load_with_metadata(filepath)

        assert "matrix" in data
        assert "vector" in data
        np.testing.assert_array_equal(data["matrix"], np.eye(2))
        assert metadata["description"] == "Test"

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError."""
        filepath = tmp_path / "nonexistent.h5"

        with pytest.raises(FileNotFoundError):
            load_with_metadata(filepath)

    def test_roundtrip_preserves_data(self, tmp_path: Path) -> None:
        """Save and load preserves data exactly."""
        filepath = tmp_path / "test.h5"
        original = {
            "dense": np.random.rand(10, 10),
            "complex": np.random.rand(5, 5) + 1j * np.random.rand(5, 5),
            "integers": np.arange(100),
        }

        save_with_metadata(filepath, original, description="Roundtrip test")

        loaded, _ = load_with_metadata(filepath)

        for key in original:
            np.testing.assert_array_equal(loaded[key], original[key])

    def test_metadata_contains_git_info(self, tmp_path: Path) -> None:
        """Loaded metadata contains git information."""
        filepath = tmp_path / "test.h5"
        save_with_metadata(filepath, {"arr": np.eye(2)})

        _, metadata = load_with_metadata(filepath)

        # These may be None if not in a git repo, but keys should exist
        assert "git_sha" in metadata
        assert "git_branch" in metadata
        assert "git_dirty" in metadata
