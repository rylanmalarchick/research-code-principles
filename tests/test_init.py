"""Tests for bible init command."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentbible.cli.init import (
    get_template_variables,
    substitute_variables,
    to_python_identifier,
    validate_project_name,
)
from agentbible.cli.main import cli


class TestProjectNameValidation:
    """Tests for validate_project_name()."""

    def test_valid_name(self) -> None:
        """Valid project names are accepted."""
        assert validate_project_name("myproject") == (True, "")
        assert validate_project_name("my-project") == (True, "")
        assert validate_project_name("my_project") == (True, "")
        assert validate_project_name("MyProject123") == (True, "")

    def test_empty_name(self) -> None:
        """Empty name is rejected."""
        is_valid, error = validate_project_name("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_name_with_spaces(self) -> None:
        """Names with spaces are rejected."""
        is_valid, error = validate_project_name("my project")
        assert is_valid is False
        assert "spaces" in error.lower()

    def test_name_starting_with_dash(self) -> None:
        """Names starting with dash are rejected."""
        is_valid, error = validate_project_name("-myproject")
        assert is_valid is False
        assert "start" in error.lower()

    def test_name_starting_with_dot(self) -> None:
        """Names starting with dot are rejected."""
        is_valid, error = validate_project_name(".myproject")
        assert is_valid is False
        assert "start" in error.lower()

    def test_name_starting_with_number(self) -> None:
        """Names starting with number are rejected."""
        is_valid, error = validate_project_name("123project")
        assert is_valid is False
        assert "letter" in error.lower()


class TestPythonIdentifier:
    """Tests for to_python_identifier()."""

    def test_dashes_to_underscores(self) -> None:
        """Dashes are converted to underscores."""
        assert to_python_identifier("my-project") == "my_project"

    def test_lowercase(self) -> None:
        """Names are converted to lowercase."""
        assert to_python_identifier("MyProject") == "myproject"

    def test_mixed(self) -> None:
        """Mixed names are handled correctly."""
        assert to_python_identifier("My-Cool_Project") == "my_cool_project"


class TestVariableSubstitution:
    """Tests for template variable substitution."""

    def test_substitute_single_variable(self) -> None:
        """Single variable is substituted."""
        result = substitute_variables(
            "Project: {{PROJECT_NAME}}", {"PROJECT_NAME": "myproj"}
        )
        assert result == "Project: myproj"

    def test_substitute_multiple_variables(self) -> None:
        """Multiple variables are substituted."""
        variables = {"NAME": "test", "VERSION": "1.0"}
        result = substitute_variables("{{NAME}} v{{VERSION}}", variables)
        assert result == "test v1.0"

    def test_unknown_variable_preserved(self) -> None:
        """Unknown variables are left as-is."""
        result = substitute_variables("{{UNKNOWN}}", {"PROJECT_NAME": "test"})
        assert result == "{{UNKNOWN}}"


class TestGetTemplateVariables:
    """Tests for get_template_variables()."""

    def test_basic_variables(self) -> None:
        """Basic variables are set correctly."""
        variables = get_template_variables("my-project", None, None, None)

        assert variables["PROJECT_NAME"] == "my-project"
        assert variables["PROJECT_NAME_UNDERSCORE"] == "my_project"
        assert "YEAR" in variables
        assert "DATE" in variables

    def test_custom_author(self) -> None:
        """Custom author overrides git config."""
        variables = get_template_variables(
            "test", author="Custom Author", email="custom@example.com", description=None
        )

        assert variables["AUTHOR_NAME"] == "Custom Author"
        assert variables["AUTHOR_EMAIL"] == "custom@example.com"

    def test_custom_description(self) -> None:
        """Custom description is used."""
        variables = get_template_variables(
            "test", author=None, email=None, description="My custom description"
        )

        assert variables["PROJECT_DESCRIPTION"] == "My custom description"


class TestInitCommand:
    """Integration tests for bible init command."""

    def test_init_creates_project(self, tmp_path: Path) -> None:
        """init command creates project directory."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["init", "test-project", "--no-git", "--no-venv"]
            )

            assert result.exit_code == 0
            assert Path("test-project").exists()
            assert Path("test-project/pyproject.toml").exists()
            assert Path("test-project/README.md").exists()
            assert Path("test-project/.cursorrules").exists()

    def test_init_substitutes_variables(self, tmp_path: Path) -> None:
        """init command substitutes template variables."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                [
                    "init",
                    "my-quantum-sim",
                    "--no-git",
                    "--no-venv",
                    "-a",
                    "Test Author",
                    "-e",
                    "test@example.com",
                ],
            )

            assert result.exit_code == 0

            # Check pyproject.toml has substituted values
            # Package name uses underscores (Python convention)
            pyproject = Path("my-quantum-sim/pyproject.toml").read_text()
            assert 'name = "my_quantum_sim"' in pyproject
            assert "Test Author" in pyproject
            assert "test@example.com" in pyproject

            # Check README has project name
            readme = Path("my-quantum-sim/README.md").read_text()
            assert "my-quantum-sim" in readme

    def test_init_creates_src_directory(self, tmp_path: Path) -> None:
        """init command creates src directory with module."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["init", "test-project", "--no-git", "--no-venv"]
            )

            assert result.exit_code == 0
            assert Path("test-project/src").exists()
            assert Path("test-project/src/__init__.py").exists()

    def test_init_creates_tests_directory(self, tmp_path: Path) -> None:
        """init command creates tests directory."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["init", "test-project", "--no-git", "--no-venv"]
            )

            assert result.exit_code == 0
            assert Path("test-project/tests").exists()
            assert Path("test-project/tests/__init__.py").exists()
            assert Path("test-project/tests/conftest.py").exists()

    def test_init_cpp_template(self, tmp_path: Path) -> None:
        """init command creates C++ project from cpp-hpc-cuda template."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                [
                    "init",
                    "my-cpp-project",
                    "-t",
                    "cpp-hpc-cuda",
                    "--no-git",
                    "-a",
                    "Test Author",
                    "-e",
                    "test@example.com",
                ],
            )

            assert result.exit_code == 0
            assert Path("my-cpp-project").exists()
            assert Path("my-cpp-project/CMakeLists.txt").exists()
            assert Path("my-cpp-project/README.md").exists()
            assert Path("my-cpp-project/.cursorrules").exists()
            assert Path("my-cpp-project/include/core.hpp").exists()
            assert Path("my-cpp-project/src/core.cpp").exists()
            assert Path("my-cpp-project/tests/test_core.cpp").exists()

            # Check variable substitution in CMakeLists.txt
            cmake = Path("my-cpp-project/CMakeLists.txt").read_text()
            assert "my_cpp_project" in cmake  # PROJECT_NAME_UNDERSCORE
            assert "my-cpp-project" in cmake  # PROJECT_NAME

            # Check README has project name
            readme = Path("my-cpp-project/README.md").read_text()
            assert "my-cpp-project" in readme

            # Check next steps show cmake instructions
            assert "cmake" in result.output.lower()

    def test_init_rejects_existing_directory(self, tmp_path: Path) -> None:
        """init command fails if directory exists."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("existing-project").mkdir(parents=True)

            result = runner.invoke(
                cli, ["init", "existing-project", "--no-git", "--no-venv"]
            )

            assert result.exit_code == 1
            assert "already exists" in result.output

    def test_init_force_overwrites(self, tmp_path: Path) -> None:
        """init --force overwrites existing directory."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("existing-project").mkdir(parents=True)
            Path("existing-project/old-file.txt").write_text("old content")

            result = runner.invoke(
                cli, ["init", "existing-project", "--no-git", "--no-venv", "--force"]
            )

            assert result.exit_code == 0
            assert not Path("existing-project/old-file.txt").exists()
            assert Path("existing-project/pyproject.toml").exists()

    def test_init_rejects_invalid_name(self, tmp_path: Path) -> None:
        """init command rejects invalid project names."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["init", "-invalid-name", "--no-git", "--no-venv"]
            )

            assert result.exit_code != 0

    def test_init_unknown_template(self, tmp_path: Path) -> None:
        """init command rejects unknown template."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                [
                    "init",
                    "test-project",
                    "-t",
                    "unknown-template",
                    "--no-git",
                    "--no-venv",
                ],
            )

            # Click should reject this before our code runs
            assert result.exit_code != 0

    def test_init_shows_next_steps(self, tmp_path: Path) -> None:
        """init command shows next steps."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["init", "test-project", "--no-git", "--no-venv"]
            )

            assert result.exit_code == 0
            assert "Next steps" in result.output
            assert "cd test-project" in result.output


class TestInitGitVenv:
    """Tests for git and venv initialization (skipped in CI if not available)."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip git tests in CI",
    )
    def test_init_creates_git_repo(self, tmp_path: Path) -> None:
        """init command initializes git repo by default."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "test-project", "--no-venv"])

            assert result.exit_code == 0
            assert Path("test-project/.git").exists()

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip venv tests in CI",
    )
    def test_init_creates_venv(self, tmp_path: Path) -> None:
        """init command creates venv for Python templates."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "test-project", "--no-git"])

            assert result.exit_code == 0
            assert Path("test-project/.venv").exists()
