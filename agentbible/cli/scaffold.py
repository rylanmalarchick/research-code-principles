"""Implementation of the bible scaffold command.

Generates module stubs with proper docstrings, type hints, and test files.
Follows AgentBible principles: specification before code, full type hints,
Google-style docstrings.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    pass

console = Console()


def parse_fields(fields_str: str) -> list[tuple[str, str]]:
    """Parse 'name:type,name:type' into list of tuples.

    Args:
        fields_str: Comma-separated field definitions like "x:int,y:float".

    Returns:
        List of (name, type) tuples.

    Raises:
        ValueError: If field format is invalid.
    """
    result: list[tuple[str, str]] = []
    for field in fields_str.split(","):
        field = field.strip()
        if not field:
            continue
        if ":" not in field:
            raise ValueError(f"Invalid field format: {field!r}. Expected 'name:type'")
        name, type_ = field.split(":", 1)
        result.append((name.strip(), type_.strip()))
    return result


def parse_methods(methods_str: str) -> list[str]:
    """Parse comma-separated method names.

    Args:
        methods_str: Comma-separated method names like "run,validate,process".

    Returns:
        List of method names.
    """
    return [m.strip() for m in methods_str.split(",") if m.strip()]


def to_module_name(filepath: str) -> str:
    """Extract module name from filepath.

    Args:
        filepath: Path like "src/optimizer.py".

    Returns:
        Module name like "optimizer".
    """
    return Path(filepath).stem


def to_test_path(filepath: str) -> Path:
    """Convert source path to test path.

    Args:
        filepath: Path like "src/optimizer.py".

    Returns:
        Test path like "tests/test_optimizer.py".

    Examples:
        >>> to_test_path("src/bridge.py")
        PosixPath('tests/test_bridge.py')
        >>> to_test_path("mymodule.py")
        PosixPath('tests/test_mymodule.py')
    """
    module_name = to_module_name(filepath)
    return Path("tests") / f"test_{module_name}.py"


def generate_module_header(module_name: str) -> str:
    """Generate module docstring and imports.

    Args:
        module_name: Name of the module.

    Returns:
        Module header with docstring and standard imports.
    """
    return f'''"""TODO: {module_name} module description.

This module provides...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

'''


def generate_class_stub(
    class_name: str,
    methods: list[str] | None = None,
) -> str:
    """Generate a class stub with docstrings and type hints.

    Args:
        class_name: Name of the class.
        methods: Optional list of method names to generate.

    Returns:
        Class definition as a string.
    """
    methods = methods or []
    lines = [
        f"class {class_name}:",
        f'    """TODO: Describe {class_name}.',
        "",
        "    Attributes:",
        "        TODO: List attributes.",
        '    """',
        "",
        "    def __init__(self) -> None:",
        '        """Initialize the instance.',
        "",
        "        Args:",
        "            TODO: Add arguments.",
        "",
        "        Raises:",
        "            TODO: Document exceptions.",
        '        """',
        "        raise NotImplementedError",
    ]

    for method in methods:
        lines.extend(
            [
                "",
                f"    def {method}(self) -> None:",
                f'        """TODO: Describe {method}.',
                "",
                "        Returns:",
                "            TODO: Document return value.",
                "",
                "        Raises:",
                "            TODO: Document exceptions.",
                '        """',
                "        raise NotImplementedError",
            ]
        )

    return "\n".join(lines)


def generate_dataclass_stub(
    class_name: str,
    fields: list[tuple[str, str]],
    with_validation: bool = False,
) -> str:
    """Generate a dataclass stub with optional validation.

    Args:
        class_name: Name of the dataclass.
        fields: List of (name, type) tuples.
        with_validation: If True, add __post_init__ validation.

    Returns:
        Dataclass definition as a string.
    """
    field_lines = []
    for name, type_ in fields:
        field_lines.append(f"    {name}: {type_}")

    # Generate docstring with attributes
    attr_docs = []
    for name, type_ in fields:
        attr_docs.append(f"        {name}: TODO: Describe {name}.")

    # Generate validation if requested
    validation_lines = []
    if with_validation:
        numeric_fields = [
            name for name, type_ in fields if type_ in ("int", "float")
        ]
        if numeric_fields:
            validation_lines.append("    def __post_init__(self) -> None:")
            validation_lines.append('        """Validate field values."""')
            for name in numeric_fields:
                validation_lines.extend(
                    [
                        f"        if self.{name} < 0:",
                        f'            raise ValueError(f"{name} must be non-negative, got {{self.{name}}}")',
                    ]
                )
        else:
            validation_lines.append("    def __post_init__(self) -> None:")
            validation_lines.append('        """Validate field values."""')
            validation_lines.append("        pass")

    result = f'''from dataclasses import dataclass


@dataclass
class {class_name}:
    """TODO: Describe {class_name}.

    Attributes:
{chr(10).join(attr_docs)}
    """

{chr(10).join(field_lines)}
'''

    if validation_lines:
        result += "\n" + "\n".join(validation_lines) + "\n"

    return result


def generate_function_stub(func_name: str) -> str:
    """Generate a function stub with docstring and type hints.

    Args:
        func_name: Name of the function.

    Returns:
        Function definition as a string.
    """
    return f'''def {func_name}() -> None:
    """TODO: Describe {func_name}.

    Args:
        TODO: Add arguments.

    Returns:
        TODO: Document return value.

    Raises:
        TODO: Document exceptions.
    """
    raise NotImplementedError
'''


def generate_test_stub(
    module_name: str,
    class_name: str | None = None,
    dataclass_name: str | None = None,
    functions: list[str] | None = None,
    methods: list[str] | None = None,
) -> str:
    """Generate a test file stub.

    Args:
        module_name: Name of the module being tested.
        class_name: Optional class to generate tests for.
        dataclass_name: Optional dataclass to generate tests for.
        functions: Optional list of functions to generate tests for.
        methods: Optional list of methods to generate tests for (with class).

    Returns:
        Test file content as a string.
    """
    lines = [
        f'"""Tests for {module_name} module."""',
        "",
        "import pytest",
        "",
    ]

    # Add import comment
    if class_name:
        lines.append(f"# TODO: Uncomment after implementation")
        lines.append(f"# from <package>.{module_name} import {class_name}")
        lines.append("")
    elif dataclass_name:
        lines.append(f"# TODO: Uncomment after implementation")
        lines.append(f"# from <package>.{module_name} import {dataclass_name}")
        lines.append("")
    elif functions:
        func_imports = ", ".join(functions)
        lines.append(f"# TODO: Uncomment after implementation")
        lines.append(f"# from <package>.{module_name} import {func_imports}")
        lines.append("")

    # Generate test class for classes
    if class_name:
        lines.extend(
            [
                "",
                f"class Test{class_name}:",
                f'    """Tests for {class_name} class."""',
                "",
                "    def test_init(self) -> None:",
                '        """Test initialization."""',
                '        pytest.skip("TODO: Implement")',
            ]
        )
        # Add method tests
        for method in methods or []:
            lines.extend(
                [
                    "",
                    f"    def test_{method}(self) -> None:",
                    f'        """Test {method} method."""',
                    f'        pytest.skip("TODO: Implement")',
                ]
            )

    # Generate test class for dataclasses
    if dataclass_name:
        lines.extend(
            [
                "",
                f"class Test{dataclass_name}:",
                f'    """Tests for {dataclass_name} dataclass."""',
                "",
                "    def test_creation(self) -> None:",
                '        """Test dataclass creation."""',
                '        pytest.skip("TODO: Implement")',
                "",
                "    def test_validation(self) -> None:",
                '        """Test field validation."""',
                '        pytest.skip("TODO: Implement")',
            ]
        )

    # Generate function tests
    for func in functions or []:
        lines.extend(
            [
                "",
                f"def test_{func}() -> None:",
                f'    """Test {func} function."""',
                '    pytest.skip("TODO: Implement")',
            ]
        )

    return "\n".join(lines) + "\n"


def run_scaffold(
    filepath: str,
    class_name: str | None = None,
    dataclass_name: str | None = None,
    methods: str | None = None,
    fields: str | None = None,
    functions: str | None = None,
    no_test: bool = False,
    validate: bool = False,
) -> int:
    """Execute the scaffold command.

    Args:
        filepath: Path to the module to create.
        class_name: Optional class name to generate.
        dataclass_name: Optional dataclass name to generate.
        methods: Comma-separated method names (with --class).
        fields: Comma-separated field:type pairs (with --dataclass).
        functions: Comma-separated function names.
        no_test: If True, skip test file generation.
        validate: If True, add validation to dataclass.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    source_path = Path(filepath)
    module_name = to_module_name(filepath)
    test_path = to_test_path(filepath)

    # Check if file already exists
    if source_path.exists():
        console.print(f"[red]Error:[/] File already exists: {source_path}")
        console.print("Use --force to overwrite (not implemented yet)")
        return 1

    # Build source file content
    content_parts: list[str] = [generate_module_header(module_name)]

    method_list: list[str] = []
    field_list: list[tuple[str, str]] = []
    function_list: list[str] = []

    # Parse options
    if methods:
        method_list = parse_methods(methods)
    if fields:
        try:
            field_list = parse_fields(fields)
        except ValueError as e:
            console.print(f"[red]Error:[/] {e}")
            return 1
    if functions:
        function_list = parse_methods(functions)

    # Generate content based on options
    if dataclass_name:
        content_parts.append(
            generate_dataclass_stub(dataclass_name, field_list, with_validation=validate)
        )
    elif class_name:
        content_parts.append(generate_class_stub(class_name, method_list))
    elif function_list:
        for func in function_list:
            content_parts.append(generate_function_stub(func))
            content_parts.append("")
    else:
        # Default: just create empty module
        content_parts.append("# TODO: Add module content\n")

    source_content = "\n".join(content_parts)

    # Ensure parent directory exists
    source_path.parent.mkdir(parents=True, exist_ok=True)

    # Write source file
    source_path.write_text(source_content)
    console.print(f"[green]Created:[/] {source_path}")

    # Generate test file
    if not no_test:
        test_content = generate_test_stub(
            module_name=module_name,
            class_name=class_name,
            dataclass_name=dataclass_name,
            functions=function_list if function_list else None,
            methods=method_list if method_list else None,
        )

        # Ensure tests directory exists
        test_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if test file exists
        if test_path.exists():
            console.print(f"[yellow]Warning:[/] Test file already exists: {test_path}")
        else:
            test_path.write_text(test_content)
            console.print(f"[green]Created:[/] {test_path}")

    # Print next steps
    console.print()
    console.print("[bold]Next steps:[/]")
    console.print(f"  1. Implement the code in {source_path}")
    if not no_test:
        console.print(f"  2. Update import in {test_path}")
        console.print(f"  3. Implement tests in {test_path}")
    console.print("  4. Run: pytest tests/")

    return 0
