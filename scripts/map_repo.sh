#!/bin/bash
# map_repo.sh - Generate a concise repository map for AI context
#
# This script creates a "map" of the repository showing:
# - Directory structure
# - File sizes (to estimate token usage)
# - Class/function stubs (for Python/C++)
#
# Usage:
#   ./scripts/map_repo.sh > repo_map.txt
#   cat repo_map.txt | wc -l  # Check line count
#
# The output is designed to give an AI agent enough context to
# understand the architecture without consuming 50k+ tokens.

set -e

# Configuration
MAX_DEPTH=${MAX_DEPTH:-4}
EXCLUDE_DIRS=".git|node_modules|__pycache__|.mypy_cache|.pytest_cache|build|dist|.venv|venv"

echo "# Repository Map"
echo "# Generated: $(date -Iseconds)"
echo "# Directory: $(pwd)"
echo ""

# ============================================================================
# Directory Structure
# ============================================================================
echo "## Directory Structure"
echo ""
echo '```'
# Use find with depth limit, exclude common non-essential directories
find . -maxdepth "$MAX_DEPTH" -type d \
    | grep -vE "($EXCLUDE_DIRS)" \
    | sort \
    | sed 's|^./||' \
    | while read -r dir; do
        depth=$(echo "$dir" | tr -cd '/' | wc -c)
        indent=$(printf '%*s' "$((depth * 2))" '')
        basename=$(basename "$dir")
        echo "${indent}${basename}/"
    done
echo '```'
echo ""

# ============================================================================
# File Listing with Sizes
# ============================================================================
echo "## Files (by size)"
echo ""
echo "| File | Lines | Size |"
echo "|------|-------|------|"

find . -type f \( -name "*.py" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cu" -o -name "*.md" -o -name "*.toml" -o -name "*.yaml" -o -name "*.yml" \) \
    | grep -vE "($EXCLUDE_DIRS)" \
    | while read -r file; do
        lines=$(wc -l < "$file" 2>/dev/null || echo "0")
        size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "?")
        echo "| ${file#./} | $lines | $size |"
    done | sort -t'|' -k3 -rn | head -50

echo ""

# ============================================================================
# Python Class/Function Stubs
# ============================================================================
echo "## Python Definitions"
echo ""

find . -name "*.py" -type f | grep -vE "($EXCLUDE_DIRS)" | sort | while read -r file; do
    # Extract class and function definitions
    defs=$(grep -nE "^(class |def |async def )" "$file" 2>/dev/null | head -20)
    if [ -n "$defs" ]; then
        echo "### ${file#./}"
        echo '```python'
        echo "$defs" | sed 's/:.*/:/'  # Truncate after colon
        echo '```'
        echo ""
    fi
done

# ============================================================================
# C++ Class/Function Stubs
# ============================================================================
echo "## C++ Definitions"
echo ""

find . \( -name "*.hpp" -o -name "*.h" -o -name "*.cpp" \) -type f | grep -vE "($EXCLUDE_DIRS)" | sort | while read -r file; do
    # Extract class definitions and public methods
    defs=$(grep -nE "^(class |struct |template|.*\(.*\).*\{$|.*\(.*\);$)" "$file" 2>/dev/null | grep -v "^[[:space:]]*#" | head -20)
    if [ -n "$defs" ]; then
        echo "### ${file#./}"
        echo '```cpp'
        echo "$defs"
        echo '```'
        echo ""
    fi
done

echo ""
echo "# End of Repository Map"
