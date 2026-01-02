// SPDX-License-Identifier: MIT
// Copyright (c) {{YEAR}} {{AUTHOR_NAME}}

#include "core.hpp"

#include <iostream>

int main() {
    std::cout << "{{PROJECT_NAME}} - C++/HPC Research Project\n";
    std::cout << "Build with: cmake -B build && cmake --build build\n";
    std::cout << "Run tests with: ctest --test-dir build\n";
    return 0;
}
