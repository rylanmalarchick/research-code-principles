cmake_minimum_required(VERSION 3.24)
project(agentbible_cpp LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Werror)

find_package(CUDA QUIET)
if(CUDA_FOUND)
  message(STATUS "CUDA detected: ${CUDA_VERSION_STRING}")
endif()

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../..
  OUTPUT_VARIABLE AGENTBIBLE_GIT_SHA
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nlohmann_json CONFIG QUIET)
find_package(GTest CONFIG QUIET)

if(NOT nlohmann_json_FOUND OR NOT GTest_FOUND)
  set(CPM_DOWNLOAD_VERSION 0.40.2)
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM.cmake")
  if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    file(
      DOWNLOAD
      https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
      ${CPM_DOWNLOAD_LOCATION}
    )
  endif()
  if(EXISTS ${CPM_DOWNLOAD_LOCATION})
    include(${CPM_DOWNLOAD_LOCATION})
  endif()
endif()

if(NOT nlohmann_json_FOUND)
  if(COMMAND CPMAddPackage)
    CPMAddPackage("gh:nlohmann/json@3.11.3")
  else()
    message(FATAL_ERROR "nlohmann_json not found and CPM.cmake is unavailable")
  endif()
endif()

if(NOT GTest_FOUND)
  if(COMMAND CPMAddPackage)
    CPMAddPackage("gh:google/googletest@1.15.2")
  else()
    message(FATAL_ERROR "GTest not found and CPM.cmake is unavailable")
  endif()
endif()

add_library(agentbible_headers INTERFACE)
target_include_directories(agentbible_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(agentbible_headers INTERFACE nlohmann_json::nlohmann_json)
target_compile_definitions(agentbible_headers INTERFACE AGENTBIBLE_GIT_SHA="${AGENTBIBLE_GIT_SHA}")

add_subdirectory(tests)
