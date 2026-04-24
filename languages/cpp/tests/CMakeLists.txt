add_executable(
  agentbible_cpp_tests
  test_conformance.cpp
  test_validators.cpp
  test_provenance.cpp
)

target_link_libraries(
  agentbible_cpp_tests
  PRIVATE
    agentbible_headers
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(agentbible_cpp_tests)
