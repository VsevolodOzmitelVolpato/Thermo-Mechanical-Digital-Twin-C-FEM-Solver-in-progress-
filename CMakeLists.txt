cmake_minimum_required(VERSION 3.20)
project(thermomech_twin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1) Your library: every .cpp except the tests.
#    A missing "return" becomes a compile error instead of a silent bug.
add_library(twin_core
    src/CSR_and_Vectors.cpp
    src/JacobiPreconditioner.cpp
    src/CGSolver.cpp
    src/Fem1D.cpp
    src/Thermoelastic1D.cpp
    src/Mesh2D.cpp
    src/Fem2D.cpp
    src/Triplets.cpp
    src/Assembly2D.cpp
    src/Dirichlet2D.cpp
    src/Solver2D.cpp)
target_include_directories(twin_core PUBLIC include)
target_compile_options(twin_core PRIVATE -Wall -Wextra -Werror=return-type)
# Programma che stampa le tabelle di convergenza (per il README).
add_executable(convergence_study apps/convergence_study.cpp)
target_link_libraries(convergence_study PRIVATE twin_core)
target_compile_options(convergence_study PRIVATE -Wall -Wextra)

# 2) GoogleTest: CMake downloads it for you the first time you configure.
include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
FetchContent_MakeAvailable(googletest)

# 3) The test program, linked against your library.
#    Every new test file goes in this list.
enable_testing()
add_executable(unit_tests
    tests/test_csr.cpp
    tests/test_cg.cpp
    tests/test_fem1d.cpp
    tests/test_thermoelastic1d.cpp
    tests/test_mesh2d.cpp
    tests/test_fem2d.cpp
    tests/test_triplets.cpp
    tests/test_assembly2d.cpp
    tests/test_dirichlet2d.cpp
    tests/test_solver2d.cpp)
target_link_libraries(unit_tests PRIVATE twin_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(unit_tests)
