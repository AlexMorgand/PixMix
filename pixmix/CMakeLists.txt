cmake_minimum_required(VERSION 3.16)
project(pixmix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# pkg-config selects the CPU OpenCV. A CUDA-enabled OpenCVConfig under /usr/local
# can fail configuration when its exact CUDA version is not installed.
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)
find_package(OpenMP)

add_library(pixmix_core STATIC
    src/features.cc
    src/inpainter.cc
    src/selector.cc
    src/tracker.cc
    src/pipeline.cc)
target_include_directories(pixmix_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pixmix_core PUBLIC PkgConfig::OPENCV)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pixmix_core PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_options(pixmix_core PRIVATE -Wall -Wextra -Wno-unused-parameter)

add_executable(pixmix src/main.cc)
target_link_libraries(pixmix PRIVATE pixmix_core)
target_compile_options(pixmix PRIVATE -Wall -Wextra -Wno-unused-parameter)

add_executable(pixmix_tests tests/test_pixmix.cc)
target_link_libraries(pixmix_tests PRIVATE pixmix_core)
target_compile_options(pixmix_tests PRIVATE -Wall -Wextra -Wno-unused-parameter)

enable_testing()
add_test(NAME pixmix_tests COMMAND pixmix_tests)
