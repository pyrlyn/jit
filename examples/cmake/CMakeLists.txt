# Consumer example: links examples/c/main.c against pyrlyn-jit::pyrlyn-jit.
#   cmake -S examples/cmake -B build            # uses ../.. via add_subdirectory
#   cmake -S examples/cmake -B build -DPYRLYN_JIT_USE_FIND_PACKAGE=ON -DCMAKE_PREFIX_PATH=<prefix>
cmake_minimum_required(VERSION 3.20)
project(pyrlyn_jit_cmake_example LANGUAGES C)
option(PYRLYN_JIT_USE_FIND_PACKAGE "Use an installed pyrlyn-jit via find_package" OFF)
if(PYRLYN_JIT_USE_FIND_PACKAGE)
  find_package(pyrlyn-jit 0.1 REQUIRED CONFIG)
else()
  add_subdirectory(../.. pyrlyn-jit)
endif()
add_executable(example_c ../c/main.c)
set_target_properties(example_c PROPERTIES C_STANDARD 11)
target_link_libraries(example_c PRIVATE pyrlyn-jit::pyrlyn-jit)
add_executable(example_c_shared ../c/main.c)
set_target_properties(example_c_shared PROPERTIES C_STANDARD 11)
target_link_libraries(example_c_shared PRIVATE pyrlyn-jit::pyrlyn-jit_shared)
