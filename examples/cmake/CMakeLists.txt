# Consumer example: links examples/c/main.c against zigjit::zigjit.
#   cmake -S examples/cmake -B build            # uses ../.. via add_subdirectory
#   cmake -S examples/cmake -B build -DZIGJIT_USE_FIND_PACKAGE=ON -DCMAKE_PREFIX_PATH=<prefix>
cmake_minimum_required(VERSION 3.20)
project(zigjit_cmake_example LANGUAGES C)
option(ZIGJIT_USE_FIND_PACKAGE "Use an installed zigjit via find_package" OFF)
if(ZIGJIT_USE_FIND_PACKAGE)
  find_package(zigjit 0.1 REQUIRED CONFIG)
else()
  add_subdirectory(../.. zigjit)
endif()
add_executable(example_c ../c/main.c)
set_target_properties(example_c PROPERTIES C_STANDARD 11)
target_link_libraries(example_c PRIVATE zigjit::zigjit)
add_executable(example_c_shared ../c/main.c)
set_target_properties(example_c_shared PROPERTIES C_STANDARD 11)
target_link_libraries(example_c_shared PRIVATE zigjit::zigjit_shared)
