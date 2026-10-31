# Thin CMake adapter: build.zig stays the primary build. This file only drives `zig build`
# and exposes the result as IMPORTED targets pyrlyn-jit::pyrlyn-jit (static) and pyrlyn-jit::pyrlyn-jit_shared.
# Works with add_subdirectory, FetchContent, CPM and (after install) find_package(pyrlyn-jit).
cmake_minimum_required(VERSION 3.20)
project(pyrlyn-jit VERSION 0.1.0 LANGUAGES C)

find_program(PYRLYN_JIT_ZIG_EXECUTABLE zig REQUIRED DOC "Zig compiler (0.16) used to build pyrlyn-jit")
set(PYRLYN_JIT_OPTIMIZE "" CACHE STRING "zig -Doptimize (Debug, ReleaseSafe, ReleaseFast, ReleaseSmall); empty = from CMAKE_BUILD_TYPE")
set(PYRLYN_JIT_TARGETS "x86_64,aarch64" CACHE STRING "Backends to compile in (-Dtargets)")
option(PYRLYN_JIT_INSTALL "Install pyrlyn-jit with the CMake install step" ${PROJECT_IS_TOP_LEVEL})

if(PYRLYN_JIT_OPTIMIZE)
  set(_zj_opt "${PYRLYN_JIT_OPTIMIZE}")
elseif(CMAKE_BUILD_TYPE MATCHES "^(Release|MinSizeRel)$")
  set(_zj_opt ReleaseFast)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(_zj_opt ReleaseSafe)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(_zj_opt Debug)
else()
  set(_zj_opt ReleaseSafe)
endif()

set(_zj_prefix "${CMAKE_CURRENT_BINARY_DIR}/zig-out")
set(_zj_static "${_zj_prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}pyrlyn-jit${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(_zj_shared "${_zj_prefix}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}pyrlyn-jit${CMAKE_SHARED_LIBRARY_SUFFIX}")
file(MAKE_DIRECTORY "${_zj_prefix}/include")
file(GLOB_RECURSE _zj_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.zig" "${CMAKE_CURRENT_SOURCE_DIR}/include/*")

add_custom_command(
  OUTPUT "${_zj_static}" "${_zj_shared}"
  COMMAND "${PYRLYN_JIT_ZIG_EXECUTABLE}" build install -Doptimize=${_zj_opt} -Dtargets=${PYRLYN_JIT_TARGETS}
          -Dversion=${PROJECT_VERSION} --prefix "${_zj_prefix}"
          --cache-dir "${CMAKE_CURRENT_BINARY_DIR}/zig-cache"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/build.zig" ${_zj_sources}
  COMMENT "zig build (pyrlyn-jit, -Doptimize=${_zj_opt})"
  VERBATIM)
add_custom_target(pyrlyn_jit_build ALL DEPENDS "${_zj_static}" "${_zj_shared}")

add_library(pyrlyn-jit STATIC IMPORTED GLOBAL)
set_target_properties(pyrlyn-jit PROPERTIES IMPORTED_LOCATION "${_zj_static}"
  INTERFACE_INCLUDE_DIRECTORIES "${_zj_prefix}/include")
add_dependencies(pyrlyn-jit pyrlyn_jit_build)
add_library(pyrlyn_jit_shared SHARED IMPORTED GLOBAL)
set_target_properties(pyrlyn_jit_shared PROPERTIES IMPORTED_LOCATION "${_zj_shared}"
  INTERFACE_INCLUDE_DIRECTORIES "${_zj_prefix}/include")
add_dependencies(pyrlyn_jit_shared pyrlyn_jit_build)
if(NOT WIN32)
  # The dylib/so does not exist at configure time, so CMake cannot derive a build RPATH for it.
  set_property(TARGET pyrlyn_jit_shared PROPERTY INTERFACE_LINK_OPTIONS "LINKER:-rpath,${_zj_prefix}/lib")
endif()
add_library(pyrlyn-jit::pyrlyn-jit ALIAS pyrlyn-jit)
add_library(pyrlyn-jit::pyrlyn-jit_shared ALIAS pyrlyn_jit_shared)

if(PYRLYN_JIT_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(FILES "${_zj_static}" "${_zj_shared}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY "${_zj_prefix}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(FILES "${_zj_prefix}/lib/pkgconfig/pyrlyn-jit.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
  set(_zj_cfgdir ${CMAKE_INSTALL_LIBDIR}/cmake/pyrlyn-jit)
  configure_package_config_file(cmake/pyrlyn-jitConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/pyrlyn-jitConfig.cmake"
    INSTALL_DESTINATION ${_zj_cfgdir} PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR)
  write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/pyrlyn-jitConfigVersion.cmake" COMPATIBILITY SameMinorVersion)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pyrlyn-jitConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/pyrlyn-jitConfigVersion.cmake"
    DESTINATION ${_zj_cfgdir})
endif()
