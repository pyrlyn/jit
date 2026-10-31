# Thin CMake adapter: build.zig stays the primary build. This file only drives `zig build`
# and exposes the result as IMPORTED targets zigjit::zigjit (static) and zigjit::zigjit_shared.
# Works with add_subdirectory, FetchContent, CPM and (after install) find_package(zigjit).
cmake_minimum_required(VERSION 3.20)
project(zigjit VERSION 0.1.0 LANGUAGES C)

find_program(ZIGJIT_ZIG_EXECUTABLE zig REQUIRED DOC "Zig compiler (0.16) used to build zigjit")
set(ZIGJIT_OPTIMIZE "" CACHE STRING "zig -Doptimize (Debug, ReleaseSafe, ReleaseFast, ReleaseSmall); empty = from CMAKE_BUILD_TYPE")
set(ZIGJIT_TARGETS "x86_64,aarch64" CACHE STRING "Backends to compile in (-Dtargets)")
option(ZIGJIT_INSTALL "Install zigjit with the CMake install step" ${PROJECT_IS_TOP_LEVEL})

if(ZIGJIT_OPTIMIZE)
  set(_zj_opt "${ZIGJIT_OPTIMIZE}")
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
set(_zj_static "${_zj_prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}zigjit${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(_zj_shared "${_zj_prefix}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}zigjit${CMAKE_SHARED_LIBRARY_SUFFIX}")
file(MAKE_DIRECTORY "${_zj_prefix}/include")
file(GLOB_RECURSE _zj_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.zig" "${CMAKE_CURRENT_SOURCE_DIR}/include/*")

add_custom_command(
  OUTPUT "${_zj_static}" "${_zj_shared}"
  COMMAND "${ZIGJIT_ZIG_EXECUTABLE}" build install -Doptimize=${_zj_opt} -Dtargets=${ZIGJIT_TARGETS}
          -Dversion=${PROJECT_VERSION} --prefix "${_zj_prefix}"
          --cache-dir "${CMAKE_CURRENT_BINARY_DIR}/zig-cache"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/build.zig" ${_zj_sources}
  COMMENT "zig build (zigjit, -Doptimize=${_zj_opt})"
  VERBATIM)
add_custom_target(zigjit_build ALL DEPENDS "${_zj_static}" "${_zj_shared}")

add_library(zigjit STATIC IMPORTED GLOBAL)
set_target_properties(zigjit PROPERTIES IMPORTED_LOCATION "${_zj_static}"
  INTERFACE_INCLUDE_DIRECTORIES "${_zj_prefix}/include")
add_dependencies(zigjit zigjit_build)
add_library(zigjit_shared SHARED IMPORTED GLOBAL)
set_target_properties(zigjit_shared PROPERTIES IMPORTED_LOCATION "${_zj_shared}"
  INTERFACE_INCLUDE_DIRECTORIES "${_zj_prefix}/include")
add_dependencies(zigjit_shared zigjit_build)
if(NOT WIN32)
  # The dylib/so does not exist at configure time, so CMake cannot derive a build RPATH for it.
  set_property(TARGET zigjit_shared PROPERTY INTERFACE_LINK_OPTIONS "LINKER:-rpath,${_zj_prefix}/lib")
endif()
add_library(zigjit::zigjit ALIAS zigjit)
add_library(zigjit::zigjit_shared ALIAS zigjit_shared)

if(ZIGJIT_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(FILES "${_zj_static}" "${_zj_shared}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY "${_zj_prefix}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(FILES "${_zj_prefix}/lib/pkgconfig/zigjit.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
  set(_zj_cfgdir ${CMAKE_INSTALL_LIBDIR}/cmake/zigjit)
  configure_package_config_file(cmake/zigjitConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/zigjitConfig.cmake"
    INSTALL_DESTINATION ${_zj_cfgdir} PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR)
  write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/zigjitConfigVersion.cmake" COMPATIBILITY SameMinorVersion)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/zigjitConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/zigjitConfigVersion.cmake"
    DESTINATION ${_zj_cfgdir})
endif()
