cmake_minimum_required(VERSION 3.15)
project(Template)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS OFF)
option(BUILD_STATIC_LIBS ON)
option(USE_NASM "Activate the use of NASM in the project" OFF)

if(USE_NASM)
  option(NASM_FREESTANDING "Whether to not link NASM executables with the C compiler and C library" TRUE)

  find_program(ASM_COMPILER NAMES "nasm" REQUIRED)
  set(CMAKE_ASM_NASM_COMPILER ${ASM_COMPILER} CACHE PATH "NASM Assembler Path" FORCE)
  enable_language(ASM_NASM)

  string(REPLACE "<CMAKE_ASM_NASM_COMPILER>" "<CMAKE_C_COMPILER>" CMAKE_ASM_NASM_LINK_EXECUTABLE ${CMAKE_ASM_NASM_LINK_EXECUTABLE})

  if(NASM_FREESTANDING)
    set(CMAKE_ASM_NASM_LINK_FLAGS "-nostdlib")
  endif()
endif()

include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR} ${CMAKE_INSTALL_FULL_LIBDIR}")
message(STATUS "Using default install RPATH of: ${CMAKE_INSTALL_RPATH}")

file(GLOB SUBMODULES "external/*")

foreach(SUBMODULE ${SUBMODULES})
  if(IS_DIRECTORY ${SUBMODULE})
    if(EXISTS "${SUBMODULE}/CMakeLists.txt")
      message(STATUS "Adding submodule: ${SUBMODULE}")
      add_subdirectory(${SUBMODULE})
    endif()
  endif()
endforeach()

add_subdirectory(src)
