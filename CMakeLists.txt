#---- Specify the CMake version required.
CMAKE_MINIMUM_REQUIRED(VERSION 2.7)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

OPTION(LayoutRight "Option to use LayoutRight in FactorMatrices" OFF)
IF (LayoutRight)
  ADD_DEFINITIONS(-DPT_LAYOUTRIGHT)
  MESSAGE("-- LayoutRight is ENABLED; compiling with -DPT_LAYOUTRIGHT")
  SET(PT_LAYOUTRIGHT TRUE)
ENDIF()

# CMAKE File for building against an installed Trilinos

INCLUDE(CMakeDependentOption)

IF (NOT DEFINED SET_COMPILERS_AUTOMATICALLY)
  SET(SET_COMPILERS_AUTOMATICALLY TRUE)
ENDIF()

IF (NOT SET_COMPILERS_AUTOMATICALLY)
  MESSAGE(FATAL_ERROR "Why are you trying to avoid the Trilinos compilers?")
ENDIF()

# Advertise that we support shared libs
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Get Trilinos as one entity
IF (DEFINED Trilinos_DIR)
  SET(CMAKE_PREFIX_PATH ${Trilinos_DIR} ${CMAKE_PREFIX_PATH})
ENDIF()
FIND_PACKAGE(Trilinos REQUIRED)

# Trilinos_BIN_DIRS probably should be defined in the Trilinos config. 
# Until it is, set it here.
# This is needed to find SEACAS tools used during testing (epu, etc).
#SET(Trilinos_BIN_DIRS "${Trilinos_DIR}/bin")
#SET(Trilinos_LIB_DIRS "${Trilinos_DIR}/lib")

MESSAGE("\nFound Trilinos: ")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
#MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
#MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
#MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
#MESSAGE("   Trilinos_BIN_DIRS = ${Trilinos_BIN_DIRS}")
#MESSAGE("   Trilinos_LIB_DIRS = ${Trilinos_LIB_DIRS}")
#MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
#MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
#MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
#MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
#MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("   Trilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE("End of Trilinos details\n")

# Compress Trilinos lists that are used
#${Trilinos_INCLUDE_DIRS}
#${Trilinos_TPL_INCLUDE_DIRS}
#${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} ${Trilinos_EXTRA_LD_FLAGS}

IF (Trilinos_INCLUDE_DIRS) 
  LIST(REVERSE Trilinos_INCLUDE_DIRS)
  LIST(REMOVE_DUPLICATES Trilinos_INCLUDE_DIRS)
  LIST(REVERSE Trilinos_INCLUDE_DIRS)
ENDIF()

IF (Trilinos_TPL_INCLUDE_DIRS) 
  LIST(REVERSE Trilinos_TPL_INCLUDE_DIRS)
  LIST(REMOVE_DUPLICATES Trilinos_TPL_INCLUDE_DIRS)
  LIST(REVERSE Trilinos_TPL_INCLUDE_DIRS)
ENDIF()

IF (Trilinos_LIBRARIES) 
  LIST(REVERSE Trilinos_LIBRARIES)
  LIST(REMOVE_DUPLICATES Trilinos_LIBRARIES)
  LIST(REVERSE Trilinos_LIBRARIES)
ENDIF()

IF (Trilinos_TPL_LIBRARIES) 
  LIST(REVERSE Trilinos_TPL_LIBRARIES)
  LIST(REMOVE_DUPLICATES Trilinos_TPL_LIBRARIES)
  LIST(REVERSE Trilinos_TPL_LIBRARIES)
ENDIF()

# Optional Installation helpers
# Note that some of this functionality depends on CMAKE > 2.8.8
#SET (INSTALL_ALBANY FALSE)
#IF (ENABLE_INSTALL)
#  IF (${CMAKE_VERSION} VERSION_GREATER 2.8.8 OR ${CMAKE_VERSION} VERSION_EQUAL 2.8.8)
#    SET (INSTALL_ALBANY TRUE)
#    MESSAGE("")
#    MESSAGE("Installation files for Albany will be created")
#    MESSAGE("-- install location = ${CMAKE_INSTALL_PREFIX}")
#    MESSAGE("")
#  ELSE()
#    MESSAGE("")
#    MESSAGE("Installation has been requested, but cannot continue")
#    MESSAGE("-- Please upgrade CMAKE to version >= 2.8.8")
#    MESSAGE("")
#  ENDIF()
#ELSE()
#    MESSAGE("-- Not installing Albany libraries. ENABLE_INSTALL was not defined.")
#    MESSAGE("")
#ENDIF()

#IF (INSTALL_ALBANY)
#  set(BINARY_INSTALL_DIR bin)
#  set(INCLUDE_INSTALL_DIR include)
#  set(LIB_INSTALL_DIR lib)
#  include(CMakePackageConfigHelpers)
#ENDIF ()

option (ENABLE_CROSS_COMPILE "Flag to turn on cross-compiling tools" OFF)
if (ENABLE_CROSS_COMPILE)
  set (KDD_CROSS_COMPILE ON)
  message (FATAL_ERROR "Not ready for cross-compilation; see CMakeLists.txt.")
else ()
  set (KDD_CROSS_COMPILE OFF)
endif ()

if (SET_COMPILERS_AUTOMATICALLY)
  MESSAGE("Setting and checking of compilers:")
  if (KDD_CROSS_COMPILE)
    # Force the compilers to be the same as Trilinos (GAH experimental)
    # This syntax is needed when cross compiling or the compilers get checked
    # again by the Albany configure, and will probably fail, because the options
    # won't be right and the exes won't run on the host.
    INCLUDE (CMakeForceCompiler)
    SET(CMAKE_SYSTEM_NAME Generic)
    CMAKE_FORCE_CXX_COMPILER(${Trilinos_CXX_COMPILER} Generic)
    CMAKE_FORCE_C_COMPILER(${Trilinos_C_COMPILER} Generic)
    CMAKE_FORCE_Fortran_COMPILER(${Trilinos_Fortran_COMPILER} Generic)
#    SET(CMAKE_SYSTEM_NAME ${Trilinos_SYSTEM_NAME})
#    CMAKE_FORCE_CXX_COMPILER(${Trilinos_CXX_COMPILER} ${Trilinos_CXX_COMPILER_ID})
#    CMAKE_FORCE_C_COMPILER(${Trilinos_C_COMPILER} ${Trilinos_C_COMPILER_ID})
#    CMAKE_FORCE_Fortran_COMPILER(${Trilinos_Fortran_COMPILER} ${Trilinos_Fortran_COMPILER_ID})
#    SET(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES ${Trilinos_Fortran_IMPLICIT_LINK_LIBRARIES})
  else ()
    set (CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
    set (CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
    set (CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER})
  endif ()
else()
  # Make sure the compilers match.
  MESSAGE("Checking compilers:")
  IF(NOT ${Trilinos_CXX_COMPILER} STREQUAL ${CMAKE_CXX_COMPILER})
    MESSAGE(FATAL_ERROR "C++ compilers don't match (Trilinos: ${Trilinos_CXX_COMPILER}, ${PROJECT_NAME}: ${CMAKE_CXX_COMPILER}).")
  ENDIF()
  IF(NOT ${Trilinos_C_COMPILER} STREQUAL ${CMAKE_C_COMPILER})
    MESSAGE(FATAL_ERROR "C compilers don't match (Trilinos: ${Trilinos_C_COMPILER}, ${PROJECT_NAME}: ${CMAKE_C_COMPILER}).")
  ENDIF()
  SET(Trilinos_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID})
  SET(Trilinos_C_COMPILER_ID ${CMAKE_C_COMPILER_ID})
endif()

# Build Albany as shared libraries if Trilinos was compiled that way

IF(Trilinos_BUILD_SHARED_LIBS)
  SET(KDD_LIBRARY_TYPE SHARED)
  SET(BUILD_SHARED_LIBS ON)
  MESSAGE("-- Building with shared libraries")
ELSE(Trilinos_BUILD_SHARED_LIBS)
  SET(KDD_LIBRARY_TYPE STATIC)
ENDIF(Trilinos_BUILD_SHARED_LIBS)

OPTION(KDD_BUILD_STATIC_EXE "Flag to turn on building of static executables" OFF)
IF((NOT BUILD_SHARED_LIBS) AND (KDD_BUILD_STATIC_EXE))
  MESSAGE("-- Building with static executables")
ENDIF((NOT BUILD_SHARED_LIBS) AND (KDD_BUILD_STATIC_EXE))

# End of setup and error checking
#  NOTE: PROJECT command checks for compilers, so this statement
#        is moved AFTER setting CMAKE_CXX_COMPILER from Trilinos

if (SET_COMPILERS_AUTOMATICALLY)
  PROJECT(KDD)
endif()

# AGS: Adding these lines so Ctest can be run to submit to cdash dashboard
#   Uses CTestConfig.cmake file for Cdash info.
ENABLE_TESTING()
INCLUDE(CTest)

# Make Albany inherit Trilinos compiler flags. The most important are the
# presence or absence of -NDEBUG (which can affect linking for complicated
# reasons) and -std=c++11.
SET(CMAKE_CXX_FLAGS "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
# GAH - temporarily set Albany flags for clang debugging
#SET(CMAKE_CXX_FLAGS "-ggdb -std=c++11 -O2 -march=native -DNDEBUG")

if (SET_COMPILERS_AUTOMATICALLY)
  SET(CMAKE_AR ${Trilinos_AR})
  SET(CMAKE_LINKER ${Trilinos_LINKER})
endif()

MESSAGE("-- Using ${Trilinos_CXX_COMPILER_ID} Compiler.")

# No longer needed but keep for now as this shows how to silence specific warnings on Intel compiler
#IF(${Trilinos_CXX_COMPILER_ID} MATCHES "Intel")
#  SET(USING_INTEL TRUE)
# On Intel, turn of interprocedural optimization at linking to keep linker from hanging
# This could possibly be removed with further study (GAH)
#  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-ip")
#   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd2536,2196,858,488,161,47")
#ENDIF()

## Check for some required Trilinos package dependencies (Piro, Phalanx) ##
MESSAGE("\nChecking Trilinos build for required and optional packages:")

SET(KDD_REQUIRED_TRILINOS_PACKAGES
  Tpetra
  Teuchos
  )

foreach(package IN LISTS KDD_REQUIRED_TRILINOS_PACKAGES)
  LIST(FIND Trilinos_PACKAGE_LIST "${package}" ${package}_List_ID)
  IF (${package}_List_ID GREATER -1)
    MESSAGE("-- Found ${package},    a required Trilinos package.")
  ELSE()
    MESSAGE("-- FATAL ERROR: Did not find ${package} package in Trilinos Installation.")
    MESSAGE(FATAL_ERROR "-- Configure Trilinos with guidance from trilinos-cmake script in doc directory.")
  ENDIF()
endforeach()

### Check for some OPTIONAL trilinos packages, which set DEFINES in build ###

# MPI check -- defaults to TRUE
# MESSAGE("-- Checking if MPI is enabled in Trilinos:")

LIST(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
IF (MPI_List_ID GREATER -1)
  MESSAGE("-- Looking for MPI in Trilinos:    Found")
  SET(KDD_MPI TRUE)
  ADD_DEFINITIONS(-DKDD_MPI -DSPLATT_USE_MPI)
ELSE()
  MESSAGE("-- Looking for MPI in Trilinos:    NOT found, MPI NOT ENABLED!")
  SET(KDD_MPI FALSE)
ENDIF()

# set optional dependency on Zoltan, for partitioning
# MESSAGE("-- Looking for Zoltan:")
LIST(FIND Trilinos_PACKAGE_LIST Zoltan Zoltan_List_ID)
IF (Zoltan_List_ID GREATER -1)
  ADD_DEFINITIONS(-DKDD_ZOLTAN)
  SET(KDD_ZOLTAN TRUE)
  MESSAGE("-- Looking for Zoltan:             Found")
ELSE()
  MESSAGE("-- Looking for Zoltan:             NOT found.")
  SET(KDD_ZOLTAN FALSE)
ENDIF()

# TTB
IF (DEFINED TTB_DIR)
  MESSAGE("-- Looking for TTB:")
  FIND_PACKAGE(TTB)
ENDIF()

# SPLATT
IF (DEFINED SPLATT_DIR)
  MESSAGE("-- Looking for SPLATT:")
  FIND_PACKAGE(SPLATT)
ENDIF()

## Other Miscellaneous Albany Options here:
MESSAGE("\nMiscellaneous Options:")

# optionally set global ordinal size to signed 64 bits
OPTION(ENABLE_64BIT_INT "Flag to set global ordinal size to signed 64 bit int" OFF)
IF (ENABLE_64BIT_INT)
  ADD_DEFINITIONS(-DKDD_64BIT_INT)
  MESSAGE("-- 64 bit GO is Enabled, compiling with -DKDD_64BIT_INT")
  SET(KDD_64BIT_INT TRUE)
ELSE()
  MESSAGE("-- 64 bit GO is NOT Enabled.")
  SET(KDD_64BIT_INT FALSE)
ENDIF()

# It is not currently possible to determine the status of "Kokkos_ENABLE_Cuda" from Albany. So we will deduce it
# from ${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h
IF(NOT DEFINED Kokkos_ENABLE_Cuda)
  SET(Kokkos_ENABLE_Cuda OFF)
  IF(EXISTS "${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h")
    FILE(READ ${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h CURRENT_CONFIG)
    STRING(REGEX MATCH "\#define KOKKOS_HAVE_CUDA" KOKKOS_CUDA_IS_SET ${CURRENT_CONFIG})
    IF("#define KOKKOS_HAVE_CUDA" STREQUAL "${KOKKOS_CUDA_IS_SET}")
      MESSAGE("-- Kokkos is configured to use CUDA.")
      SET(Kokkos_ENABLE_Cuda ON)
      SET(NUM_GPUS_PER_NODE 4 CACHE INT "set Number of GPUs per Node")
      MESSAGE("-- NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}")
      SET(KOKKOS_NDEVICES "--kokkos-ndevices=${NUM_GPUS_PER_NODE}")
      IF (ENABLE_CUDA_ERROR_CHECK)
        ADD_DEFINITIONS(-DALBANY_CUDA_ERROR_CHECK)
        MESSAGE("-- CUDA Error Check        is Enabled, compiling with -DALBANY_CUDA_ERROR_CHECK")
      ELSE()
        MESSAGE("-- CUDA Error Check        is NOT Enabled.")
      ENDIF()
      IF (ENABLE_CUDA_NVTX) 
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lnvToolsExt")
        ADD_DEFINITIONS(-DALBANY_CUDA_NVTX) 
        MESSAGE("-- CUDA NVTX               is Enabled, compiling with -DALBANY_CUDA_NVTX")
      ELSE()
        MESSAGE("-- CUDA NVTX               is NOT Enabled.")
      ENDIF()
    ELSE("#define KOKKOS_HAVE_CUDA" STREQUAL "${KOKKOS_CUDA_IS_SET}")
      MESSAGE("-- Kokkos is not configured with CUDA; this code will NOT compilewith CUDA.")
    ENDIF("#define KOKKOS_HAVE_CUDA" STREQUAL "${KOKKOS_CUDA_IS_SET}")
  ENDIF(EXISTS "${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h")
ENDIF(NOT DEFINED Kokkos_ENABLE_Cuda)


# Add option to check for hardware floating point exceptions using SSE, defaults to disabled
#OPTION(ENABLE_CHECK_FPE "Flag to turn on floating point exception (NAN) checking" OFF)
#IF (ENABLE_CHECK_FPE)
#  ADD_DEFINITIONS(-DALBANY_CHECK_FPE)
#  MESSAGE("-- FPE checking              is Enabled, compiling with -DALBANY_CHECK_FPE")
#  SET(ALBANY_CHECK_FPE TRUE)
#ELSE()
#  MESSAGE("-- FPE checking              is NOT Enabled.")
#  SET(ALBANY_CHECK_FPE FALSE)
#ENDIF()
#
#OPTION(ENABLE_STRONG_FPE_CHECK "Flag to turn on strong checking for floating point exceptions" OFF)
#IF (ENABLE_STRONG_FPE_CHECK)
#  ADD_DEFINITIONS(-DALBANY_STRONG_FPE_CHECK)
#  MESSAGE("-- Strong FPE checking       is Enabled, compiling with -DALBANY_STRONG_FPE_CHECK")
#  SET(ALBANY_STRONG_FPE_CHECK TRUE)
#ELSE()
#  MESSAGE("-- Strong FPE checking       is NOT Enabled.")
#  SET(ALBANY_STRONG_FPE_CHECK FALSE)
#ENDIF()
#
# add a target to generate API documentation with Doxygen
#IF( NOT DISABLE_DOXYGEN )
#  find_package(Doxygen)
#  if(DOXYGEN_FOUND)
#  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
#  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/albanyMainPage.doc ${CMAKE_CURRENT_BINARY_DIR}/albanyMainPage.doc)
#  add_custom_target(doc
#  ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#  COMMENT "Generating API documentation with Doxygen" VERBATIM
#  )
#  endif(DOXYGEN_FOUND)
#ENDIF()

## End of CMake options to Albany

#  Timeout real long running tests

IF(KDD_CTEST_TIMEOUT)
    MESSAGE("-- Setting maximum test run time to be ${KDD_CTEST_TIMEOUT} seconds")
ELSE()
    MESSAGE("-- No limits are set on maximum test run time")
ENDIF()

# Make a clean build target for convenience

ADD_CUSTOM_TARGET( cmake_clean
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/Makefile
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake
)

IF(KDD_MPI)
  IF(KDD_MPI_EXEC)
    SET(MPIEX  ${KDD_MPI_EXEC})
  ELSE()
    SET(MPIEX  ${Trilinos_MPI_EXEC})
  ENDIF()
  IF(KDD_MPI_EXEC_NUMPROCS_FLAG)
    SET(MPINPF  ${KDD_MPI_EXEC_NUMPROCS_FLAG})
  ELSE()
    SET(MPINPF ${Trilinos_MPI_EXEC_NUMPROCS_FLAG})
  ENDIF()
  IF(KDD_MPI_EXEC_MAX_NUMPROCS)
    SET(MPIMNP ${KDD_MPI_EXEC_MAX_NUMPROCS})
  ELSE()
    SET(MPIMNP ${Trilinos_MPI_EXEC_MAX_NUMPROCS})
  ENDIF()
  IF(KDD_MPI_EXEC_LEADING_OPTIONS)
    SET(MPIPRE ${KDD_MPI_EXEC_LEADING_OPTIONS})
    SEPARATE_ARGUMENTS(MPIPRE)
  ENDIF()
  IF(KDD_MPI_EXEC_TRAILING_OPTIONS)
    SET(MPIPOST ${KDD_MPI_EXEC_TRAILING_OPTIONS})
    SEPARATE_ARGUMENTS(MPIPOST)
  ENDIF()
  SET(SERIAL_CALL ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST})
  SET(PARALLEL_CALL ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST})
  STRING(REPLACE ";" " " SERIAL_CALL_TEXT "${SERIAL_CALL}")
  STRING(REPLACE ";" " " PARALLEL_CALL_TEXT "${PARALLEL_CALL}")
  MESSAGE("-- CTest serial call: ${SERIAL_CALL_TEXT}")
  MESSAGE("-- CTest parallel call: ${PARALLEL_CALL_TEXT}")
ENDIF(KDD_MPI)

INCLUDE_DIRECTORIES(
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${TTB_INCLUDE_DIRS}
  ${SPLATT_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}
)

SET(ALL_LIBRARIES ${Trilinos_LIBRARIES} ${TTB_LIBRARIES} ${SPLATT_LIBRARIES})

# Add src and test
add_subdirectory(src)
IF (DEFINED TTB_DIR)
  # Unit tests compare with TTB
  add_subdirectory(test)
ENDIF()
IF (DEFINED SPLATT_DIR)
  # Performance tests use SPLATT reader
  add_subdirectory(perf_test)
ENDIF()

