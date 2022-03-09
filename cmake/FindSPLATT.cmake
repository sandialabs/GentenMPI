# Usage of this module as follows:
#
#     find_package(SPLATT)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  SPLATT_DIR  Set this variable to the root installation of SPLATT
#
# Variables defined by this module:
#
#  SPLATT_LIBRARIES          The SPLATT libraries
#  SPLATT_INCLUDE_DIR        The location of SPLATT headers

find_path(SPLATT_INCLUDEONE
  NAMES splatt_mpi.h
  PATHS ${SPLATT_DIR}/../../src
)

find_path(SPLATT_INCLUDETWO
  NAMES splatt.h
  PATHS ${SPLATT_DIR}/include
)


find_library(SPLATT_LIBRARY
    NAMES splatt
    PATHS ${SPLATT_DIR}/lib
)


set(SPLATT_INCLUDE_DIRS ${SPLATT_INCLUDEONE} ${SPLATT_INCLUDETWO})
set(SPLATT_LIBRARIES ${SPLATT_LIBRARY})

MESSAGE("   SPLATT_INCLUDE_DIRS: ${SPLATT_INCLUDE_DIRS}")
MESSAGE("   SPLATT_LIBRARIES:    ${SPLATT_LIBRARIES}")
