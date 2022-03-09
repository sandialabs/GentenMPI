# Usage of this module as follows:
#
#     find_package(TTB)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  TTB_DIR  Set this variable to the root installation of TTB
#
# Variables defined by this module:
#
#  TTB_LIBRARIES          The TTB libraries
#  TTB_INCLUDE_DIR        The location of TTB headers

find_path(TTB_INCLUDE
  NAMES TTB_Sptensor.h
  PATHS ${TTB_DIR}/include
)

find_library(TTB_LIBRARYONE
    NAMES ttb 
    PATHS ${TTB_DIR}/lib
)

find_library(TTB_LIBRARYTWO
    NAMES ttb_mathlibs
    PATHS ${TTB_DIR}/lib
)

set(TTB_INCLUDE_DIRS ${TTB_INCLUDE})
set(TTB_LIBRARIES ${TTB_LIBRARYONE} ${TTB_LIBRARYTWO})

MESSAGE("   TTB_INCLUDE_DIRS: ${TTB_INCLUDE_DIRS}")
MESSAGE("   TTB_LIBRARIES:    ${TTB_LIBRARIES}")
