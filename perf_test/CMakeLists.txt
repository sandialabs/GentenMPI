ADD_EXECUTABLE(pt_cpals.exe pt_cpals.cpp)
TARGET_LINK_LIBRARIES(pt_cpals.exe ${ALL_LIBRARIES})

ADD_EXECUTABLE(pt_adam.exe pt_adam.cpp)
TARGET_LINK_LIBRARIES(pt_adam.exe ${ALL_LIBRARIES})

ADD_TEST(cpals_01_random pt_cpals.exe --dist=1x1x1 --modes=10x20x30 --random=100 --maxiter=10 --miniter=10 )
SET_TESTS_PROPERTIES(cpals_01_random
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(cpals_01_splatt mpirun -np 1 pt_cpals.exe --file=simple.tns --maxiter=10 --miniter=10 )
SET_TESTS_PROPERTIES(cpals_01_splatt
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(cpals_04_random mpirun -np 4 pt_cpals.exe --dist=2x1x1x2 --modes=10x20x30x40 --random=1000 --maxiter=10 --miniter=10 )
SET_TESTS_PROPERTIES(cpals_04_random
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(cpals_04_splatt mpirun -np 4 pt_cpals.exe --file=simple.tns --maxiter=10 --miniter=10 )
SET_TESTS_PROPERTIES(cpals_04_splatt
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(cpals_09_random mpirun -np 9 pt_cpals.exe --dist=1x3x3 --modes=10x27x54 --random=100 --maxiter=10 --miniter=10 )
SET_TESTS_PROPERTIES(cpals_09_random
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(cpals_09_splatt mpirun -np 9 pt_cpals.exe --file=simple.tns --maxiter=10 --miniter=10 )
SET_TESTS_PROPERTIES(cpals_09_splatt
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(adam_01_random pt_adam.exe --dist=1x1x1 --modes=10x20x30 --random=100 --maxiter=10 --miniter=10 )
SET_TESTS_PROPERTIES(adam_01_random
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(adam_01_splatt mpirun -np 1 pt_adam.exe --file=simple.tns --maxiter=10 --miniter=10 )
SET_TESTS_PROPERTIES(adam_01_splatt
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(adam_04_random mpirun -np 4 pt_adam.exe --dist=2x1x1x2 --modes=10x20x30x40 --random=1000 --maxiter=10 --miniter=10 )
SET_TESTS_PROPERTIES(adam_04_random
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(adam_04_splatt mpirun -np 4 pt_adam.exe --file=simple.tns --maxiter=10 --miniter=10 )
SET_TESTS_PROPERTIES(adam_04_splatt
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(adam_09_random mpirun -np 9 pt_adam.exe --dist=1x3x3 --modes=10x27x54 --random=100 --maxiter=10 --miniter=10 )
SET_TESTS_PROPERTIES(adam_09_random
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(adam_09_splatt mpirun -np 9 pt_adam.exe --file=simple.tns --maxiter=10 --miniter=10 )
SET_TESTS_PROPERTIES(adam_09_splatt
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_EXECUTABLE(copy.exe copy.cpp)
TARGET_LINK_LIBRARIES(copy.exe ${ALL_LIBRARIES})

ADD_EXECUTABLE(copymv.exe copymv.cpp)
TARGET_LINK_LIBRARIES(copymv.exe ${ALL_LIBRARIES})

configure_file(
   ${CMAKE_SOURCE_DIR}/data/simple.tns
   ${CMAKE_CURRENT_BINARY_DIR}/simple.tns 
   COPYONLY
)
