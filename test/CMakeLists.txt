############################################################
SET (MPI4 mpirun -np 4)
SET (MPI7 mpirun -np 7)
SET (MPI9 mpirun -np 9)

############################################################
ADD_EXECUTABLE(pt_test_compare.exe pt_test_compare.cpp)
TARGET_LINK_LIBRARIES(pt_test_compare.exe ${ALL_LIBRARIES})

ADD_TEST(test_compare pt_test_compare.exe)
SET_TESTS_PROPERTIES(test_compare
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_factormatrix.exe pt_test_factormatrix.cpp)
TARGET_LINK_LIBRARIES(pt_test_factormatrix.exe ${ALL_LIBRARIES})

ADD_TEST(test_factormatrix pt_test_factormatrix.exe)
SET_TESTS_PROPERTIES(test_factormatrix
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_factormatrix_4 ${MPI4} pt_test_factormatrix.exe)
SET_TESTS_PROPERTIES(test_factormatrix_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_factormatrix_7 ${MPI7} pt_test_factormatrix.exe)
SET_TESTS_PROPERTIES(test_factormatrix_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)
ADD_TEST(test_factormatrix_9 ${MPI9} pt_test_factormatrix.exe)
SET_TESTS_PROPERTIES(test_factormatrix_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_ktensor.exe pt_test_ktensor.cpp)
TARGET_LINK_LIBRARIES(pt_test_ktensor.exe ${ALL_LIBRARIES})

ADD_TEST(test_ktensor pt_test_ktensor.exe)
SET_TESTS_PROPERTIES(test_ktensor
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_ktensor_4 ${MPI4} pt_test_ktensor.exe)
SET_TESTS_PROPERTIES(test_ktensor_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_ktensor_7 ${MPI7} pt_test_ktensor.exe)
SET_TESTS_PROPERTIES(test_ktensor_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)
ADD_TEST(test_ktensor_9 ${MPI9} pt_test_ktensor.exe)
SET_TESTS_PROPERTIES(test_ktensor_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_sptensor.exe pt_test_sptensor.cpp)
TARGET_LINK_LIBRARIES(pt_test_sptensor.exe ${ALL_LIBRARIES})

ADD_TEST(test_sptensor pt_test_sptensor.exe)
SET_TESTS_PROPERTIES(test_sptensor
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_sptensor_4 ${MPI4} pt_test_sptensor.exe)
SET_TESTS_PROPERTIES(test_sptensor_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_sptensor_7 ${MPI7} pt_test_sptensor.exe)
SET_TESTS_PROPERTIES(test_sptensor_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)
ADD_TEST(test_sptensor_9 ${MPI9} pt_test_sptensor.exe)
SET_TESTS_PROPERTIES(test_sptensor_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_sptensor_boundingbox.exe pt_test_sptensor_boundingbox.cpp)
TARGET_LINK_LIBRARIES(pt_test_sptensor_boundingbox.exe ${ALL_LIBRARIES})

ADD_TEST(test_sptensor_boundingbox pt_test_sptensor_boundingbox.exe)
SET_TESTS_PROPERTIES(test_sptensor_boundingbox
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_sptensor_boundingbox_4 ${MPI4} pt_test_sptensor_boundingbox.exe)
SET_TESTS_PROPERTIES(test_sptensor_boundingbox_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_sptensor_boundingbox_7 ${MPI7} pt_test_sptensor_boundingbox.exe)
SET_TESTS_PROPERTIES(test_sptensor_boundingbox_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)
ADD_TEST(test_sptensor_boundingbox_9 ${MPI9} pt_test_sptensor_boundingbox.exe)
SET_TESTS_PROPERTIES(test_sptensor_boundingbox_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_sampledSptensor.exe pt_test_sampledSptensor.cpp)
TARGET_LINK_LIBRARIES(pt_test_sampledSptensor.exe ${ALL_LIBRARIES})

ADD_TEST(test_sampledSptensor pt_test_sampledSptensor.exe)
SET_TESTS_PROPERTIES(test_sampledSptensor
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_sampledSptensor_4 ${MPI4} pt_test_sampledSptensor.exe)
SET_TESTS_PROPERTIES(test_sampledSptensor_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_sampledSptensor_7 ${MPI7} pt_test_sampledSptensor.exe)
SET_TESTS_PROPERTIES(test_sampledSptensor_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)
ADD_TEST(test_sampledSptensor_9 ${MPI9} pt_test_sampledSptensor.exe)
SET_TESTS_PROPERTIES(test_sampledSptensor_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_multivec.exe pt_multivec.cpp)
TARGET_LINK_LIBRARIES(pt_multivec.exe ${ALL_LIBRARIES})

#Parallel tests not needed.
#ADD_TEST(multivec pt_multivec.exe)
#SET_TESTS_PROPERTIES(multivec
#   PROPERTIES
#   PASS_REGULAR_EXPRESSION "PASS"
#   FAIL_REGULAR_EXPRESSION "FAIL"
#)

############################################################
ADD_EXECUTABLE(pt_test_random.exe pt_test_random.cpp)
TARGET_LINK_LIBRARIES(pt_test_random.exe ${ALL_LIBRARIES})

ADD_TEST(test_random_4 ${MPI4} pt_test_random.exe)
SET_TESTS_PROPERTIES(test_random_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_random_7 ${MPI7} pt_test_random.exe)
SET_TESTS_PROPERTIES(test_random_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_random_9 ${MPI9} pt_test_random.exe)
SET_TESTS_PROPERTIES(test_random_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_system.exe pt_test_system.cpp)
TARGET_LINK_LIBRARIES(pt_test_system.exe ${ALL_LIBRARIES})
ADD_TEST(test_system pt_test_system.exe)
SET_TESTS_PROPERTIES(test_system
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_system_4 ${MPI4} pt_test_system.exe)
SET_TESTS_PROPERTIES(test_system_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_system_7 ${MPI7} pt_test_system.exe)
SET_TESTS_PROPERTIES(test_system_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_system_9 ${MPI9} pt_test_system.exe)
SET_TESTS_PROPERTIES(test_system_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_modeleval.exe pt_test_modeleval.cpp)
TARGET_LINK_LIBRARIES(pt_test_modeleval.exe ${ALL_LIBRARIES})
ADD_TEST(test_modeleval pt_test_modeleval.exe)
SET_TESTS_PROPERTIES(test_modeleval
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_modeleval_4 ${MPI4} pt_test_modeleval.exe)
SET_TESTS_PROPERTIES(test_modeleval_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_modeleval_7 ${MPI7} pt_test_modeleval.exe)
SET_TESTS_PROPERTIES(test_modeleval_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_modeleval_9 ${MPI9} pt_test_modeleval.exe)
SET_TESTS_PROPERTIES(test_modeleval_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_mixed.exe pt_test_mixed.cpp)
TARGET_LINK_LIBRARIES(pt_test_mixed.exe ${ALL_LIBRARIES})

ADD_TEST(test_mixed pt_test_mixed.exe)
SET_TESTS_PROPERTIES(test_mixed
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_mixed_4 ${MPI4} pt_test_mixed.exe)
SET_TESTS_PROPERTIES(test_mixed_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_mixed_7 ${MPI7} pt_test_mixed.exe)
SET_TESTS_PROPERTIES(test_mixed_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)
ADD_TEST(test_mixed_9 ${MPI9} pt_test_mixed.exe)
SET_TESTS_PROPERTIES(test_mixed_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_gram.exe pt_test_gram.cpp)
TARGET_LINK_LIBRARIES(pt_test_gram.exe ${ALL_LIBRARIES})

ADD_TEST(test_gram pt_test_gram.exe)
SET_TESTS_PROPERTIES(test_gram
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_gram_4 ${MPI4} pt_test_gram.exe)
SET_TESTS_PROPERTIES(test_gram_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_gram_7 ${MPI7} pt_test_gram.exe)
SET_TESTS_PROPERTIES(test_gram_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_gram_9 ${MPI9} pt_test_gram.exe)
SET_TESTS_PROPERTIES(test_gram_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_innerprod.exe pt_test_innerprod.cpp)
TARGET_LINK_LIBRARIES(pt_test_innerprod.exe ${ALL_LIBRARIES})

ADD_TEST(test_innerprod pt_test_innerprod.exe)
SET_TESTS_PROPERTIES(test_innerprod
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_innerprod_4 ${MPI4} pt_test_innerprod.exe)
SET_TESTS_PROPERTIES(test_innerprod_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_innerprod_7 ${MPI7} pt_test_innerprod.exe)
SET_TESTS_PROPERTIES(test_innerprod_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_innerprod_9 ${MPI9} pt_test_innerprod.exe)
SET_TESTS_PROPERTIES(test_innerprod_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_tensorhash.exe pt_test_tensorhash.cpp)
TARGET_LINK_LIBRARIES(pt_test_tensorhash.exe ${ALL_LIBRARIES})

ADD_TEST(test_tensorhash pt_test_tensorhash.exe)
SET_TESTS_PROPERTIES(test_tensorhash
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_tensorhash_4 ${MPI4} pt_test_tensorhash.exe)
SET_TESTS_PROPERTIES(test_tensorhash_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_tensorhash_7 ${MPI7} pt_test_tensorhash.exe)
SET_TESTS_PROPERTIES(test_tensorhash_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_tensorhash_9 ${MPI9} pt_test_tensorhash.exe)
SET_TESTS_PROPERTIES(test_tensorhash_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_sampledError.exe pt_test_sampledError.cpp)
TARGET_LINK_LIBRARIES(pt_test_sampledError.exe ${ALL_LIBRARIES})

ADD_TEST(test_sampledError pt_test_sampledError.exe)
SET_TESTS_PROPERTIES(test_sampledError
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_sampledError_4 ${MPI4} pt_test_sampledError.exe)
SET_TESTS_PROPERTIES(test_sampledError_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_sampledError_7 ${MPI7} pt_test_sampledError.exe)
SET_TESTS_PROPERTIES(test_sampledError_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_sampledError_9 ${MPI9} pt_test_sampledError.exe)
SET_TESTS_PROPERTIES(test_sampledError_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_mttkrp.exe pt_test_mttkrp.cpp)
TARGET_LINK_LIBRARIES(pt_test_mttkrp.exe ${ALL_LIBRARIES})

ADD_TEST(test_mttkrp pt_test_mttkrp.exe)
SET_TESTS_PROPERTIES(test_mttkrp
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_mttkrp_4 ${MPI4} pt_test_mttkrp.exe)
SET_TESTS_PROPERTIES(test_mttkrp_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_mttkrp_7 ${MPI7} pt_test_mttkrp.exe)
SET_TESTS_PROPERTIES(test_mttkrp_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_mttkrp_9 ${MPI9} pt_test_mttkrp.exe)
SET_TESTS_PROPERTIES(test_mttkrp_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_stocGrad.exe pt_test_stocGrad.cpp)
TARGET_LINK_LIBRARIES(pt_test_stocGrad.exe ${ALL_LIBRARIES})

ADD_TEST(test_stocGrad pt_test_stocGrad.exe)
SET_TESTS_PROPERTIES(test_stocGrad
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_stocGrad_7 ${MPI7} pt_test_stocGrad.exe)
SET_TESTS_PROPERTIES(test_stocGrad_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_adam.exe pt_test_adam.cpp)
TARGET_LINK_LIBRARIES(pt_test_adam.exe ${ALL_LIBRARIES})

ADD_TEST(test_adam pt_test_adam.exe)
SET_TESTS_PROPERTIES(test_adam
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_adam_7 ${MPI7} pt_test_adam.exe)
SET_TESTS_PROPERTIES(test_adam_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
ADD_EXECUTABLE(pt_test_cpals.exe pt_test_cpals.cpp)
TARGET_LINK_LIBRARIES(pt_test_cpals.exe ${ALL_LIBRARIES})

ADD_TEST(test_cpals pt_test_cpals.exe)
SET_TESTS_PROPERTIES(test_cpals
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_cpals_4 ${MPI4} pt_test_cpals.exe)
SET_TESTS_PROPERTIES(test_cpals_4
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_cpals_7 ${MPI7} pt_test_cpals.exe)
SET_TESTS_PROPERTIES(test_cpals_7
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

ADD_TEST(test_cpals_9 ${MPI9} pt_test_cpals.exe)
SET_TESTS_PROPERTIES(test_cpals_9
   PROPERTIES
   PASS_REGULAR_EXPRESSION "PASS"
   FAIL_REGULAR_EXPRESSION "FAIL"
)

############################################################
#ADD_EXECUTABLE_AND_TEST(
#   TestKokkosExperiments
#   SOURCES pt_kokkos_experiments.cpp
#   PASS_REGULAR_EXPRESSION "PASS"
#   FAIL_REGULAR_EXPRESSION "FAIL"
#)
###
