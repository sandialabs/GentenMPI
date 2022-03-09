// GentenMPI:  Distributed Memory Generalized Sparse Tensor Decomposition
//     Karen Devine -- kddevin@sandia.gov
//     Grey Ballard -- ballard@wfu.edu
//
// Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
// 
// For five (5) years from 3/9/2020 the United States Government is granted for
// itself and others acting on its behalf a paid-up, nonexclusive, irrevocable
// worldwide license in this data to reproduce, prepare derivative works, and
// perform publicly and display publicly, by or on behalf of the Government.
// There is provision for the possible extension of the term of this license.
// Subsequent to that period or any extension granted, the United States
// Government is granted for itself and others acting on its behalf a paid-up,
// nonexclusive, irrevocable worldwide license in this data to reproduce, 
// prepare derivative works, distribute copies to the public, perform publicly 
// and display publicly, and to permit others to do so. The specific term of the
// license can be identified by inquiry made to National Technology and
// Engineering Solutions of Sandia, LLC or DOE.
// 
// NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF
// ENERGY, NOR NATIONAL TECHNOLOGY AND ENGINEERING SOLUTIONS OF SANDIA, LLC, NOR
// ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES 
// ANY LEGAL RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY
// INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS
// USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
//  
// Any licensee of this software has the obligation and responsibility to abide
// by the applicable export control laws, regulations, and general prohibitions
// relating to the export of technical data. Failure to obtain an export control
// license or other authority from the Government may result in criminal
// liability under U.S. laws.
 
// Test squareLocalMatrix and gramianMatrix.
// Compare gramian matrix to TTB.


#include "pt_sptensor.hpp"
#include "pt_ktensor.hpp"
#include "pt_system.hpp"
#include "pt_random.hpp"
#include "Tpetra_Core.hpp"

static int verbosity = 0;

//////////////////////////////////////////////////////////////////////////////

int testRandom(
  pt::mode_t nModes,
  const std::vector<size_t> &modeSizes,
  const std::vector<int> &dist,
  const Teuchos::RCP<const Teuchos::Comm<int> > &comm
)
{
  int me = comm->getRank();
  int ierr = 0;
  pt::rank_t rank = 10;
  size_t randomsize = 100;
  
  if (me == 0) {
    std::cout << "testRandom with nModes: " << nModes << "\n modeSizes: ";
    for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
    std::cout << std::endl;        
  }

  typedef double scalar_t;
  typedef pt::distSptensor<scalar_t> sptensor_t;
  typedef pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef pt::distKtensor<factormatrix_t> ktensor_t;
  typedef pt::distSystem<sptensor_t, ktensor_t> distsys_t;

  pt::randomData<sptensor_t, ktensor_t> rd(comm);

  // Generate random ktensor with appropriate dimensions
  if (me == 0) std::cout << "Creating random ktensor..." << std::endl;

  ktensor_t *ktensor = rd.createRandomKtensor(rank, modeSizes);

  if (verbosity) ktensor->print("RandomKtensor");

  // Generate nonzero elements in sptensor relative to ktensor
  if (me == 0) std::cout << "Creating random sptensor..." << std::endl;

  sptensor_t *sptensor;
  try {
    sptensor = rd.createRandomSptensor(ktensor, randomsize, dist);
  }
  catch (std::exception &e) {
    delete ktensor;
    throw e;
  }

  sptensor->printStats("RandomSptensor");
  if (verbosity) sptensor->print("RandomSptensor");

  // See whether there are sufficient nonzeros generated
  if (me == 0) std::cout << "Checking nonzero count..." << std::endl;
  size_t nnz = sptensor->getLocalNumNonZeros();
  size_t gnnz;
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, 1, &nnz, &gnnz);

  if (gnnz < 0.85 * randomsize) {
    if (me == 0) 
      std::cout << "FAIL:  Too few nonzeros:  " << gnnz << " of " << randomsize
                << std::endl;        
    ierr++;
  }

  // Create a distributed system to print communication info
  distsys_t *distsys = new distsys_t(sptensor, ktensor);
  distsys->printStats("DistSystem");

  delete distsys;
  delete ktensor;
  delete sptensor;

  return ierr;
}

//////////////////////////////////////////////////////////////////////////////

int main(int narg, char *arg[])
{
  Tpetra::ScopeGuard scopeguard(&narg, &arg);

  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int me = comm->getRank();
  int np = comm->getSize();

  // Accept command-line parameters
  Teuchos::CommandLineProcessor cmdp (false, false);
  cmdp.setOption("verbosity", &verbosity, "0 = quiet; 1 = print ids ");
  cmdp.parse(narg, arg);

  // Run the tests; count how many fail.
  int ierr = 0;

  /////////////////////////////////
  if (me == 0) std::cout << "Test 1..." << std::endl;
  try
  {
    // Tests random generation with Kokkos::UnorderedMap and type char
    pt::mode_t nModes = 5;
    std::vector<size_t> modeSizes(nModes);
    std::vector<int> dist(nModes);
    modeSizes[0] = 10;
    modeSizes[1] = 20;
    modeSizes[2] = 30;
    modeSizes[3] = 40;
    modeSizes[4] = 50;
    dist[0] = np;
    dist[1] = 1;
    dist[2] = 1;
    dist[3] = 1;
    dist[4] = 1;

    ierr += testRandom(nModes, modeSizes, dist, comm);
  }
  catch (std::exception &e) {
    std::cout << me << " FAIL:  " << e.what() << std::endl;
    ierr++;
  }

  /////////////////////////////////
  if (me == 0) std::cout << "Test 2..." << std::endl;
  try
  { 
    // Test random generation with Kokkos::UnorderedMap and type char
    pt::mode_t nModes = 8;
    std::vector<size_t> modeSizes(nModes);
    std::vector<int> dist(nModes);
    modeSizes[0] = 10;
    modeSizes[1] = 20;
    modeSizes[2] = 30;
    modeSizes[3] = 40;
    modeSizes[4] = 50;
    modeSizes[5] = 60;
    modeSizes[6] = 70;
    modeSizes[7] = 80;
    dist[0] = 1;
    dist[1] = 1;
    dist[2] = 1;
    dist[3] = 1;
    dist[4] = 1;
    dist[6] = 1;
    dist[5] = 1;
    dist[7] = np;

    ierr += testRandom(nModes, modeSizes, dist, comm);
  }
  catch (std::exception &e) {
    std::cout << me << " FAIL:  " << e.what() << std::endl;
    ierr++;
  }

  /////////////////////////////////
  if (me == 0) std::cout << "Test 3..." << std::endl;
  try
  { 
    // Test random generation with std and type char
    pt::mode_t nModes = 9;
    std::vector<size_t> modeSizes(nModes);
    std::vector<int> dist(nModes);
    modeSizes[0] = 10;
    modeSizes[1] = 20;
    modeSizes[2] = 30;
    modeSizes[3] = 40;
    modeSizes[4] = 50;
    modeSizes[5] = 60;
    modeSizes[6] = 70;
    modeSizes[7] = 80;
    modeSizes[8] = 90;
    dist[0] = 1;
    dist[1] = 1;
    dist[2] = 1;
    dist[3] = 1;
    dist[4] = 1;
    dist[6] = 1;
    dist[5] = 1;
    dist[7] = np;
    dist[8] = 1;

    ierr += testRandom(nModes, modeSizes, dist, comm);
  }
  catch (std::exception &e) {
    std::cout << me << " FAIL:  " << e.what() << std::endl;
    ierr++;
  }

  /////////////////////////////////
  if (me == 0) std::cout << "Test 4..." << std::endl;
  try
  {
    // Tests random generation with Kokkos::UnorderedMap and type short
    pt::mode_t nModes = 3;
    std::vector<size_t> modeSizes(nModes);
    std::vector<int> dist(nModes);
    modeSizes[0] = 10;
    modeSizes[1] = 20;
    modeSizes[2] = 300;
    dist[0] = 1;
    dist[1] = np;
    dist[2] = 1;

    ierr += testRandom(nModes, modeSizes, dist, comm);
  }
  catch (std::exception &e) {
    std::cout << me << " FAIL:  " << e.what() << std::endl;
    ierr++;
  }

  /////////////////////////////////
  if (me == 0) std::cout << "Test 5..." << std::endl;
  try
  {
    // Tests random generation with Kokkos::UnorderedMap and type short
    pt::mode_t nModes = 4;
    std::vector<size_t> modeSizes(nModes);
    std::vector<int> dist(nModes);
    modeSizes[0] = 10;
    modeSizes[1] = 20;
    modeSizes[2] = 300;
    modeSizes[3] = 40;
    dist[0] = 1;
    dist[1] = np;
    dist[2] = 1;
    dist[3] = 1;

    ierr += testRandom(nModes, modeSizes, dist, comm);
  }
  catch (std::exception &e) {
    std::cout << me << " FAIL:  " << e.what() << std::endl;
    ierr++;
  }

  /////////////////////////////////
  if (me == 0) std::cout << "Test 6..." << std::endl;
  try
  {
    // Tests random generation with std and type short
    pt::mode_t nModes = 5;
    std::vector<size_t> modeSizes(nModes);
    std::vector<int> dist(nModes);
    modeSizes[0] = 10;
    modeSizes[1] = 20;
    modeSizes[2] = 300;
    modeSizes[3] = 40;
    modeSizes[4] = 50;
    dist[0] = 1;
    dist[1] = 1;
    dist[2] = np;
    dist[3] = 1;
    dist[4] = 1;

    ierr += testRandom(nModes, modeSizes, dist, comm);
  }
  catch (std::exception &e) {
    std::cout << me << " FAIL:  " << e.what() << std::endl;
    ierr++;
  }

  /////////////////////////////////
  if (me == 0) std::cout << "Test 7..." << std::endl;
  try
  { 
    // Test random generation with std and type int
    pt::mode_t nModes = 3;
    std::vector<size_t> modeSizes(nModes);
    std::vector<int> dist(nModes);
    modeSizes[0] = 10;
    modeSizes[1] = 200000;
    modeSizes[2] = 300;
    dist[0] = 1;
    dist[1] = np;
    dist[2] = 1;

    ierr += testRandom(nModes, modeSizes, dist, comm);
  }
  catch (std::exception &e) {
    std::cout << me << " FAIL:  " << e.what() << std::endl;
    ierr++;
  }

  /////////////////////////////////
  if (me == 0) std::cout << "Test 8..." << std::endl;
  if (np % 2 == 0) {
  try
  { 
    // Test corner case where dist[m] <= modeSizes[m] 
    //      and modeSizes[m] < np for some mode
    // Want to avoid divide-by-zero error.
    pt::mode_t nModes = 3;
    std::vector<size_t> modeSizes(nModes);
    std::vector<int> dist(nModes);
    modeSizes[0] = 10;
    modeSizes[1] = 10;
    modeSizes[2] = np-1;
    dist[0] = 2;
    dist[1] = 1;
    dist[2] = np/2;

    ierr += testRandom(nModes, modeSizes, dist, comm);
  }
  catch (std::exception &e) {
    std::cout << me << " FAIL:  " << e.what() << std::endl;
    ierr++;
  }
  }

  /////////////////////////////////
  if (me == 0) std::cout << "Test 9..." << std::endl;
  try
  { 
    // Test corner case where #procs > modeSizes for some mode
    // Want to avoid divide-by-zero error.
    // This test should throw an error when np > 1
    pt::mode_t nModes = 3;
    std::vector<size_t> modeSizes(nModes);
    std::vector<int> dist(nModes);
    modeSizes[0] = 10;
    modeSizes[1] = 10;
    modeSizes[2] = np / 4 + 1;
    dist[0] = 1;
    dist[1] = 1;
    dist[2] = np;

    ierr += testRandom(nModes, modeSizes, dist, comm);
  }
  catch (std::exception &e) {
    if (np == 1) {
      std::cout << me << " FAIL:  " << e.what() << std::endl;
      ierr++;
    }
    else {
      // This test should throw an error. 
      // Move on; nothing to see here.
    }
  }

  /////////////////////////////////
#if 0
// May 30:  This test fails due to failure of Tpetra::MultiVector allocation
// with 2B entries.  Test fails on my mac.  Stand-alone Tpetra::MultiVector
// allocation with 2B entries also fails.  Separate issue; should file with
// Trilinos github.
  if (me == 0) std::cout << "Test 10..." << std::endl;
  try
  { 
    // Test random generation with std and type long long
    pt::mode_t nModes = 3;
    std::vector<size_t> modeSizes(nModes);
    std::vector<int> dist(nModes);
    modeSizes[0] = 10;
    modeSizes[1] = 2000000000;
    modeSizes[2] = 300000;
    dist[0] = 1;
    dist[1] = np;
    dist[2] = 1;

    ierr += testRandom(nModes, modeSizes, dist, comm);
  }
  catch (std::exception &e) {
    std::cout << me << " FAIL:  " << e.what() << std::endl;
    ierr++;
  }
#endif

  /////////////////////////////////
  if (ierr) std::cout << me << ": " << ierr << " errors detected." << std::endl;

  int gierr = 0;
  Teuchos::reduceAll<int,int>(*comm, Teuchos::REDUCE_SUM, 1, &ierr, &gierr);

  if (me == 0) {
    if (gierr == 0) std::cout << " PASS" << std::endl;
    else            std::cout << " FAIL" << std::endl;
  }

  return gierr;
}
