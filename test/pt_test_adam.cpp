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
 
#include "pt_test_generators.hpp"
#include "pt_test_distFromGenerators.hpp"
#include "pt_test_compare.hpp"
#include "pt_system.hpp"
#include "pt_random.hpp"
#include "pt_lossfns.hpp"

#include "Teuchos_TimeMonitor.hpp"
#include "Tpetra_Core.hpp"

static int verbosity = 3;

//////////////////////////////////////////////////////////////////////////////
// Computes gradient 3 ways and compares results
template <typename sptensor_t, typename ktensor_t>
int runAdam(
  sptensor_t *sptensor,
  ktensor_t *ktensor,
  const Teuchos::RCP<const Teuchos::Comm<int> > &comm
)
{
  typedef typename sptensor_t::scalar_t scalar_t;

  int ierr=0;
  int me = comm->getRank();

  typedef typename pt::distSystem<sptensor_t, ktensor_t> distSystem_t;
  distSystem_t *distSys;
 
  // Get sense of achievable norm with CP-ALS
  if (me == 0) std::cout << "\n Running CP-ALS" << std::endl;

  ktensor_t *ktensorTest = new ktensor_t(ktensor);
  distSys = new distSystem_t(sptensor, ktensorTest);

  const double tolerance = 1e-1;
  const int minIter = 5, maxIter = 50;
  int numIterCPALS = 0;
  scalar_t resNormCPALS = 0.0;
  distSys->cp_als(tolerance, minIter, maxIter, numIterCPALS, resNormCPALS);

  if (me == 0) {
    std::cout << "...took " << numIterCPALS << " iterations" << std::endl;
    std::cout << "...achieved " << resNormCPALS << " residual norm" 
              << std::endl;
  }

  delete distSys;
  delete ktensorTest;

  Teuchos::TimeMonitor::summarize();
  Teuchos::TimeMonitor::zeroOutTimers();
  
  // set loss function for stochastic methods
  pt::L2_lossFunction<scalar_t> l2;

  // run Adam with L2 loss function and stratified sampling
  if (me == 0) std::cout << "\n Running Adam -- SemiStratified " << std::endl;
  
  ktensorTest = new ktensor_t(ktensor);
  distSys = new distSystem_t(sptensor, ktensorTest);

  // set problem parameters
  Teuchos::ParameterList paramsSemi;
  paramsSemi.set("sampling", "semi-stratified");
  paramsSemi.set("randomizeKtensor", true);
  paramsSemi.set("maxEpochs", 20);
  paramsSemi.set("debug", true);
  paramsSemi.set("stats", true);

  // run starting from random ktensor
  distSys->GCP_Adam(l2, paramsSemi);

  scalar_t resNormAdamSemi = distSys->getResidualNorm();
  if (me == 0) {
    std::cout << "GCP_Adam with Semi-stratified sampling" << std::endl;
    std::cout << "...achieved " << resNormAdamSemi
              << " residual norm" << std::endl;
  }
  
  delete distSys;
  delete ktensorTest;

  Teuchos::TimeMonitor::summarize();
  Teuchos::TimeMonitor::zeroOutTimers();

  // run Adam with L2 loss function and stratified sampling
  if (me == 0) std::cout << "\n Running Adam -- Stratified " << std::endl;
  
  ktensorTest = new ktensor_t(ktensor);
  distSys = new distSystem_t(sptensor, ktensorTest);

  // set problem parameters
  Teuchos::ParameterList paramsStrat;
  paramsStrat.set("sampling", "stratified");
  paramsStrat.set("randomizeKtensor", true);
  paramsStrat.set("maxEpochs", 20);
  paramsStrat.set("debug", true);
  paramsStrat.set("stats", true);

  // run starting from random ktensor
  distSys->GCP_Adam(l2, paramsStrat);

  scalar_t resNormAdamStrat = distSys->getResidualNorm();
  if (me == 0) {
    std::cout << "GCP_Adam with Stratified sampling" << std::endl;
    std::cout << "...achieved " << resNormAdamStrat
              << " residual norm" << std::endl;
  }
  
  delete distSys;
  delete ktensorTest;

  Teuchos::TimeMonitor::summarize();
  Teuchos::TimeMonitor::zeroOutTimers();

  if (me == 0) {
    std::cout << "RESIDUAL NORMS:  CP-ALS = " << resNormCPALS
              << "  GCP-SEMISTRAT = " << resNormAdamSemi
              << "  GCP-STRAT = " << resNormAdamStrat << std::endl;
    std::cout << "NOT DONE YET:  STILL NEED TO CHECK THE ANSWER" << std::endl;
  }

  return ierr;
}

template <typename sptensor_t, typename ktensor_t, typename loss_t>
int runAdamOtherLossFcns(
  sptensor_t *sptensor,
  ktensor_t *ktensor,
  const Teuchos::RCP<const Teuchos::Comm<int> > &comm
)
{
  typedef typename sptensor_t::scalar_t scalar_t;

  int ierr=0;
  int me = comm->getRank();

  typedef typename pt::distSystem<sptensor_t, ktensor_t> distSystem_t;
  distSystem_t *distSys;
  
  // set loss function for stochastic methods
  loss_t loss;

  // run Adam with specified loss function and stratified sampling
  if (me == 0) std::cout << "\n Running Adam -- SemiStratified " << std::endl;
  
  ktensor_t* ktensorTest = new ktensor_t(ktensor);
  distSys = new distSystem_t(sptensor, ktensorTest);

  // set problem parameters
  Teuchos::ParameterList paramsSemi;
  paramsSemi.set("sampling", "semi-stratified");
  paramsSemi.set("randomizeKtensor", true);
  paramsSemi.set("maxEpochs", 20);
  paramsSemi.set("debug", true);
  paramsSemi.set("stats", true);

  // run starting from random ktensor
  distSys->GCP_Adam(loss, paramsSemi);

  scalar_t resNormAdamSemi = distSys->getResidualNorm();
  if (me == 0) {
    std::cout << "GCP_Adam with Semi-stratified sampling" << std::endl;
    std::cout << "...achieved " << resNormAdamSemi
              << " residual l2 norm" << std::endl;
  }
  
  delete distSys;
  delete ktensorTest;

  Teuchos::TimeMonitor::summarize();
  Teuchos::TimeMonitor::zeroOutTimers();

  return ierr;
}
 

//////////////////////////////////////////////////////////////////////////////
// A small test problem 

int smallTest(const Teuchos::RCP<const Teuchos::Comm<int> > &comm)
{
  int me = comm->getRank();
  int np = comm->getSize();

  typedef double scalar_t;
  typedef typename pt::distSptensor<scalar_t> sptensor_t;
  typedef typename pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename pt::distKtensor<factormatrix_t> ktensor_t;
  typedef typename factormatrix_t::valueview_t factormatrixview_t;
  typedef typename factormatrix_t::gno_t gno_t;

  // Build the tensor
  pt::mode_t nModes = 3;
  std::vector<size_t> modeSizes(nModes);
  modeSizes[0] = 2;  modeSizes[1] = 3;  modeSizes[2] = 7;

  // View of the global problem; 
  // one wouldn't normally have a global view, but since this problem is
  // SOOOO small, we can do it.

  size_t gnnz = 11;  // global
  Kokkos::View<gno_t **> gindices("gindices", gnnz, nModes);
  Kokkos::View<double *> gvalues("gvalues", gnnz);

  gindices(0,0) = 0;  gindices(0,1) = 0;  gindices(0,2) = 0;
  gvalues(0) = 2.0;
  gindices(1,0) = 1;  gindices(1,1) = 0;  gindices(1,2) = 0;
  gvalues(1) = 1.0;
  gindices(2,0) = 0;  gindices(2,1) = 1;  gindices(2,2) = 0;
  gvalues(2) = 1.0;
  gindices(3,0) = 1;  gindices(3,1) = 1;  gindices(3,2) = 0;
  gvalues(3) = 1.0;
  gindices(4,0) = 0;  gindices(4,1) = 2;  gindices(4,2) = 0;
  gvalues(4) = 1.0;
  gindices(5,0) = 0;  gindices(5,1) = 0;  gindices(5,2) = 1;
  gvalues(5) = 1.0;
  gindices(6,0) = 0;  gindices(6,1) = 2;  gindices(6,2) = 1;
  gvalues(6) = 1.0;
  gindices(7,0) = 0;  gindices(7,1) = 0;  gindices(7,2) = 3;
  gvalues(7) = 1.0;
  gindices(8,0) = 1;  gindices(8,1) = 0;  gindices(8,2) = 3;
  gvalues(8) = 1.0;
  gindices(9,0) = 0;  gindices(9,1) = 1;  gindices(9,2) = 3;
  gvalues(9) = 1.0;
  gindices(10,0) = 1;  gindices(10,1) = 1;  gindices(10,2) = 3;
  gvalues(10) = 1.0;


  // create bounding boxes for sampling: split last mode among processors
  std::vector<gno_t> bbmin(nModes);
  std::vector<size_t> bbmodesize(nModes);
  for (pt::mode_t m = 0; m < nModes-1; m++) 
    bbmodesize[m] = modeSizes[m];
  size_t lastModeSize = modeSizes[nModes-1];
  bbmin[nModes-1] = me * (lastModeSize / np) + 
                    std::min<gno_t>(me, lastModeSize % np);
  bbmodesize[nModes-1] = (lastModeSize / np) + (me < int(lastModeSize % np));

  // This proc's subview of the global problem:  
  // assign nonzeros by bounding box, using the split of last mode above
  // Depending on the fact 
  size_t nnz = 0;
  for (size_t i = 0; i < gnnz; i++) {
    if (gindices(i, 2) >= bbmin[2] && 
        gindices(i,2) < bbmin[2]+static_cast<gno_t>(bbmodesize[2])) {
      for (pt::mode_t m = 0; m < nModes; m++) gindices(nnz,m) = gindices(i,m);
      gvalues(nnz) = gvalues(i);
      nnz++;
    }
  }
 
  typedef typename Kokkos::View<gno_t **>::size_type ksize_t;
  Kokkos::View<gno_t **> indices = 
          Kokkos::subview(gindices,
                          std::pair<ksize_t,ksize_t>(0, nnz), 
                          Kokkos::ALL());

  Kokkos::View<double *> values = 
          Kokkos::subview(gvalues, 
                          std::pair<ksize_t,ksize_t>(0, nnz));
  
  // Create the sparse tensor
  sptensor_t sptensor(nModes, modeSizes, indices, values, comm, bbmin, bbmodesize);
  
  sptensor.print("SPARSE TENSOR");

  // Initial factor matrices with default Tpetra maps
  pt::rank_t rank = 2;

  ktensor_t ktensor(rank, modeSizes, comm);

  ktensor.setLambda(1.);

  // Global view of initial ktensor entries;
  // one wouldn't normally have a global view, but since this problem is
  // SOOOO small, we can do it.
  factormatrixview_t gfactorA("gfactorA", modeSizes[0], rank);
  factormatrixview_t gfactorB("gfactorB", modeSizes[1], rank);
  factormatrixview_t gfactorC("gfactorC", modeSizes[2], rank);

  gfactorA(0,0) = 0.1;
  gfactorA(1,0) = 0.2;
  gfactorA(0,1) = 0.3;
  gfactorA(1,1) = 0.4;
  gfactorB(0,0) = 0.1;
  gfactorB(1,0) = 0.2;
  gfactorB(2,0) = 0.5;
  gfactorB(0,1) = 0.7;
  gfactorB(1,1) = -0.1;
  gfactorB(2,1) = -0.15;
  gfactorC(0,0) = 0.7;
  gfactorC(1,0) = 0.6;
  gfactorC(2,0) = 0.5;
  gfactorC(3,0) = 0.4;
  gfactorC(4,0) = 0.3;
  gfactorC(5,0) = 0.2;
  gfactorC(6,0) = 0.1;
  gfactorC(0,1) = -0.1;
  gfactorC(1,1) = 0.1;
  gfactorC(2,1) = -0.1;
  gfactorC(3,1) = 0.1;
  gfactorC(4,1) = -0.1;
  gfactorC(5,1) = 0.1;
  gfactorC(6,1) = -0.1;

  std::vector<factormatrixview_t> gfactor(nModes);
  gfactor[0] = gfactorA;
  gfactor[1] = gfactorB;
  gfactor[2] = gfactorC;

  // Build subviews for local factor values
  for (pt::mode_t m = 0; m < nModes; m++) {
    factormatrix_t *fm = ktensor.getFactorMatrix(m);
    gno_t firstgno = fm->getMap()->getMinGlobalIndex();
    int len = fm->getLocalLength();
    factormatrixview_t local = 
                       Kokkos::subview(gfactor[m],
                       std::pair<ksize_t,ksize_t>(firstgno, firstgno+len), 
                       Kokkos::ALL());
    Kokkos::deep_copy(fm->getLocalView(), local);
  }
  ktensor.print("KTENSOR");

  if (me == 0) {
    std::cout << std::endl
              << "TESTING WITH <DOUBLE, INT, INT> AND RANK="
              << rank << ", NMODES=" << nModes << " (";
    for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
    std::cout << ")" << std::endl;
  }

  return runAdam<sptensor_t, ktensor_t>(&sptensor, &ktensor, comm);
}

//////////////////////////////////////////////////////////////////////////////
// A small test problem for all loss functions
// * this is adapted from other small test function, 
// * creating binary tensor with nonnegative initialization to satisfy 
// * assumptions of all loss functions
template <typename loss_t>
int smallTestOtherLossFcns(const Teuchos::RCP<const Teuchos::Comm<int> > &comm)
{
  int me = comm->getRank();
  int np = comm->getSize();

  typedef double scalar_t;
  typedef typename pt::distSptensor<scalar_t> sptensor_t;
  typedef typename pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename pt::distKtensor<factormatrix_t> ktensor_t;
  typedef typename factormatrix_t::valueview_t factormatrixview_t;
  typedef typename factormatrix_t::gno_t gno_t;

  // Build the tensor
  pt::mode_t nModes = 3;
  std::vector<size_t> modeSizes(nModes);
  modeSizes[0] = 2;  modeSizes[1] = 3;  modeSizes[2] = 7;

  // View of the global problem; 
  // one wouldn't normally have a global view, but since this problem is
  // SOOOO small, we can do it.

  size_t gnnz = 11;  // global
  Kokkos::View<gno_t **> gindices("gindices", gnnz, nModes);
  Kokkos::View<double *> gvalues("gvalues", gnnz);

  gindices(0,0) = 0;  gindices(0,1) = 0;  gindices(0,2) = 0;
  gvalues(0) = 1.0;
  gindices(1,0) = 1;  gindices(1,1) = 0;  gindices(1,2) = 0;
  gvalues(1) = 1.0;
  gindices(2,0) = 0;  gindices(2,1) = 1;  gindices(2,2) = 0;
  gvalues(2) = 1.0;
  gindices(3,0) = 1;  gindices(3,1) = 1;  gindices(3,2) = 0;
  gvalues(3) = 1.0;
  gindices(4,0) = 0;  gindices(4,1) = 2;  gindices(4,2) = 0;
  gvalues(4) = 1.0;
  gindices(5,0) = 0;  gindices(5,1) = 0;  gindices(5,2) = 1;
  gvalues(5) = 1.0;
  gindices(6,0) = 0;  gindices(6,1) = 2;  gindices(6,2) = 1;
  gvalues(6) = 1.0;
  gindices(7,0) = 0;  gindices(7,1) = 0;  gindices(7,2) = 3;
  gvalues(7) = 1.0;
  gindices(8,0) = 1;  gindices(8,1) = 0;  gindices(8,2) = 3;
  gvalues(8) = 1.0;
  gindices(9,0) = 0;  gindices(9,1) = 1;  gindices(9,2) = 3;
  gvalues(9) = 1.0;
  gindices(10,0) = 1;  gindices(10,1) = 1;  gindices(10,2) = 3;
  gvalues(10) = 1.0;


  // create bounding boxes for sampling: split last mode among processors
  std::vector<gno_t> bbmin(nModes);
  std::vector<size_t> bbmodesize(nModes);
  for (pt::mode_t m = 0; m < nModes-1; m++) 
    bbmodesize[m] = modeSizes[m];
  size_t lastModeSize = modeSizes[nModes-1];
  bbmin[nModes-1] = me * (lastModeSize / np) + 
                    std::min<gno_t>(me, lastModeSize % np);
  bbmodesize[nModes-1] = (lastModeSize / np) + (me < int(lastModeSize % np));

  // This proc's subview of the global problem:  
  // assign nonzeros by bounding box, using the split of last mode above
  // Depending on the fact 
  size_t nnz = 0;
  for (size_t i = 0; i < gnnz; i++) {
    if (gindices(i, 2) >= bbmin[2] && 
        gindices(i,2) < bbmin[2]+static_cast<gno_t>(bbmodesize[2])) {
      for (pt::mode_t m = 0; m < nModes; m++) gindices(nnz,m) = gindices(i,m);
      gvalues(nnz) = gvalues(i);
      nnz++;
    }
  }
 
  typedef typename Kokkos::View<gno_t **>::size_type ksize_t;
  Kokkos::View<gno_t **> indices = 
          Kokkos::subview(gindices,
                          std::pair<ksize_t,ksize_t>(0, nnz), 
                          Kokkos::ALL());

  Kokkos::View<double *> values = 
          Kokkos::subview(gvalues, 
                          std::pair<ksize_t,ksize_t>(0, nnz));
  
  // Create the sparse tensor
  sptensor_t sptensor(nModes, modeSizes, indices, values, comm, bbmin, bbmodesize);
  
  sptensor.print("SPARSE TENSOR");

  // Initial factor matrices with default Tpetra maps
  pt::rank_t rank = 2;

  ktensor_t ktensor(rank, modeSizes, comm);

  ktensor.setLambda(1.);

  // Global view of initial ktensor entries;
  // one wouldn't normally have a global view, but since this problem is
  // SOOOO small, we can do it.
  factormatrixview_t gfactorA("gfactorA", modeSizes[0], rank);
  factormatrixview_t gfactorB("gfactorB", modeSizes[1], rank);
  factormatrixview_t gfactorC("gfactorC", modeSizes[2], rank);

  gfactorA(0,0) = 0.1;
  gfactorA(1,0) = 0.2;
  gfactorA(0,1) = 0.3;
  gfactorA(1,1) = 0.4;
  gfactorB(0,0) = 0.1;
  gfactorB(1,0) = 0.2;
  gfactorB(2,0) = 0.5;
  gfactorB(0,1) = 0.7;
  gfactorB(1,1) = 0.1;
  gfactorB(2,1) = 0.15;
  gfactorC(0,0) = 0.7;
  gfactorC(1,0) = 0.6;
  gfactorC(2,0) = 0.5;
  gfactorC(3,0) = 0.4;
  gfactorC(4,0) = 0.3;
  gfactorC(5,0) = 0.2;
  gfactorC(6,0) = 0.1;
  gfactorC(0,1) = 0.1;
  gfactorC(1,1) = 0.1;
  gfactorC(2,1) = 0.1;
  gfactorC(3,1) = 0.1;
  gfactorC(4,1) = 0.1;
  gfactorC(5,1) = 0.1;
  gfactorC(6,1) = 0.1;

  std::vector<factormatrixview_t> gfactor(nModes);
  gfactor[0] = gfactorA;
  gfactor[1] = gfactorB;
  gfactor[2] = gfactorC;

  // Build subviews for local factor values
  for (pt::mode_t m = 0; m < nModes; m++) {
    factormatrix_t *fm = ktensor.getFactorMatrix(m);
    gno_t firstgno = fm->getMap()->getMinGlobalIndex();
    int len = fm->getLocalLength();
    factormatrixview_t local = 
                       Kokkos::subview(gfactor[m],
                       std::pair<ksize_t,ksize_t>(firstgno, firstgno+len), 
                       Kokkos::ALL());
    Kokkos::deep_copy(fm->getLocalView(), local);
  }
  ktensor.print("KTENSOR");

  if (me == 0) {
    std::cout << std::endl
              << "TESTING WITH <DOUBLE, INT, INT> AND RANK="
              << rank << ", NMODES=" << nModes << " (";
    for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
    std::cout << ")" << std::endl;
  }

  return runAdamOtherLossFcns<sptensor_t, ktensor_t,  loss_t>(&sptensor, &ktensor, comm);
}

//////////////////////////////////////////////////////////////////////////////
// A random test problem 
// 

template <typename scalar_t>
int randomTest(const std::string &msg, scalar_t density, 
               pt::mode_t nModes, std::vector<size_t> &modeSizes,
               pt::rank_t rank,
               const Teuchos::RCP<const Teuchos::Comm<int> > &comm,
               std::vector<int> &dist)
{
  int me = comm->getRank();
  
  typedef pt::distSptensor<scalar_t> sptensor_t;
  typedef pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef pt::distKtensor<factormatrix_t> ktensor_t;
  typedef typename factormatrix_t::gno_t gno_t;

  // Generate nonzero elements in sptensor relative to ktensor
  if (me == 0) std::cout << "Creating random sptensor..." << std::endl;
  size_t totalIndices = std::accumulate(std::begin(modeSizes), 
                                        std::end(modeSizes), 1, 
                                        std::multiplies<size_t>());
  size_t targetNnz = density * totalIndices;

  pt::randomData<sptensor_t, ktensor_t> rd(comm);
  ktensor_t *ktensor = rd.createRandomKtensor(rank, modeSizes);
  if (verbosity >= 2) ktensor->print("RandomKtensor");

  sptensor_t *sptensor;
  try {
    sptensor = rd.createRandomSptensor(ktensor, targetNnz, dist);
  }
  catch (std::exception &e) {
    delete ktensor;
    throw e;
  }
  sptensor->printStats("RandomSptensor");
  if (verbosity == 3) sptensor->print("RandomSptensor");

  int ierr = runAdam<sptensor_t, ktensor_t>(sptensor, ktensor, comm);

  delete ktensor;
  delete sptensor;

  return ierr;
}



int main(int narg, char *arg[])
{
  // Usual Teuchos::Comm initialization
  Tpetra::ScopeGuard scopeguard(&narg, &arg);
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int me = comm->getRank();
  int np = comm->getSize();

  // Accept command-line parameters
  Teuchos::CommandLineProcessor cmdp (false, false);
  cmdp.setOption("verbosity", &verbosity,
                 "0 = quiet; 1 = print status; 2 = print factor matrices; "
                 "3 = print tensor");
  cmdp.parse(narg, arg);

  // Run the tests; count how many fail.
  int ierr = 0;

  ierr += smallTest(comm);

  {
    ierr += smallTestOtherLossFcns<pt::Poisson_lossFunction<double>>(comm);
    ierr += smallTestOtherLossFcns<pt::Bernoulli_odds_lossFunction<double>>(comm);
    ierr += smallTestOtherLossFcns<pt::Bernoulli_logit_lossFunction<double>>(comm);
    ierr += smallTestOtherLossFcns<pt::Poisson_log_lossFunction<double>>(comm);
    ierr += smallTestOtherLossFcns<pt::Rayleigh_lossFunction<double>>(comm);
    ierr += smallTestOtherLossFcns<pt::Gamma_lossFunction<double>>(comm);
    
  }
  
  {
    double density = 0.5;
    pt::rank_t rank = 8;
    pt::mode_t nModes = 4;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 3;
    modeSizes[1] = np*2;
    modeSizes[2] = 3;
    modeSizes[3] = 2;
    std::vector<int> dist(nModes);
    dist[0] = 1;
    dist[1] = np;
    dist[2] = 1;
    dist[3] = 1;
    ierr += randomTest<double>("small random", density,
                                               nModes, modeSizes,
                                               rank, comm, dist);
  }

  {
    double density = 0.05;
    pt::rank_t rank = 3;
    pt::mode_t nModes = 3;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = np * 20;
    modeSizes[1] = np * 10;
    modeSizes[2] = 50;
    std::vector<int> dist(nModes);
    dist[0] = np;
    dist[1] = 1;
    dist[2] = 1;
    ierr += randomTest<double>("large random", density,
                                               nModes, modeSizes,
                                               rank, comm, dist);
  }

  // Gather total error counts
  if (ierr)
    std::cout << me << ": " << ierr << " errors detected." << std::endl;

  int gierr = 0;
  Teuchos::reduceAll<int,int>(*comm, Teuchos::REDUCE_SUM, 1, &ierr, &gierr);

  if (me == 0) {
    if (gierr == 0) std::cout << " PASS" << std::endl;
    else            std::cout << " FAIL" << std::endl;
  }

  return gierr;
}
