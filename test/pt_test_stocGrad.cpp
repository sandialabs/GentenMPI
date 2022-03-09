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

#include "Teuchos_TimeMonitor.hpp"
#include "Tpetra_Core.hpp"

static int verbosity = 100;

//////////////////////////////////////////////////////////////////////////////
// Computes gradient 3 ways and compares results
template <typename sptensor_t, typename ktensor_t>
int compareGradients(
  sptensor_t *X,
  ktensor_t *K,
  int oversamplingFactor, 
  const Teuchos::RCP<const Teuchos::Comm<int> > &comm)
{
  int ierr=0;
  int me = comm->getRank();
  
  if (me==0)
    std::cout << "GB: localtensorsize: " << X->getLocalTensorSize() 
              << " numNonzeros: " << X->getLocalNumNonZeros() 
              << " numIndices: " << X->getLocalNumIndices() << std::endl;
 
  typedef typename sptensor_t::scalar_t scalar_t;
  typedef typename ktensor_t::factormatrix_t factormatrix_t;
 
  K->distributeLambda(0);
 
  pt::distSystem<sptensor_t, ktensor_t> distSys(X, K);

  std::vector<size_t> modeSizes = X->getModeSizes();
  int nModes = modeSizes.size();
  int rank = K->getFactorRank();
  
  if (me == 0) std::cout << "Computing Exact Gradient" << std::endl;
  
  typedef typename pt::squareLocalMatrix<scalar_t> slm_t;
  typedef typename pt::gramianMatrix<factormatrix_t> gram_t;
  
  // initialize gradient matrices matching system's ktensor
  ktensor_t exactGradient(rank, modeSizes, comm);
  
  // initialize upsilon for gradient computation
  slm_t upsilon(rank);
    
  // build gram matrices in each mode
  std::vector<gram_t *> gamma(nModes);
  for (pt::mode_t m = 0; m < nModes; m++) 
    gamma[m] = new gram_t(K->getFactorMatrix(m), true);
    
  // compute exact gradient for each mode: G = 2(upsilon*U - mttkrp) 
  Kokkos::View<scalar_t *> twos("test_stocGrad::twos",rank);
  Kokkos::deep_copy(twos, 2.);
  for (pt::mode_t m = 0; m < nModes; m++) {
    // compute upsilon for mode m
    upsilon.setValues(1.);
    for (pt::mode_t n = 0; n < nModes; n++) {
      if (n != m) {
        upsilon.hadamard(*(gamma[n]));
      }
    }

    // perform MTTKRP for mode m and store result in gradient ktensor
    distSys.mttkrp(m,exactGradient.getFactorMatrix(m));
    
    // compute gradient for mode m by multiplying factor matrix by upsilon
    // and subtracting the mttkrp result
    pt::symmRectMatmul(upsilon, *(K->getFactorMatrix(m)), 
                   *(exactGradient.getFactorMatrix(m)));
                   
    // scale results by 2
    exactGradient.getFactorMatrix(m)->scale(twos);
    
  }
  
  if (me == 0) std::cout << "Computing Stochastic Gradient" << std::endl;
    
  // set loss function and choose sample sizes
  pt::L2_lossFunction<scalar_t> f;
  size_t nNonZeros = X->getLocalNumNonZeros();
  double nZeros = X->getLocalTensorSize() - nNonZeros;
  size_t nzSample = oversamplingFactor*nNonZeros;
  size_t zSample = oversamplingFactor*nZeros;

  // initialize stochastic gradient matrices matching system's ktensor
  ktensor_t stocGradient(rank, modeSizes, comm);
  ktensor_t stocGradientSemiStrat(rank, modeSizes, comm);
  ktensor_t stocGradientFullTensor(rank, modeSizes, comm);

  Teuchos::RCP<Teuchos::Time>
    timeSample(Teuchos::TimeMonitor::getNewTimer(
                                     "Stoc Grad   Sample")),
    timeSystem(Teuchos::TimeMonitor::getNewTimer(
                                     "Stoc Grad   System")),
    timeDfDm(Teuchos::TimeMonitor::getNewTimer(
                                     "Stoc Grad   dF/dM")),
    timeMTTKRP(Teuchos::TimeMonitor::getNewTimer(
                                     "Stoc Grad   MTTKRP"));

  // compute stochastic gradient both ways
  pt::StratifiedSamplingStrategy<sptensor_t> stratified(X);
  sptensor_t *Y = NULL;
  distSys.stocGrad( nzSample, zSample, f, &stocGradient, Y, &stratified, 1, 
                    timeSample, timeSystem, timeDfDm, timeMTTKRP );
  Teuchos::TimeMonitor::summarize();
  Teuchos::TimeMonitor::zeroOutTimers();
  delete Y;
  Y = NULL;

  pt::SemiStratifiedSamplingStrategy<sptensor_t> semiStratified(X);
  distSys.stocGrad( nzSample, zSample, f, &stocGradientSemiStrat, Y,
                    &semiStratified, 1, 
                    timeSample, timeSystem, timeDfDm, timeMTTKRP );
  Teuchos::TimeMonitor::summarize();
  Teuchos::TimeMonitor::zeroOutTimers();
  delete Y;
  Y = NULL;

  pt::FullTensorSamplingStrategy<sptensor_t> fullTensor(X);
  distSys.stocGrad( nNonZeros, nZeros, f, &stocGradientFullTensor, Y,
                    &fullTensor, 1, 
                    timeSample, timeSystem, timeDfDm, timeMTTKRP );
  Teuchos::TimeMonitor::summarize();
  Teuchos::TimeMonitor::zeroOutTimers();
  delete Y;
  Y = NULL;

  if (me == 0) 
    std::cout << "NOT DONE YET:  STILL NEED TO CHECK THE ANSWER" << std::endl;
    
  // tolerance for passing unit test, 
  // tied to number of samples based on empirical 
  // results but not sure what mathematical relationship should be
  scalar_t tol = 5e-2;
  
  // check the difference between exactGradient and 
  // (stratified) stocGradient ktensors
  factormatrix_t *exactFM, *stocFM;
  scalar_t exactVal, stocVal;
  scalar_t err, maxErr = 0;
  for (pt::mode_t m = 0; m < nModes; m++) {
    for (size_t i = 0; i < exactGradient.getFactorMatrix(m)->getLocalLength(); i++) {
      for (pt::rank_t r = 0; r < rank; r++) {
        exactFM = exactGradient.getFactorMatrix(m);
        exactVal = (*exactFM)(i, r);
        stocFM = stocGradient.getFactorMatrix(m);
        stocVal = (*stocFM)(i, r);
        err = std::abs(exactVal-stocVal);
        if (err > maxErr) maxErr = err;
        if (!pt::nearlyEqual<scalar_t>(exactVal,stocVal,tol)) {
          std::cout << me << ": Error with strat stocGrad in mode " 
                    << m << ":  (" << i << "," << r << "): "
                    << stocVal << " != " << exactVal << std::endl;
          ierr++;
        }
      }
    }
  }
  std::cout << me << ": maximum strat error is " << maxErr << std::endl;
  
  // check the difference between exactGradient and semi-stratified ktensors
  maxErr = 0;
  for (pt::mode_t m = 0; m < nModes; m++) {
    for (size_t i = 0; i < exactGradient.getFactorMatrix(m)->getLocalLength(); i++) {
      for (pt::rank_t r = 0; r < rank; r++) {
        exactFM = exactGradient.getFactorMatrix(m);
        exactVal = (*exactFM)(i, r);
        stocFM = stocGradientSemiStrat.getFactorMatrix(m);
        stocVal = (*stocFM)(i, r);
        err = std::abs(exactVal-stocVal);
        if (err > maxErr) maxErr = err;
        if (!pt::nearlyEqual<scalar_t>(exactVal,stocVal,tol)) {
          std::cout << me << ": Error with semi-strat stocGrad in mode " << m << ":  (" <<
                    i << "," << r << "): "
                    << stocVal << " != " << exactVal << std::endl;
          ierr++;
        }
      }
    }
  }
  std::cout << me << ": maximum semi-strat error is " << maxErr << std::endl;

  // check the difference between exactGradient and full-tensor ktensors
  maxErr = 0;
  tol = 1e-12; // should be equivalent in exact arithmetic
  for (pt::mode_t m = 0; m < nModes; m++) {
    for (size_t i = 0; i < exactGradient.getFactorMatrix(m)->getLocalLength(); i++) {
      for (pt::rank_t r = 0; r < rank; r++) {
        exactFM = exactGradient.getFactorMatrix(m);
        exactVal = (*exactFM)(i, r);
        stocFM = stocGradientFullTensor.getFactorMatrix(m);
        stocVal = (*stocFM)(i, r);
        err = std::abs(exactVal-stocVal);
        if (err > maxErr) maxErr = err;
        if (!pt::nearlyEqual<scalar_t>(exactVal,stocVal,tol)) {
          std::cout << me << ": Error with fulltensor stocGrad in mode " << m << ":  (" <<
                    i << "," << r << "): "
                    << stocVal << " != " << exactVal << std::endl;
          ierr++;
        }
      }
    }
  }
  std::cout << me << ": maximum exact error is " << maxErr << std::endl;

  return ierr;
}
 

//////////////////////////////////////////////////////////////////////////////
// A small test problem 

int smallTest(const Teuchos::RCP<const Teuchos::Comm<int> > &comm)
{
  int me = comm->getRank();
  int np = comm->getSize();

  if (me == 0) std::cout << "Beginning smallTest" << std::endl;

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
  gindices(2,0) = 0;  gindices(2,1) = 1;  gindices(2,2) = 1;
  gvalues(2) = 1.0;
  gindices(3,0) = 1;  gindices(3,1) = 1;  gindices(3,2) = 1;
  gvalues(3) = 1.0;
  gindices(4,0) = 0;  gindices(4,1) = 2;  gindices(4,2) = 2;
  gvalues(4) = 1.0;
  gindices(5,0) = 0;  gindices(5,1) = 0;  gindices(5,2) = 2;
  gvalues(5) = 1.0;
  gindices(6,0) = 0;  gindices(6,1) = 2;  gindices(6,2) = 3;
  gvalues(6) = 1.0;
  gindices(7,0) = 0;  gindices(7,1) = 0;  gindices(7,2) = 3;
  gvalues(7) = 1.0;
  gindices(8,0) = 1;  gindices(8,1) = 0;  gindices(8,2) = 4;
  gvalues(8) = 1.0;
  gindices(9,0) = 0;  gindices(9,1) = 1;  gindices(9,2) = 5;
  gvalues(9) = 1.0;
  gindices(10,0) = 1; gindices(10,1) = 1; gindices(10,2) = 6;
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

  
  return compareGradients<sptensor_t, ktensor_t>(&sptensor,&ktensor,1000,comm);
  
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
  if (me == 0) std::cout << "Beginning randomTest " << msg << std::endl;
  
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
  sptensor_t *sptensor;
  try {
    sptensor = rd.createRandomSptensor(ktensor, targetNnz, dist);
  }
  catch (std::exception &e) {
    delete ktensor;
    throw e;
  }
  sptensor->printStats("RandomSptensor");
  if (verbosity) sptensor->print("RandomSptensor");

  int ierr = compareGradients<sptensor_t, ktensor_t>(sptensor, ktensor,
                                                     10000, comm);

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
    double density = 0.5;
    pt::rank_t rank = 8;
    pt::mode_t nModes = 4;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 3;
    modeSizes[1] = np;
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
    modeSizes[0] = np * 5;
    modeSizes[1] = np * 10;
    modeSizes[2] = 5;
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
