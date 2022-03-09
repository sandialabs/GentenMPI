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
 
// Thought from Cliff's talk on how to test CP-ALS.
// Generate the factor matrices, and from them, construct the tensor.
// Then, from the tensor, compute the factors and compare to generating factors.
//
// Separately:
// Cliff has a binary data generator; see presentation from 4/24/17
// Last slide has the way to do it in R.

#include "pt_test_generators.hpp"
#include "pt_test_distFromGenerators.hpp"
#include "pt_test_ttbFromGenerators.hpp"
#include "pt_test_compare.hpp"
#include "pt_system.hpp"

#include <TTB_Ktensor.h>
#include <TTB_Sptensor.h>
#include <TTB_RandomMT.h>
#include <TTB_CpAls.h>

#include "Teuchos_TimeMonitor.hpp"
#include "Tpetra_Core.hpp"

static int verbosity = 1;

//////////////////////////////////////////////////////////////////////////////
// A small test problem just to check correctness of CP-ALS.
// TODO:  Add TTB comparison.

int correctnessTest(const Teuchos::RCP<const Teuchos::Comm<int> > &comm)
{
  int ierr = 0;
  int me = comm->getRank();
  int np = comm->getSize();

  typedef typename pt::distSptensor<double> sptensor_t;
  typedef typename pt::distFactorMatrix<double> factormatrix_t;
  typedef typename pt::distKtensor<factormatrix_t> ktensor_t;
  typedef typename factormatrix_t::valueview_t factormatrixview_t;
  typedef typename factormatrix_t::gno_t gno_t;

  // Run the same simple test as from TTB_Test/TTB_Test_CpAls.cpp:
  // small tensor with known solution

  // Build the tensor
  pt::mode_t nModes = 3;
  std::vector<size_t> modeSizes(nModes);
  modeSizes[0] = 2;  modeSizes[1] = 3;  modeSizes[2] = 4;

  // View of the global problem; 
  // one wouldn't normally have a global view, but since this problem is
  // SOOOO small, we can do it.

  gno_t gnnz = 11;  // global
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
  gindices(10,0) = 1; gindices(10,1) = 1; gindices(10,2) = 3;
  gvalues(10) = 1.0;

  // This proc's subview of the global problem
  int nnz = gnnz / np + (me < (gnnz % np));  // local
  int first = me * (gnnz / np) + (me < (gnnz % np) ? me : gnnz % np);  // local
  typedef typename Kokkos::View<gno_t **>::size_type ksize_t;
  Kokkos::View<gno_t **> indices = 
          Kokkos::subview(gindices,
                          std::pair<ksize_t,ksize_t>(first, first+nnz), 
                          Kokkos::ALL());

  Kokkos::View<double *> values = 
          Kokkos::subview(gvalues, 
                          std::pair<ksize_t,ksize_t>(first, first+nnz));
  
  // Create the sparse tensor
  sptensor_t sptensor(nModes, modeSizes, indices, values, comm);

  sptensor.print("SPARSE TENSOR");

  // Initial factor matrices with default Tpetra maps
  pt::rank_t rank = 2;

  ktensor_t ktensor(rank, modeSizes, comm);

  ktensor.setLambda(1.);
  ktensor.getLambdaView()(0) = 2.;  // Matches TTB's first test

  // Global view of initial ktensor entries;
  // one wouldn't normally have a global view, but since this problem is
  // SOOOO small, we can do it.
  factormatrixview_t gfactorA("gfactorA", modeSizes[0], rank);
  factormatrixview_t gfactorB("gfactorB", modeSizes[1], rank);
  factormatrixview_t gfactorC("gfactorC", modeSizes[2], rank);

  gfactorA(0,0) = 0.8;
  gfactorA(1,0) = 0.2;
  gfactorA(0,1) = 0.5;
  gfactorA(1,1) = 0.5;
  gfactorB(0,0) = 0.5;
  gfactorB(1,0) = 0.1;
  gfactorB(2,0) = 0.5;
  gfactorB(0,1) = 0.5;
  gfactorB(1,1) = 0.5;
  gfactorB(2,1) = 0.1;
  gfactorC(0,0) = 0.7;
  gfactorC(1,0) = 0.7;
  gfactorC(2,0) = 0.1;
  gfactorC(3,0) = 0.1;
  gfactorC(0,1) = 0.7;
  gfactorC(1,1) = 0.1;
  gfactorC(2,1) = 0.1;
  gfactorC(3,1) = 0.7;

  std::vector<factormatrixview_t> gfactor(nModes);
  gfactor[0] = gfactorA;
  gfactor[1] = gfactorB;
  gfactor[2] = gfactorC;

  // Build subviews for local factor values
  for (pt::mode_t m = 0; m < nModes; m++) {
    factormatrix_t *fm = ktensor.getFactorMatrix(m);
    int firstgno = fm->getMap()->getMinGlobalIndex();
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

  pt::distSystem<sptensor_t, ktensor_t> distSys(&sptensor, &ktensor);

  double tol = 1.e-6;
  int maxIters = 100;
  int numIters = 0;
  double resNorm = 0.;

  if (me == 0) std::cout << "Calling CP_ALS" << std::endl;

  distSys.cp_als(tol, 1, maxIters, numIters, resNorm);

  if (me == 0) 
    std::cout << "NOT DONE YET:  STILL NEED TO CHECK THE ANSWER" << std::endl;

  return ierr;
}

//////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
int randomTest(const std::string &msg, scalar_t density, 
               pt::mode_t nModes, std::vector<size_t> &modeSizes,
               pt::rank_t rank, 
               const Teuchos::RCP<const Teuchos::Comm<int> > &comm)
{
  int ierr = 0;
  int me = comm->getRank();

  Teuchos::RCP<Teuchos::Time> 
        timerDIST(Teuchos::TimeMonitor::getNewTimer("DISTRIBUTED")),
        timerTTB(Teuchos::TimeMonitor::getNewTimer("TTB"));

  if (me == 0) {
    std::cout << std::endl << msg
              << ":  TESTING WITH <" << typeid(scalar_t).name() << "> "
              << "rank = " << rank << ", NMODES=" << nModes << " (";
    for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
    std::cout << ")" << std::endl;
  }
  
  typedef SptensorGenerator<scalar_t> sptensorGenerator_t;
  typedef generatedDistSptensor<sptensorGenerator_t> generatedDistSptensor_t;
  typedef typename generatedDistSptensor_t::sptensor_t sptensor_t;

  typedef pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef pt::distKtensor<factormatrix_t> ktensor_t;

  // Random ktensor with default maps
  ktensor_t *ktensor = new ktensor_t(rank, modeSizes, comm);
  ktensor->setRandomUniform();

  // Random sptensor with same dimensions as ktensor; block distribution
  sptensorGenerator_t sptensorGenerator(nModes, modeSizes, rank, density);
  generatedDistSptensor_t distSptensor(sptensorGenerator, ktensor, comm, 0);
  sptensor_t *sptensor = distSptensor.getSptensor();
  sptensor->setValues(1.);

  // A little output
  std::cout << me << " DIST Sptensor nnz = " << sptensor->getLocalNumNonZeros()
            << std::endl;

  // Create distSystem with sptensor and ktensor
  typedef typename pt::distSystem<sptensor_t, ktensor_t> distSystem_t;
  distSystem_t distSys(sptensor, ktensor);

  double tol = 1.e-6;
  int maxIters = 100;
  int numIters = 0;
  double resNorm = 0.;

  if (me == 0) std::cout << "Calling DIST CP_ALS" << std::endl;

  comm->barrier();

  {
    Teuchos::TimeMonitor lcltimer(*timerDIST);
    distSys.cp_als(tol, 1, maxIters, numIters, resNorm);
  }

  if (me == 0) 
    std::cout << "DIST CP_ALS numIters " << numIters
              << " resNorm " << resNorm << std::endl;
  
  // Now try same problem with TTB
  if (me == 0) {
    TTB::RandomMT  cRNG(1);
    TTB::Ktensor *ttb_ktensor = new TTB::Ktensor(rank, nModes);
    TTB::IndxArray ttb_modeSizes(nModes);
    for (int m = 0; m < nModes; m++) ttb_modeSizes[m] = modeSizes[m];
    ttb_ktensor->setRandomUniform(ttb_modeSizes, rank, false, cRNG);

    typedef generatedTTBSptensor<sptensorGenerator_t> generatedTTBSptensor_t;
    generatedTTBSptensor_t ttbSptensor(sptensorGenerator, ttb_ktensor, 0);
    TTB::Sptensor *ttb_sptensor = ttbSptensor.getSptensor();
    for (ttb_indx nz = 0; nz < ttb_sptensor->nnz(); nz++)
      ttb_sptensor->value(nz) = 1.;
    std::cout << me << " TTB Sptensor nnz = " 
              << ttb_sptensor->nnz() << std::endl;
    

    ttb_indx ttb_numIters = 0;
    ttb_real ttb_resNorm = 0.;
    try
    {
       // Request performance information on every iteration.
       // Allocation adds two more for start and stop states of the algorithm.
       ttb_indx  nMaxPerfSize = 2 + maxIters;
       TTB::CpAlsPerfInfo *  perfInfo = new TTB::CpAlsPerfInfo[nMaxPerfSize];
       std::cout << "Calling TTB CP_ALS" << std::endl;
       {
         Teuchos::TimeMonitor lcltimer(*timerTTB);
         TTB::cpals_core<TTB::Sptensor>(*ttb_sptensor, *ttb_ktensor,
                                        tol, maxIters, -1.0, 1,
                                        ttb_numIters, ttb_resNorm,
                                        1, perfInfo);
       }
       std::cout << "TTB CP_ALS numIters " << ttb_numIters 
                 << " resNorm " << ttb_resNorm << std::endl;
   
       delete[] perfInfo;
    }
    catch(std::string sExc)
    {
      std::cout << "Error thrown from TTB::cpals_core " << sExc << std::endl;
    }
    delete ttb_ktensor;
  }

  if (me == 0) 
    std::cout << "NOT DONE YET:  STILL NEED TO CHECK THE ANSWER" << std::endl;

  delete ktensor;

  Teuchos::TimeMonitor::summarize();
  Teuchos::TimeMonitor::zeroOutTimers();
  return ierr;
}

//////////////////////////////////////////////////////////////////////////////

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

  ierr += correctnessTest(comm);

  {
    double density = 0.005;
    pt::rank_t rank = 10;
    pt::mode_t nModes = 5;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 3;
    modeSizes[1] = 5;
    modeSizes[2] = 2;
    modeSizes[3] = 2;
    modeSizes[4] = 5;
    ierr += randomTest<double>("small random", density,
                                               nModes, modeSizes,
                                               rank, comm);
  }

  {
    double density = 0.005;
    pt::rank_t rank = 1;
    pt::mode_t nModes = 3;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = np * 20;
    modeSizes[1] = np * 10;
    modeSizes[2] = np * 5;
    ierr += randomTest<double>("large random", density,
                                               nModes, modeSizes,
                                               rank, comm);
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
