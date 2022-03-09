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
 
// Can we test it by building mode-2 tensor and comparing with matrix ops?
// Is MTTKRP == MatVec in mode-2 tensor?
// Test first in serial, without the imports/exports/distributions.
// Is numVectors == int in Tpetra::MultiVector?


#include "pt_test_generators.hpp"
#include "pt_test_distFromGenerators.hpp"
#include "pt_test_ttbFromGenerators.hpp"
#include "pt_test_compare.hpp"
#include "pt_system.hpp"
#include "Tpetra_Core.hpp"

#include <TTB_MixedFormatOps.h>

static int verbosity = 1;

//////////////////////////////////////////////////////////////////////////////
template <typename distmttkrp_t, typename benchmarkmttkrp_t>
int compareDistBenchmark(distmttkrp_t &dist, benchmarkmttkrp_t &benchmark)
{
  int ierr = 0;

  typedef typename distmttkrp_t::factormatrix_t factormatrix_t;
  typedef typename factormatrix_t::gno_t gno_t;
  typedef typename factormatrix_t::scalar_t scalar_t;

  factormatrix_t *distResult = dist.getResult();

  auto distMap = distResult->getMap();
  int me = distMap->getComm()->getRank();

  pt::rank_t rank = distResult->getFactorRank();
  size_t len = distResult->getLocalLength();
  typename factormatrix_t::valueview_t data = distResult->getLocalView();

  // Loop over result values; compare to benchmark solution
  for (size_t i = 0; i < len; i++) {
    gno_t gidx = distMap->getGlobalElement(i);

    for (pt::rank_t r = 0; r < rank; r++) {
      scalar_t aValue = benchmark.getGlobalValue(gidx, r);
      scalar_t dValue = data(i,r);

      if (!pt::nearlyEqual(aValue, dValue,
          20.*std::numeric_limits<scalar_t>::epsilon())) {
        ierr++;
        std::cout << me
                  << " Error: gid " << gidx << " lid " << i << " rank " << r
                  << ": " << aValue << " != " << dValue 
                  << " (" << std::abs<scalar_t>(aValue - dValue) << ")" 
                  << std::endl;
      }
    }
  }
  return ierr;
}

//////////////////////////////////////////////////////////////////////////////
// Compare two serial mttkrp
template <typename mttkrp_t, typename benchmarkmttkrp_t>
int compareBenchmark(mttkrp_t &exp, benchmarkmttkrp_t &benchmark)
{
  int ierr = 0;

  pt::rank_t rank = benchmark.getResult()->getFactorRank();
  size_t len = benchmark.getResult()->getLocalLength();

  // Loop over result values; compare to benchmark solution
  for (size_t i = 0; i < len; i++) {
    for (pt::rank_t r = 0; r < rank; r++) {

      double aValue = benchmark.getGlobalValue(i, r);
      double dValue = exp.getGlobalValue(i, r);

      if (!pt::nearlyEqual(aValue, dValue,
          20.*std::numeric_limits<double>::epsilon())) {
        ierr++;
        std::cout << " Error: gid " << i << " rank " << r
                  << ": " << aValue << " != " << dValue 
                  << " (" << std::abs<double>(aValue - dValue) << ")" 
                  << std::endl;
      }
    }
  }
  return ierr;
}
//////////////////////////////////////////////////////////////////////////////
// Analytic mttkrp results, given a generated Ktensor and Sptensor as input

template <typename sptensorGenerator_t, typename ktensorGenerator_t>
class analyticMTTKRP {
public:

  typedef typename sptensorGenerator_t::gno_t gno_t;
  typedef typename sptensorGenerator_t::scalar_t scalar_t;

  typedef typename pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename factormatrix_t::map_t map_t;
  typedef typename factormatrix_t::valueview_t valueview_t;

  analyticMTTKRP(
    pt::mode_t mode,
    sptensorGenerator_t &sptensor,
    ktensorGenerator_t &ktensor
  )
  {
    pt::rank_t rank = ktensor.getFactorRank();
    size_t modeSize = ktensor.getSize(mode);
    pt::mode_t nModes = ktensor.getNumModes();

    // build factor matrix of global problem on each proc using serial comm 
    Teuchos::RCP<const Teuchos::Comm<int> > serialcomm = 
                       Teuchos::rcp(new Teuchos::SerialComm<int>);
    result = new factormatrix_t(rank, modeSize, serialcomm);
    
    // set values in factor matrix using analytic input values
    valueview_t data = result->getLocalView();

    typedef typename sptensorGenerator_t::nonzero_t nonzero_t;
  
    nonzero_t nz = sptensor.getFirstNonzero();
    while (sptensor.validNonzero(nz)) {
      gno_t idx = sptensor.getIndex(nz, mode);
      for (pt::rank_t r = 0; r < rank; r++) {
        scalar_t tmp = sptensor.getGlobalValue(nz);
        for (pt::mode_t m = 0; m < nModes; m++) {
          if (m == mode) continue;
          tmp *= ktensor.getGlobalValue(m, sptensor.getIndex(nz, m), r);
        }
        data(idx, r) += tmp;
      }
      nz = sptensor.getNextNonzero();
    }
  }

  ~analyticMTTKRP() { delete result; }

  inline scalar_t getGlobalValue(gno_t gidx, pt::rank_t r) { 
    return result->getLocalView()(gidx, r);
  }

  factormatrix_t *getResult() { return result; }

  inline void print(const std::string &msg) { result->print(msg); }

private:
  factormatrix_t *result;
};

//////////////////////////////////////////////////////////////////////////////
// distributed mttkrp results, given a Ktensor and Sptensor as input
template <typename distsys_t>
class distMTTKRP {
public:

  typedef typename distsys_t::factormatrix_t factormatrix_t;
  typedef typename distsys_t::map_t map_t;

  distMTTKRP(pt::mode_t mode, distsys_t &distSys) :
             result(NULL), resultMap(NULL)
  {
    // Build a result factor matrix, testing different layouts
    // For even-numbered modes:
    //   build result factor matrix using same map as ktensor's mode
    //   factormatrix
    // For odd-numbered modes:
    //   build result factor matrix using default Trilinos map
    

    if (mode % 2) {
      const map_t *kMap = distSys.getKtensor()->getFactorMap(mode);
      resultMap = new map_t(kMap->getGlobalNumElements(),
                            kMap->getIndexBase(), kMap->getComm());
  
      result = new factormatrix_t(distSys.getKtensor()->getFactorRank(),
                                  resultMap);
    }
    else {
      result = new factormatrix_t(distSys.getKtensor()->getFactorRank(),
                                  distSys.getKtensor()->getFactorMap(mode));
    }

    
    // perform mttkrp into result
    distSys.mttkrp(mode, result);
  }

  ~distMTTKRP() { 
    delete result; 
    if (resultMap) delete resultMap;
  }

  factormatrix_t *getResult() { return result; }

  inline void print(const std::string &msg) { result->print(msg); }


private:
  
  factormatrix_t *result;
  map_t *resultMap;
};

//////////////////////////////////////////////////////////////////////////////

class ttbMTTKRP {
public:

  typedef typename TTB::FacMatrix factormatrix_t;
  typedef typename TTB::Sptensor sptensor_t;
  typedef typename TTB::Ktensor ktensor_t;
  typedef ttb_indx gno_t;
  typedef ttb_real scalar_t;

  ttbMTTKRP(pt::mode_t mode, sptensor_t &sptensor, ktensor_t &ktensor)
  {
    pt::rank_t rank = ktensor.ncomponents();
    size_t modeSize = ktensor[mode].nRows();

    // build factor matrix of global problem on each proc using serial comm 
    result = new factormatrix_t(modeSize, rank);
    
    // call mttkrp
    mttkrp(sptensor, ktensor, mode, *result);

  }

  ~ttbMTTKRP() { delete result; }

  inline scalar_t getGlobalValue(gno_t gidx, pt::rank_t r) { 
    return result->entry(gidx, r);
  }

  factormatrix_t *getResult() { return result; }

private:
  factormatrix_t *result;
};

//////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
class testMTTKRP 
{
public:
  typedef KtensorGenerator<scalar_t> ktensorGenerator_t;
  typedef SptensorGenerator<scalar_t> sptensorGenerator_t;

  typedef generatedDistKtensor<ktensorGenerator_t> generatedDistKtensor_t;
  typedef generatedDistSptensor<sptensorGenerator_t> generatedDistSptensor_t;

  typedef typename generatedDistKtensor_t::ktensor_t ktensor_t;
  typedef typename generatedDistSptensor_t::sptensor_t sptensor_t;

  typedef generatedTTBKtensor<ktensorGenerator_t> generatedTTBKtensor_t;
  typedef generatedTTBSptensor<sptensorGenerator_t> generatedTTBSptensor_t;

  testMTTKRP(pt::mode_t nModes_, std::vector<size_t> &modeSizes_,
             pt::rank_t rank_, double density_,
             const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
    nModes(nModes_),
    ktensorGenerator(nModes_, modeSizes_, rank_),
    sptensorGenerator(nModes_, modeSizes_, rank_, density_),
    distKtensor(ktensorGenerator, comm_),
    distSptensor(sptensorGenerator, distKtensor.getKtensor(), comm_, verbosity),
    ttbKtensor(ktensorGenerator),
    ttbSptensor(sptensorGenerator, ttbKtensor.getKtensor(), verbosity),
    comm(comm_)
  {
    if (comm->getRank()==0) std::cout << "testMTTKRP constructor" << std::endl;
  }

  int run(const std::string &msg) 
  {
    int ierr = 0;
    int me = comm->getRank();

    // Some basic output
    size_t nnz = distSptensor.getSptensor()->getLocalNumNonZeros();
    size_t gnnz;
    Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, 1, &nnz, &gnnz);

    if (me == 0) {
      std::cout << " Number of local nonzeros in tensor = " << gnnz
                << " (density = " 
                << double(gnnz)/double(sptensorGenerator.getMaxNNZ()) << ")"
                << std::endl;
    }

    // Detailed output, if requested
    if (me == 0) {
      if (verbosity > 2) sptensorGenerator.print(msg);
      if (verbosity > 1) ktensorGenerator.print(msg);
    }
    
    if (verbosity > 2) distSptensor.print(msg);
    if (verbosity > 1) distKtensor.print(msg);

    // Create distSystem with sptensor and ktensor
    typedef typename pt::distSystem<sptensor_t, ktensor_t> distSystem_t;

    distSystem_t distSys(distSptensor.getSptensor(), 
                         distKtensor.getKtensor());

    // Let's begin the test now.
    for (pt::mode_t m = 0; m < nModes; m++) {

      // Create distributed mttkrp
      if (me == 0) 
        std::cout << "COMPUTING DISTRIBUTED FOR MODE " << m << std::endl;

      typedef distMTTKRP<distSystem_t> distMTTKRP_t;
      distMTTKRP_t dist(m, distSys);

      if (verbosity > 1) dist.print(msg);

#ifdef USE_ANALYTIC
      // Create analytic mttkrp and compare results
      if (me == 0) 
        std::cout << "COMPUTING ANALYTIC FOR MODE " << m << std::endl;

      typedef analyticMTTKRP<sptensorGenerator_t, ktensorGenerator_t> 
              analyticMTTKRP_t;
      analyticMTTKRP_t analytic(m, sptensorGenerator, ktensorGenerator);

      if (me == 0 && verbosity > 1) analytic.print(msg);
      
      if (me == 0) 
        std::cout << "RUNNING ANALYTIC COMPARISON FOR MODE " << m << std::endl;

      ierr += 
            compareDistBenchmark<distMTTKRP_t,analyticMTTKRP_t>(dist,analytic);
#endif // USE_ANALYTIC

      // Create ttb mttkrp and compare results
      if (me == 0) std::cout << "COMPUTING TTB FOR MODE " << m << std::endl;

      ttbMTTKRP ttb(m, *(ttbSptensor.getSptensor()),
                       *(ttbKtensor.getKtensor()));

      if (me == 0) 
        std::cout << "RUNNING TTB COMPARISON FOR MODE " << m << std::endl;

      ierr += compareDistBenchmark<distMTTKRP_t, ttbMTTKRP>(dist, ttb);

#ifdef KDD_DO_NOT_SKIP
      if (me == 0) {
        std::cout << "COMPARE TTB TO ANALYTIC FOR MODE " << m << std::endl;
        ierr += compareBenchmark<ttbMTTKRP, analyticMTTKRP_t>(ttb, analytic);
      }
#endif
    }

    return ierr;
  }

private:
  pt::mode_t nModes;

  ktensorGenerator_t ktensorGenerator;
  sptensorGenerator_t sptensorGenerator;

  generatedDistKtensor_t distKtensor;
  generatedDistSptensor_t distSptensor;

  generatedTTBKtensor_t ttbKtensor;
  generatedTTBSptensor_t ttbSptensor;

  const Teuchos::RCP<const Teuchos::Comm<int> > comm;
};

//////////////////////////////////////////////////////////////////////////////

int main(int narg, char *arg[])
{
  // Usual Teuchos::Comm initialization
  Tpetra::ScopeGuard scopeguard(&narg, &arg);
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int me = comm->getRank();

  // Accept command-line parameters
  Teuchos::CommandLineProcessor cmdp (false, false);
  cmdp.setOption("verbosity", &verbosity,
                 "0 = quiet; 1 = print status; 2 = print factor matrices; "
                 "3 = print tensor");
  cmdp.parse(narg, arg);

  // Run the tests; count how many fail.
  int ierr = 0;

  /////////////////////////////////
  {
    double density = 0.005;
    pt::rank_t rank = 1;
    pt::mode_t nModes = 2;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 3;
    modeSizes[1] = 5;

    testMTTKRP<double> test(nModes, modeSizes, rank, density, comm);
    ierr += test.run("zero test: ");
  }

  /////////////////////////////////
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

    testMTTKRP<double> test(nModes, modeSizes, rank, density, comm);
    ierr += test.run("first test: ");
  }

  /////////////////////////////////
  { 
    double density = 0.01;
    pt::rank_t rank = 2;
    pt::mode_t nModes = 3;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 30;
    modeSizes[1] = 40;
    modeSizes[2] = 20;

    testMTTKRP<double> test(nModes, modeSizes, rank, density, comm);
    ierr += test.run("second test: ");
  }

  /////////////////////////////////

  {
    double density = 0.005;
    pt::rank_t rank = 10;
    pt::mode_t nModes = 5;
    std::vector<size_t> modeSizes(nModes);

    modeSizes[0] = 30;
    modeSizes[1] = 5;
    modeSizes[2] = 20;
    modeSizes[3] = 10;
    modeSizes[4] = 50;

    testMTTKRP<double> test(nModes, modeSizes, rank, density, comm);
    ierr += test.run("third test: ");
  }

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
