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
#include "pt_test_ttbFromGenerators.hpp"
#include "pt_system.hpp"

#include <TTB_MixedFormatOps.h>

static int verbosity = 1;

//////////////////////////////////////////////////////////////////////////////
int compare(gram_t &gram, ttbgram_t &ttb)
{
  int ierr = 0;
  const double eps = 1e-8;

  typedef typename factormatrix_t::gno_t gno_t;
  typedef typename factormatrix_t::scalar_t scalar_t;

  size_t len = gram.getRank();

  if (len != ttb.getRank()) {
  }

  if (len != ttb.getSize()) {
  }

  size_t len = distResult->getLocalLength();
  typename factormatrix_t::valueview_t data = gram->getView();

  // Loop over result values; compare to benchmark solution
  for (size_t i = 0; i < len; i++) {
    for (size_t j = 0; j < len; j++) {

      scalar_t aValue = ttb.entry(i,j);
      scalar_t dValue = data(i,j);

      if (std::abs<scalar_t>(aValue - dValue) > eps) {
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
  const double eps = 1e-8;

  pt::rank_t rank = benchmark.getResult()->getFactorRank();
  size_t len = benchmark.getResult()->getLocalLength();

  // Loop over result values; compare to benchmark solution
  for (size_t i = 0; i < len; i++) {
    for (pt::rank_t r = 0; r < rank; r++) {

      double aValue = benchmark.getGlobalValue(i, r);
      double dValue = exp.getGlobalValue(i, r);

      if (std::abs<double>(aValue - dValue) > eps) {
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
class testGram 
{
public:
  typedef KtensorGenerator<scalar_t> ktensorGenerator_t;

  typedef generatedTTBKtensor<ktensorGenerator_t> generatedTTBKtensor_t;
  typedef typename TTB:FacMatrix ttbgram_t;

  typedef generatedDistKtensor<ktensorGenerator_t> generatedDistKtensor_t;
  typedef typename generatedDistKtensor_t::ktensor_t ktensor_t;
  typedef typename ktensor_t::factormatrix_t factormatrix_t;
  typedef typename pt::gramianMatrix<factormatrix_t> gram_t;

  testGram(pt::mode_t nModes_, std::vector<size_t> &modeSizes_,
           pt::rank_t rank_,
           const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
    nModes(nModes_),
    ktensorGenerator(nModes_, modeSizes_, rank_),
    distKtensor(ktensorGenerator, comm_),
    ttbKtensor(ktensorGenerator),
    comm(comm_)
  {
    if (comm->getRank()==0) std::cout << "testGram constructor" << std::endl;
  }

  int run(const std::string &msg) 
  {
    int ierr = 0;
    int me = comm->getRank();

    // Detailed output, if requested
    if (me == 0) {
      if (verbosity > 2) ktensorGenerator.print(msg);
    }
    
    if (verbosity > 2) distKtensor.print(msg);

    // Let's begin the test now.
    for (pt::mode_t m = 0; m < nModes; m++) {

      if (me == 0) std::cout << "PT GRAMIAN FOR MODE " << m << std::endl;
      pt::gramianMatrix gram(distKtensor.getKtensor()->getFactorMatrix(m));

      if (verbosity > 1) gram.print("Gramian");

      // Create ttb gramian and compare results
      if (me == 0) std::cout << "TTB GRAMIAN FOR MODE " << m << std::endl;

      TTB::FacMatrix fac;
      fac.gramian(ttbKtensor.getKtensor()[m]); // TTB computes gramian in-place

      if (me == 0) 
        std::cout << "COMPARING FOR MODE " << m << std::endl;

      ierr += compare(gram, fac);
    }

    return ierr;
  }

private:
  pt::mode_t nModes;
  ktensorGenerator_t ktensorGenerator;
  generatedDistKtensor_t distKtensor;
  generatedTTBKtensor_t ttbKtensor;
  const Teuchos::RCP<const Teuchos::Comm<int> > comm;
};

//////////////////////////////////////////////////////////////////////////////

int main(int narg, char *arg[])
{
  // Usual Teuchos::Comm initialization
  Teuchos::GlobalMPISession mpiSession(&narg, &arg, NULL);
  Teuchos::RCP<const Teuchos::Comm<int> > comm =
    Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  int me = comm->getRank();

  Kokkos::initialize(narg, arg);

  // Accept command-line parameters
  Teuchos::CommandLineProcessor cmdp (false, false);
  cmdp.setOption("verbosity", &verbosity,
                 "0 = quiet; 1 = print status; 2 = print gramian matrix; "
                 "3 = print factor matrices");
  cmdp.parse(narg, arg);

  // Run the tests; count how many fail.
  int ierr = 0;

  /////////////////////////////////
  {
    pt::rank_t rank = 10;
    pt::mode_t nModes = 5;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 3;
    modeSizes[1] = 5;
    modeSizes[2] = 2;
    modeSizes[3] = 2;
    modeSizes[4] = 5;

    testGram<double> test(nModes, modeSizes, rank, comm);
    ierr += test.run("first test: ");
  }

  /////////////////////////////////
  { 
    pt::rank_t rank = 2;
    pt::mode_t nModes = 3;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 30;
    modeSizes[1] = 40;
    modeSizes[2] = 20;

    testGram<double> test(nModes, modeSizes, rank, comm);
    ierr += test.run("second test: ");
  }

  /////////////////////////////////

  {
    pt::rank_t rank = 10;
    pt::mode_t nModes = 5;
    std::vector<size_t> modeSizes(nModes);

    modeSizes[0] = 30;
    modeSizes[1] = 5;
    modeSizes[2] = 20;
    modeSizes[3] = 10;
    modeSizes[4] = 50;

    testGram<double> test(nModes, modeSizes, rank, comm);
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

  Kokkos::finalize();
  return gierr;
}
