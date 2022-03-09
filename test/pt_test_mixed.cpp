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
 
// Test mixed-format operations


#include "pt_mixed.hpp"
#include "pt_test_distFromGenerators.hpp"
#include "pt_test_ttbFromGenerators.hpp"
#include "pt_test_compare.hpp"
#include "Tpetra_Core.hpp"

#include <TTB_FacMatrix.h>

static int verbosity = 1;

//////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
class testMixedFunctions 
{
public:
  typedef KtensorGenerator<scalar_t> ktensorGenerator_t;
  typedef generatedDistKtensor<ktensorGenerator_t> generatedDistKtensor_t;
  typedef generatedTTBKtensor<ktensorGenerator_t> generatedTTBKtensor_t;

  typedef typename generatedDistKtensor_t::ktensor_t ktensor_t;
  typedef typename ktensor_t::factormatrix_t factormatrix_t;
  typedef typename factormatrix_t::valueview_t valueview_t;
  typedef typename pt::squareLocalMatrix<scalar_t> slm_t;

  testMixedFunctions(pt::mode_t nModes_, std::vector<size_t> &modeSizes_,
                        pt::rank_t rank_,
                        const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
    nModes(nModes_),
    rank(rank_),
    ktensorGenerator(nModes_, modeSizes_, rank_),
    distKtensor(ktensorGenerator, comm_, false),
    ttbKtensor(ktensorGenerator, false),
    comm(comm_)
  {
    if (comm->getRank()==0) 
      std::cout << "testMixedFunctions constructor:  nModes=" << nModes 
                << " rank=" << rank << std::endl;
  }

  int run(const std::string &msg);

private:
  pt::mode_t nModes;
  pt::rank_t rank;

  ktensorGenerator_t ktensorGenerator;
  generatedDistKtensor_t distKtensor;
  generatedTTBKtensor_t ttbKtensor;

  const Teuchos::RCP<const Teuchos::Comm<int> > comm;

};

//////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
int testMixedFunctions<scalar_t>::run(const std::string &msg) 
{
  int ierr = 0;
  int me = comm->getRank();

  // Some basic output

  // Detailed output, if requested
  if (me == 0) {
    if (verbosity > 1) ktensorGenerator.print(msg);
  }
  if (verbosity > 1) distKtensor.print(msg);

  // Format the output of scalar_t
  std::cout.precision(22);
  std::cout << std::fixed;

  // Let's begin the test now.
  for (pt::mode_t m = 0; m < nModes; m++) {

    // Create a square local matrix with identity * -1
    slm_t testmatrix(rank);
    for (pt::rank_t r = 0; r < rank; r++) testmatrix(r, r) = -1.;

    // Copy the factor matrix, as solveTransposeRHS overwrites it.
    factormatrix_t *ptfm = distKtensor.getKtensor()->getFactorMatrix(m);
    factormatrix_t copyptfm(*ptfm);

    pt::solveTransposeRHS(testmatrix, copyptfm);

    for (size_t i = 0; i < ptfm->getLocalLength(); i++) {
      for (pt::rank_t r = 0; r < rank; r++) {
        if (!pt::nearlyEqual<scalar_t>(copyptfm(i, r), -1.*(*ptfm)(i, r))) {
          std::cout << "Error with solveTransposeRHS using testmatrix = I * -1 ("
                    << i << "," << r << "): "
                    << copyptfm(i, r) << " != " << (*ptfm)(i, r) << std::endl;
          ierr++;
        }
      }
    }

    // Make sure testmatrix did not change
    for (pt::rank_t r = 0; r < rank; r++) {
      for (pt::rank_t rr = 0; rr < rank; rr++) {
        if (r != rr && testmatrix(r, rr) != 0.) {
          std::cout << "Error with solveTransposeRHS: testmatrix changed (" << r << "," << rr << "): "
                    << testmatrix(r, rr) << " != 0." << std::endl;
          ierr++;
        }
        else if (r == rr && testmatrix(r, rr) != -1.) {
          std::cout << "Error with solveTransposeRHS: testmatrix changed (" << r << "," << rr << "): "
                    << testmatrix(r, rr) << " != 1." << std::endl;
          ierr++;
        }
      }
    }
    
    // Now test sqSymmRectMatmul
    
    // Compute copyptfm = ptfm * (-I) - copyptfm = -ptfm
    copyptfm.setValues(0);
    pt::symmRectMatmul(testmatrix, *ptfm, copyptfm);
    for (size_t i = 0; i < ptfm->getLocalLength(); i++) {
      for (pt::rank_t r = 0; r < rank; r++) {
        if (!pt::nearlyEqual<scalar_t>(copyptfm(i, r), -(*ptfm)(i, r))) {
          std::cout << "Error with symmRectMatmul using testmatrix = I * -1 ("
                    << i << "," << r << "): "
                    << copyptfm(i, r) << " != " << -(*ptfm)(i, r) << std::endl;
          ierr++;
        }
      }
    }
    
    // Compute copyptfm = ptfm * (-I) - (-ptfm) = 0
    pt::symmRectMatmul(testmatrix, *ptfm, copyptfm);
    for (size_t i = 0; i < ptfm->getLocalLength(); i++) {
      for (pt::rank_t r = 0; r < rank; r++) {
        if (!pt::nearlyEqual<scalar_t>(copyptfm(i, r), 0)) {
          std::cout << "Error with 2nd ssymmRectMatmul using testmatrix = I * -1 ("
                    << i << "," << r << "): "
                    << copyptfm(i, r) << " != 0" << std::endl;
          ierr++;
        }
      }
    }

#if 0  // TODO CHECK RESULT AFTER GET COMPILING!

    // Create ttb gramian 
    TTB::FacMatrix ttb;
    ttb.gramian((*ttbKtensor.getKtensor())[m]);

    for (pt::rank_t r = 0; r < rank; r++) {
      for (pt::rank_t rr = 0; rr < rank; rr++) {

        // Testing overloaded accessor
        if (!pt::nearlyEqual<scalar_t>(ttb.entry(r, rr), ptgram(r, rr),
                             20.*std::numeric_limits<scalar_t>::epsilon())) {
          std::cout << "Error vs ttb (" << r << "," << rr << "): "
                    << ttb.entry(r, rr) << " != " << ptgram(r, rr) << std::endl;
          ierr++;
        }

        // Testing view accessor
        if (ptgram(r, rr) != ptgramview(r, rr)) {
          std::cout << "Error view (" << r << "," << rr << "): "
                    << ptgram(r, rr) << " != " << ptgramview(r, rr)
                    << std::endl;
          ierr++;
        }
      }
    }
#endif
  }

  return ierr;
}

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
    pt::rank_t rank = 10;
    pt::mode_t nModes = 5;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 3;
    modeSizes[1] = 5;
    modeSizes[2] = 2;
    modeSizes[3] = 2;
    modeSizes[4] = 5;

    testMixedFunctions<double> test(nModes, modeSizes, rank, comm);
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

    testMixedFunctions<double> test(nModes, modeSizes, rank, comm);
    ierr += test.run("second test: ");
  }

  /////////////////////////////////

  {
    pt::rank_t rank = 100;
    pt::mode_t nModes = 5;
    std::vector<size_t> modeSizes(nModes);

    modeSizes[0] = 30;
    modeSizes[1] = 5;
    modeSizes[2] = 20;
    modeSizes[3] = 10;
    modeSizes[4] = 50;

    testMixedFunctions<double> test(nModes, modeSizes, rank, comm);
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
