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
#include "pt_test_compare.hpp"
#include "pt_system.hpp"
#include "Tpetra_Core.hpp"

#include <TTB_Array.h>
#include <TTB_MixedFormatOps.h>

static int verbosity = 1;

//////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
class testInnerprod 
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

  testInnerprod(pt::mode_t nModes_, std::vector<size_t> &modeSizes_,
             pt::rank_t rank_, double density_,
             const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
    nModes(nModes_),
    rank(rank_),
    ktensorGenerator(nModes_, modeSizes_, rank_),
    sptensorGenerator(nModes_, modeSizes_, rank_, density_),
    distKtensor(ktensorGenerator, comm_),
    distSptensor(sptensorGenerator, distKtensor.getKtensor(), comm_, verbosity),
    ttbKtensor(ktensorGenerator),
    ttbSptensor(sptensorGenerator, ttbKtensor.getKtensor(), verbosity),
    comm(comm_)
  {
  }

  int run(const std::string &msg) 
  {
    int ierr = 0;
    int me = comm->getRank();
    const scalar_t tolerance = 20.*std::numeric_limits<scalar_t>::epsilon();

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
    // Do not do the communication at construction; let it happen with
    // the first inner product.
    typedef typename pt::distSystem<sptensor_t, ktensor_t> distSystem_t;

    distSystem_t distSys(distSptensor.getSptensor(), 
                         distKtensor.getKtensor(),
                         distSystem_t::UPDATE_NONE);


    // Sanity check the input sparse tensors
    scalar_t distSpNorm = distSptensor.getSptensor()->frobeniusNorm();
    scalar_t ttbSpNorm = ttbSptensor.getSptensor()->norm();
    if (!pt::nearlyEqual(distSpNorm, ttbSpNorm, 5e-11)) {
      std::cout << msg << " Error:  spTensor norms differ:  "
                << distSpNorm << " != " << ttbSpNorm << std::endl;
      ierr++;
    }

    // Sanity check the input ktensors
    for (pt::mode_t m = 0; m < nModes; m++) {
      Kokkos::View<scalar_t *> distNorm("distNorm", rank);
      distKtensor.getKtensor()->getFactorMatrix(m)->norm2(distNorm);
      TTB::Array ttbNorm(rank, 0.);
      (*(ttbKtensor.getKtensor()))[m].colNorms(TTB::NormTwo, ttbNorm);

      for (pt::rank_t r = 0; r < rank; r++) {
        if (!pt::nearlyEqual(distNorm(r), ttbNorm[r], 1e-15)) {
          std::cout << msg << " Error in ktensor norms " << m << " " << r 
                    << ":  " << distNorm(r) << " != " << ttbNorm[r]
                    << std::endl;
          ierr++;
        }
      }
    }

    // Compare innerprod using pt and TTB with scaleValues = 1.
    Kokkos::View<scalar_t *> distScaleValues("scaleValues", rank);
    Kokkos::deep_copy(distScaleValues, 1.);

    scalar_t distResult = distSys.innerprod(distScaleValues);

    TTB::Array ttbScaleValues(rank, 1.);
    scalar_t ttbResult = TTB::innerprod(*(ttbSptensor.getSptensor()),
                                        *(ttbKtensor.getKtensor()),
                                        ttbScaleValues);

    if (comm->getRank() == 0)
      std::cout << msg << " Innerprod with scaleValues = 1.:  "
                << distResult << "  " << ttbResult << std::endl;
               

    if (!pt::nearlyEqual(distResult, ttbResult, tolerance)) {
      std::cout << msg << " Error with scaleValues = 1.; " 
                << distResult << " != " << ttbResult << std::endl;
      ierr++;
    }

    // Compare innerprod using pt and TTB with scaleValues = -r for each rank r
    for (pt::rank_t r = 0; r < rank; r++) {
      scalar_t scaleValue = -r;
      distScaleValues(r) = scaleValue;
      ttbScaleValues[r] = scaleValue;
    }

    distResult = distSys.innerprod(distScaleValues);

    ttbResult = TTB::innerprod(*(ttbSptensor.getSptensor()),
                               *(ttbKtensor.getKtensor()),
                               ttbScaleValues);
    if (comm->getRank() == 0)
      std::cout << msg << " Innerprod with scaleValues = -r.:  "
                << distResult << "  " << ttbResult << std::endl;
               

    if (!pt::nearlyEqual(distResult, ttbResult, tolerance)) {
      std::cout << msg << " Error with scaleValues = -r; " 
                << distResult << " != " << ttbResult << std::endl;
      ierr++;
    }

    return ierr;
  }

private:
  pt::mode_t nModes;
  pt::rank_t rank;

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
    pt::rank_t rank = 10;
    pt::mode_t nModes = 5;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 3;
    modeSizes[1] = 5;
    modeSizes[2] = 2;
    modeSizes[3] = 2;
    modeSizes[4] = 5;

    testInnerprod<double> test(nModes, modeSizes, rank, density, comm);
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

    testInnerprod<double> test(nModes, modeSizes, rank, density, comm);
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

    testInnerprod<double> test(nModes, modeSizes, rank, density, comm);
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
