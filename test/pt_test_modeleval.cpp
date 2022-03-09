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
 
// Unit test for distSystem class
// Does not test mttkrp; see pt_test_mttkrp.cpp for mttkrp tests.

#include "pt_system.hpp"
#include "pt_test_compare.hpp"
#include "Tpetra_Core.hpp"

template <typename scalar_t>
class testModelEval {

public:

  typedef typename pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename pt::distKtensor<factormatrix_t> ktensor_t;
  typedef typename pt::distSptensor<scalar_t> sptensor_t;
  typedef typename pt::distSystem<sptensor_t, ktensor_t> distsys_t;

  typedef typename factormatrix_t::map_t map_t;
  typedef typename factormatrix_t::valueview_t valueview_t;
  typedef typename factormatrix_t::gno_t gno_t;

  // Constructor:  initializes values
  testModelEval(const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
              std::vector<size_t> &modeSizes_,
              pt::rank_t rank_,
              typename sptensor_t::gnoview_t &globalIndices_,
              typename sptensor_t::valueview_t &values_
  ):
    comm(comm_),
    me(comm->getRank()),
    np(comm->getSize()),
    modeSizes(modeSizes_),
    nModes(modeSizes_.size()),
    rank(rank_),
    globalIndices(globalIndices_),
    values(values_)
  { }

  // Destructor
  ~testModelEval() { }

  // How to run the tests within testModelEval
  int run();


private:
  const Teuchos::RCP<const Teuchos::Comm<int> > comm;
  const int me;
  const int np;

  std::vector<size_t> modeSizes;
  const pt::mode_t nModes;
  const pt::rank_t rank;

  typename sptensor_t::gnoview_t globalIndices;
  typename sptensor_t::valueview_t values;

  inline scalar_t testVal(gno_t gid, pt::rank_t r) 
  { 
    return scalar_t(gid*(r+1)); 
  }
}; 

////////////////////////////////////////////////////////////////////////////////
// How to run the tests within testModelEval
template <typename scalar_t>
int testModelEval<scalar_t>::run()
{
  int ierr = 0;

  std::vector<const map_t *> ktensorMaps(nModes);

  // Build sptensor
  sptensor_t sptensor(nModes, modeSizes, globalIndices, values, comm);

  ///////////////////////////////////////////////////////////
  // Build a ktensor that uses default Tpetra::Maps
  for (pt::mode_t m = 0; m < nModes; m++)
    ktensorMaps[m] = new map_t(modeSizes[m], 0, comm);
   
  ktensor_t ktensor(rank, ktensorMaps, comm);

  // Set ktensor values to something predictable so that we can 
  // analytically compute the model value
  for (pt::mode_t m = 0; m < nModes; m++) {
    typename ktensor_t::factormatrix_t *fm = ktensor.getFactorMatrix(m);
    size_t len = fm->getLocalLength();
    typename ktensor_t::factormatrix_t::valueview_t data = fm->getLocalView();
    for (size_t i = 0; i < len; i++)
      for (pt::rank_t r = 0; r < rank; r++) 
        data(i, r) = testVal(fm->getMap()->getGlobalElement(i), r);
  }
  const scalar_t LAMBDA = 0.5;
  ktensor.setLambda(LAMBDA);

  // Build a system
  pt::distSystem<sptensor_t, ktensor_t> distsys(&sptensor, &ktensor);

  // For each sptensor index, compute the model value analytically and
  // using computeModelAtIndex(); compare
  for (size_t n = 0; n < sptensor.getLocalNumIndices(); n++) {

    // Compute the "exact" solution based on testVal
    Kokkos::View<scalar_t*> prod("prod", rank);
    Kokkos::deep_copy(prod, 1.);
    for (pt::mode_t m = 0; m < nModes; m++) {
      gno_t gid = sptensor.getGlobalIndices()(n, m);
      for (pt::rank_t r = 0; r < rank; r++) prod[r] *= testVal(gid, r);
    }
    scalar_t exactval = 0.;
    for (pt::rank_t r = 0; r < rank; r++) exactval += prod[r];
    exactval *= LAMBDA;

    // Compute the model using the first function prototype
    scalar_t modelval = distsys.computeModelAtIndex(n);
    if (!pt::nearlyEqual(modelval, exactval)) {
      std::cout << "Error method 1:  modelval " << modelval << " != " 
                << exactval << " exactval for global index (";
      for (pt::mode_t m = 0; m < nModes; m++) 
        std::cout << sptensor.getGlobalIndices()(n,m) << " ";
      std::cout << ")" << std::endl;
      ierr++;
    }

    // Compute the model using the second function prototype
    modelval = distsys.computeModelAtIndex(n, prod);
    if (!pt::nearlyEqual(modelval, exactval)) {
      std::cout << "Error method 2:  modelval " << modelval << " != " 
                << exactval << " exactval for global index (";
      for (pt::mode_t m = 0; m < nModes; m++) 
        std::cout << sptensor.getGlobalIndices()(n,m) << " ";
      std::cout << ")" << std::endl;
      ierr++;
    }
  }

  // clean up 
  for (pt::mode_t m = 0; m < nModes; m++) delete ktensorMaps[m];

  return ierr;
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

int main(int narg, char **arg)
{
  // Usual Teuchos::Comm initialization
  Tpetra::ScopeGuard scopeguard(&narg, &arg);
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int me = comm->getRank();
  int np = comm->getSize();

  int ierr = 0;
  pt::mode_t nModes;
  pt::rank_t rank;
  size_t nnz;

  { // A generated tensor with 10 nonzeros per processor
    nnz = 10;

    nModes = 4;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = np;
    modeSizes[1] = 2 * (np - 1) + (nnz - 1) + 1;
    modeSizes[2] = 2 * (np - 1) + 2 * (nnz - 1) + 1;
    modeSizes[3] = nnz;
   
    typedef float scalar_t;
    typedef typename testModelEval<scalar_t>::gno_t gno_t;
    Kokkos::View<gno_t **> indices("indices", nnz, nModes);
    Kokkos::View<scalar_t *> values("values", nnz);
    
    for (size_t nz = 0; nz < nnz; nz++) {
      indices(nz, 0) = me;
      indices(nz, 1) = 2 * me + nz;
      indices(nz, 2) = 2 * (me + nz);  // only even indices used
      indices(nz, 3) = nz;
      values(nz) = 1.;
    }

    rank = 4;

    if (me == 0) {
      std::cout << std::endl
                << "TESTING WITH <FLOAT> AND RANK=" 
                << rank << ", NMODES=" << nModes << " (";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
      std::cout << ")" << std::endl;
    }

    testModelEval<scalar_t> test(comm, modeSizes, rank, indices, values);
    ierr += test.run();
  }

  { // A generated tensor with block-based random input

    nnz = 1000;
    nModes = 3;
    std::vector<size_t> modeSizes(nModes);
    const size_t mult[3] = {100, 200, 300};
    modeSizes[0] = mult[0] * np;
    modeSizes[1] = mult[1] * np;
    modeSizes[2] = mult[2] * np;

    typedef double scalar_t;
    typedef typename testModelEval<scalar_t>::gno_t gno_t;
    Kokkos::View<gno_t **> indices("indices", nnz, nModes);
    Kokkos::View<scalar_t *> values("values", nnz);

    srand(me);

    for (size_t nz = 0; nz < nnz; nz++) {
      for (pt::mode_t m = 0; m < nModes; m++)
        indices(nz, m) = mult[m] * me + rand() % mult[m];
      values(nz) = 1.;
    }

    rank = 10;

    if (me == 0) {
      std::cout << std::endl
                << "TESTING WITH <DOUBLE> AND RANK=" 
                << rank << ", NMODES=" << nModes << " (";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
      std::cout << ")" << std::endl;
    }

    testModelEval<scalar_t> test(comm, modeSizes, rank, indices, values);
    ierr += test.run();
  }

  if (ierr) std::cout << me << ": " << ierr << " errors detected." << std::endl;

  int gierr = 0;
  Teuchos::reduceAll<int,int>(*comm, Teuchos::REDUCE_SUM, 1, &ierr, &gierr);

  if (me == 0) {
    if (gierr == 0)
      std::cout << " PASS" << std::endl;
    else
      std::cout << " FAIL" << std::endl;
  }

  return gierr;
}
