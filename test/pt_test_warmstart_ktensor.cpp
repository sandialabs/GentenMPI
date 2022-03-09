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
 
// Unit test for distKtensor class

#include "pt_ktensor.hpp"
#include "pt_test_compare.hpp"
#include "Tpetra_Core.hpp"

template <typename scalar_t>
class testKtensor {

public:

  typedef typename pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename pt::distKtensor<factormatrix_t> ktensor_t;
  typedef typename factormatrix_t::map_t map_t;
  typedef typename factormatrix_t::valueview_t valueview_t;
  typedef typename factormatrix_t::gno_t gno_t;
  typedef typename factormatrix_t::lno_t lno_t;

  // Constructor:  initializes values
  testKtensor(const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
              std::vector<size_t> &modeSizes_,
              pt::rank_t rank_):
    comm(comm_),
    me(comm->getRank()),
    np(comm->getSize()),
    modeSizes(modeSizes_),
    nModes(modeSizes_.size()),
    rank(rank_),
    ktensorFromSizes(NULL),
    ktensorFromDefaultMaps(NULL),
    ktensorFromCyclicMaps(NULL)
  { }


  // Destructor:  frees allocated pointers
  ~testKtensor() 
  {
    if (ktensorFromSizes != NULL) delete ktensorFromSizes;
    if (ktensorFromDefaultMaps != NULL) delete ktensorFromDefaultMaps;
    if (ktensorFromCyclicMaps != NULL) delete ktensorFromCyclicMaps;
  }

  // A method for testing basic properties of a ktensor
  int checkKtensor(const std::string &msg, ktensor_t *ktensor,
                   std::vector<const map_t *> &maps);

  // How to run the tests within testKtensor
  int run();

  // Build a cyclic map (like dealing cards)
  map_t *buildCyclicMap(size_t nGlobal)
  {
    // Compute number of local entries
    size_t nMyGids = nGlobal / comm->getSize();
    nMyGids += (size_t(me) < (nGlobal % comm->getSize()) ? 1 : 0);

    // Build list of local entries
    Kokkos::View<gno_t *> myGids("local GIDs", nMyGids);
    gno_t myFirstGid = me;
    for (size_t i = 0; i < nMyGids; i++)
      myGids(i) = myFirstGid + (i * np);

    // Return the cyclic map
    const Tpetra::global_size_t inv =
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    return new map_t(inv, myGids, 0, comm);
  }

  scalar_t buildValueFromLocalId(const map_t *map, lno_t lid, pt::rank_t r)
  {
    return scalar_t((map->getGlobalElement(lid)+1)*1000 + r);
  }

private:
  const Teuchos::RCP<const Teuchos::Comm<int> > comm;
  const int me;
  const int np;

  std::vector<size_t> modeSizes;
  const pt::mode_t nModes;
  const pt::rank_t rank;


  ktensor_t *ktensorFromSizes;
  ktensor_t *ktensorFromDefaultMaps;
  ktensor_t *ktensorFromCyclicMaps;
}; 


////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
int testKtensor<scalar_t>::checkKtensor(
  const std::string &msg,
  ktensor_t *ktensor,
  std::vector<const map_t *> &maps
)
{
  int ierr = 0;
  if (me == 0) std::cout << "checkKtensor with " << msg << std::endl;

  // Check sizes
  if (ktensor->getNumModes() != nModes) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in nModes "
              << ktensor->getNumModes() << " != "
              << nModes << std::endl;
  }

  if (ktensor->getFactorRank() != rank) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in ranks "
              << ktensor->getFactorRank() << " != " << rank << std::endl;
  }

  for (pt::mode_t m = 0; m < nModes; m++) {
    if (ktensor->getModeSize(m) != modeSizes[m]) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in modesize " << m << " "
                << ktensor->getModeSize(m) << " != " << modeSizes[m]
                << std::endl;
    }
  }

  // For initial tensor, lambda should be all ones
  Kokkos::View<scalar_t *> lambda = ktensor->getLambdaView();
  for (pt::rank_t r = 0; r < rank; r++) {
    if (lambda(r) != 1.) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in initial lambda " 
                << r << ":  " << lambda(r) << " != " << 1.
                << std::endl;
    }
  }

  // Make sure we can do something with the factor matrices
  for (pt::mode_t m = 0; m < nModes; m++) {

    factormatrix_t *factor = ktensor->getFactorMatrix(m);

    if (factor == NULL) {
      ierr++;
      std::cout << me << " " << msg << ":  Error -- NULL factor matrix "
                << m << std::endl;
      continue;
    }

    // Check Factor Matrix sizes
    size_t tmp = factor->getLocalLength();
    size_t gtmp;
    Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, 1, &tmp, &gtmp);
    if (gtmp != modeSizes[m]) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in getLocalLength " << m << " "
                << factor->getLocalLength() << " != " << modeSizes[m] 
                << std::endl;
    }

    if (factor->getFactorRank() != rank) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in getFactorRank " << m << " "
                << factor->getFactorRank() << " != " << rank
                << std::endl;
    }

    // Check Factor Matrix values
    {
      valueview_t data = factor->getLocalView();
      auto map = factor->getMap();
      for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
        for (size_t j = 0; j < factor->getLocalLength(); j++) {
          scalar_t expected = buildValueFromLocalId(map,j,r);
          if (data(j, r) != expected) {
            ierr++;
            std::cout << me << " " << msg << ":  Error in setValues "
                      << data(j, r) << " != " << expected
                      << std::endl;
          }
        }
      }
    }

    // Check that can put data into factor matrix
    {
      factor->setValues(m);

      valueview_t data = factor->getLocalView();
      for (size_t j = 0; j < factor->getLocalLength(); j++) {
        for (pt::rank_t r = 0; r < rank; r++) {
          if (data(j, r) != m) {
            ierr++;
            std::cout << me << " " << msg << ":  Error in data " << m << " "
                      << data(j, r) << " != " << m
                      << std::endl;
          }
        }
      }
    }
  }


  // Test the normalization routines, testing factormatrix values and lambda
  // Reset the ktensor to initial values; check NORM_ONE
  {
    ktensor->setLambda(1.);
    for (pt::mode_t m = 0; m < nModes; m++) {
      factormatrix_t *factor = ktensor->getFactorMatrix(m);
      factor->setValues(m);
    }

    ktensor->normalize(pt::NORM_ONE);
    scalar_t prod = 1.;
    for (pt::mode_t m = 0; m < nModes; m++) {
      // earlier, we stored scalar m in factor matrix m, so computing the
      // expected value is easy
      scalar_t norm = m*modeSizes[m];
      scalar_t expected = (m > 0 ? scalar_t(m) / norm : 0.);
      prod *= norm;
  
      typename factormatrix_t::valueview_t data;
      data = ktensor->getFactorMatrix(m)->getLocalView();
      size_t len = ktensor->getFactorMatrix(m)->getLocalLength();
  
      for (size_t j = 0; j < len; j++) {
        for (pt::rank_t r = 0; r < rank; r++) {
          if (!pt::nearlyEqual(data(j, r), expected)) {
            ierr++;
            std::cout << me << " " << msg << ":  Error in NORM_ONE data(" 
                      << j << "," << r << ") mode "
                      << m << ":  " << data(j, r) << " != " << expected
                      << std::endl;
          }
        }
      }
    }
  
    lambda = ktensor->getLambdaView();
    for (pt::rank_t r = 0; r < rank; r++) {
      if (lambda(r) != prod) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in NORM_ONE lambda " 
                  << r << ":  " << lambda(r) << " != " << prod
                  << std::endl;
      }
    }
  }

  // Test the normalization routines, testing factormatrix values and lambda
  // Reset the ktensor to initial values; check NORM_INF
  {
    ktensor->setLambda(1.);
    for (pt::mode_t m = 0; m < nModes; m++) {
      factormatrix_t *factor = ktensor->getFactorMatrix(m);
      factor->setValues(m);
    }

    ktensor->normalize(pt::NORM_INF);
    scalar_t prod = 1.;
    for (pt::mode_t m = 0; m < nModes; m++) {
      // earlier, we stored scalar m in factor matrix m, so computing the
      // expected value is easy
      scalar_t norm = m;
      scalar_t expected = (m > 0 ? scalar_t(m) / norm : 0.);
      prod *= norm;
  
      typename factormatrix_t::valueview_t data;
      data = ktensor->getFactorMatrix(m)->getLocalView();
      size_t len = ktensor->getFactorMatrix(m)->getLocalLength();
  
      for (size_t j = 0; j < len; j++) {
        for (pt::rank_t r = 0; r < rank; r++) {
          if (data(j, r) != expected) {
            ierr++;
            std::cout << me << " " << msg << ":  Error in NORM_INF data(" 
                      << j << "," << r << ") mode "
                      << m << ":  " << data(j, r) << " != " << expected
                      << std::endl;
          }
        }
      }
    }
  
    lambda = ktensor->getLambdaView();
    for (pt::rank_t r = 0; r < rank; r++) {
      if (lambda(r) != prod) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in NORM_INF lambda " 
                  << r << ":  " << lambda(r) << " != " << prod
                  << std::endl;
      }
    }
  }

  // Test the normalization routines, testing factormatrix values and lambda
  // Default is NORM_TWO
  {
    for (pt::mode_t m = 0; m < nModes; m++) {
      factormatrix_t *factor = ktensor->getFactorMatrix(m);
      factor->setValues(m);
    }
    ktensor->setLambda(1.);

    ktensor->normalize();
    scalar_t prod = 1.;
    for (pt::mode_t m = 0; m < nModes; m++) {
      // earlier, we stored scalar m in factor matrix m, so computing the
      // expected value is easy
      scalar_t norm = std::sqrt(scalar_t(m*m*modeSizes[m]));
      scalar_t expected = (m > 0 ? scalar_t(m) / norm : 0.);
      prod *= norm;
  
      typename factormatrix_t::valueview_t data;
      data = ktensor->getFactorMatrix(m)->getLocalView();
      size_t len = ktensor->getFactorMatrix(m)->getLocalLength();
  
      for (size_t j = 0; j < len; j++) {
        for (pt::rank_t r = 0; r < rank; r++) {
          if (data(j, r) != expected) {
            ierr++;
            std::cout << me << " " << msg << ":  Error in NORM_TWO data(" 
                      << j << "," << r << ") mode "
                      << m << ":  " << data(j, r) << " != " << expected
                      << std::endl;
          }
        }
      }
    }
  
    lambda = ktensor->getLambdaView();
    for (pt::rank_t r = 0; r < rank; r++) {
      if (lambda(r) != prod) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in NORM_TWO lambda " 
                  << r << ":  " << lambda(r) << " != " << prod
                  << std::endl;
      }
    }
  }

  // test setLambda(scalar)
  ktensor->setLambda(2.);

  lambda = ktensor->getLambdaView();
  for (pt::rank_t r = 0; r < rank; r++) {
    if (lambda(r) != 2.) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in setLambda(scalar) " 
                << r << ":  " << lambda(r) << " != " << 2
                << std::endl;
    }
  }

  // test setLambda(view)
  Kokkos::View<scalar_t *> newlambda("newlambda", rank);
  for (pt::rank_t r = 0; r < rank; r++) newlambda(r) = r+2;
  ktensor->setLambda(newlambda);

  lambda = ktensor->getLambdaView();
  for (pt::rank_t r = 0; r < rank; r++) {
    if (lambda(r) != r+2) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in setLambda(view) " 
                << r << ":  " << lambda(r) << " != " << r+2
                << std::endl;
    }
  }

  // test setLambdaView
  for (pt::rank_t r = 0; r < rank; r++) newlambda(r) = r+1;
  ktensor->setLambdaView(newlambda);

  lambda = ktensor->getLambdaView();
  for (pt::rank_t r = 0; r < rank; r++) {
    if (lambda(r) != (r+1)) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in setLambdaView " 
                << r << ":  " << lambda(r) << " != " << r+1
                << std::endl;
    }
  }

  // Retest normalization -- mode-by-mode this time; results should be same
  // as normalizing entire ktensor at once (but with lambda(r) = r+1 initially)
  {
    for (pt::mode_t m = 0; m < nModes; m++) {
      factormatrix_t *factor = ktensor->getFactorMatrix(m);
      factor->setValues(m);
    }

    scalar_t prod = 1.;
    for (pt::mode_t m = 0; m < nModes; m++) {
  
      ktensor->normalize(m);
  
      scalar_t norm = std::sqrt(scalar_t(m*m*modeSizes[m]));
      scalar_t expected = (m > 0 ? scalar_t(m) / norm : 0.);
      prod *= norm;
  
      factormatrix_t *factor = ktensor->getFactorMatrix(m);
      typename factormatrix_t::valueview_t data = factor->getLocalView();
  
      for (size_t j = 0; j < factor->getLocalLength(); j++) {
        for (pt::rank_t r = 0; r < rank; r++) {
          if (data(j, r) != expected) {
            ierr++;
            std::cout << me << " " << msg << ":  Error in normalized(m) data(" 
                      << j << "," << r << ") mode "
                      << m << ":  " << data(j, r) << " != " << expected
                      << std::endl;
          }
        }
      }
    }
  
    lambda = ktensor->getLambdaView();
    for (pt::rank_t r = 0; r < rank; r++) {
      if (lambda(r) != prod*(r+1)) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in normalized(m) lambda " 
                  << r << ":  " << lambda(r) << " != " << prod*(r+1)
                  << std::endl;
      }
    }
  }

  // Check distributeLambda function with simple initial state
  {
    pt::mode_t testmode = nModes - 1;
    factormatrix_t *factor = ktensor->getFactorMatrix(testmode);
    typename factormatrix_t::valueview_t data = factor->getLocalView();

    factor->setValues(nModes);
    ktensor->setLambda(1.);

    ktensor->normalize(testmode);         // Sets lambda to norms of factor
    ktensor->distributeLambda(testmode);  // Resets lambda to one; rescales
                                          // factor to orig values

    for (pt::rank_t r = 0; r < rank; r++) {
      for (size_t j = 0; j < factor->getLocalLength(); j++) {
        if (data(j, r) != nModes) {
          std::cout << me << " " << msg << ":  Error in distributeLambda "
                    << "factor " << j << " " << r << ": " 
                    << data(j, r) << " != " << nModes
                    << std::endl;
          ierr++;
        }
      }
      if (ktensor->getLambdaView()(r) != 1.) {
        std::cout << me << " " << msg << ":  Error in distributeLambda "
                  << "lambda " << r << ": " 
                  << ktensor->getLambdaView()(r) << " != " << 1.
                  << std::endl;
        ierr++;
      }
    }
  }

  // Check setRandomUniform; NORM_ONE should be one for all factor matrices;
  // Sum of lambda values should be one.
  {
    ktensor->setRandomUniform();
    for (pt::mode_t m = 0; m < nModes; m++) {
      Kokkos::View<scalar_t *> weights("weights", rank);
      ktensor->getFactorMatrix(m)->norm1(weights);
      for (pt::rank_t r = 0; r < rank; r++) {
        if (!pt::nearlyEqual(weights(r), scalar_t(1.),
                          20.*std::numeric_limits<scalar_t>::epsilon())) {
          std::cout << me << " " << msg << ":  Error in setRandomUniform "
                    << "data " << m << " " << r << ": " 
                    << weights(r) << " != " << 1.
                    << std::endl;
          ierr++;
        }
      }
    }
    scalar_t sum = 0.;
    for (pt::rank_t r = 0; r < rank; r++) sum += ktensor->getLambdaView()(r);
    if (!pt::nearlyEqual(sum, scalar_t(1.), 
                          20.*std::numeric_limits<scalar_t>::epsilon())) {
      std::cout << me << " " << msg << ":  Error in setRandomUniform "
                << "weight sum " << ": " 
                << sum << " != " << 1.
                << std::endl;
      ierr++;
    }
  }
  
  // Check maps
  for (pt::mode_t m = 0; m < nModes; m++) {
    
    const map_t *factormap = ktensor->getFactorMap(m);

    // Compare factormaps to input and to factor matrix's map
    if (!(factormap->isSameAs(*(maps[m]))) ||
        !(factormap->isSameAs(*(ktensor->getFactorMatrix(m)->getMap())))) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in Maps " << m << " "
                << factormap->isSameAs(*(maps[m])) << " "
                << factormap->isSameAs(*(ktensor->getFactorMatrix(m)->getMap()))
                << std::endl;
    }
  }

  // Test the copy constructor
  ktensor_t *copy = new ktensor_t(ktensor);
  for (mode_t m = 0; m < nModes; m++) {
    valueview_t fm = ktensor->getFactorMatrix(m)->getLocalView();
    valueview_t fmcopy = copy->getFactorMatrix(m)->getLocalView();
    size_t len = ktensor->getFactorMatrix(m)->getLocalLength();
    for (size_t i = 0; i < len; i++) {
      for (pt::rank_t r = 0; r < rank; r++) {
        if (fm(i,r) != fmcopy(i,r)) {
          ierr++;
          std::cout << me << " " << msg << ":  Error in copy constructor " << m
                    << " " << fm(i,r) << " != " << fmcopy(i,r)
                    << std::endl;
        }
      }
    }
  }

  copy->setRandomUniform();
  ktensor->copyData(copy);
  for (mode_t m = 0; m < nModes; m++) {
    valueview_t fm = ktensor->getFactorMatrix(m)->getLocalView();
    valueview_t fmcopy = copy->getFactorMatrix(m)->getLocalView();
    size_t len = ktensor->getFactorMatrix(m)->getLocalLength();
    for (size_t i = 0; i < len; i++) {
      for (pt::rank_t r = 0; r < rank; r++) {
        if (fm(i,r) != fmcopy(i,r)) {
          ierr++;
          std::cout << me << " " << msg << ":  Error in copyData method " << m
                    << " " << fm(i,r) << " != " << fmcopy(i,r)
                    << std::endl;
        }
      }
    }
  }
  for (pt::rank_t r = 0; r < rank; r++) {
    if (ktensor->getLambdaView()(r) != copy->getLambdaView()(r)) {
       ierr++;
       std::cout << me << " " << msg 
                 << ":  Error in copyData method lambda rank " << r
                 << " " << ktensor->getLambdaView()(r) << " != "
                 << copy->getLambdaView()(r) << std::endl;
    }
  }

  delete copy;

  // Exercise the print() method
  size_t maxmodesize = 0;
  for (pt::mode_t m = 0; m < nModes; m++) 
    maxmodesize = (modeSizes[m] > maxmodesize ? modeSizes[m] : maxmodesize);

  if (rank < 3 && nModes < 4 && maxmodesize < 12)
    ktensor->print(msg);

  return ierr;
}

////////////////////////////////////////////////////////////////////////////////
// How to run the tests within testKtensor
template <typename scalar_t>
int testKtensor<scalar_t>::run()
{
  int ierr = 0;

#ifdef PT_LAYOUTRIGHT
  using host_view_type =
        Tpetra::LayoutRightMultiVector<double>::dual_view_type::t_host;
#else
  using host_view_type =
        Tpetra::MultiVector<double>::dual_view_type::t_host;
#endif

  {
    //////////////////////////////////////////////////////////////
    // Test a ktensor created from an array of default Tpetra::Maps
    // and an array of Kokkos views

    // Create array of maps
    std::vector<const map_t *> maps(nModes);
    for (pt::mode_t m = 0; m < nModes; m++)
      maps[m] = new map_t(modeSizes[m], 0, comm);
  
    // Create array of views
    std::vector<host_view_type> views(nModes);
    for (pt::mode_t m = 0; m < nModes; m++) {
      auto *map = maps[m];
      size_t nrows = map->getNodeNumElements();

      char label[20];
      sprintf(label, "mode%02d", m);
      views[m] = host_view_type(label, nrows, rank);

      auto v = views[m];
      for (size_t i = 0; i < nrows; i++)
        for (pt::rank_t r = 0; r < rank; r++)
          v(i,r) = buildValueFromLocalId(map, i, r);
    }
   
    // Create and test ktensor
    ktensorFromDefaultMaps = new ktensor_t(rank, maps, views, comm);
 
    ierr += checkKtensor("ktensorFromDefaultMaps", ktensorFromDefaultMaps, maps);
  }

  {
    //////////////////////////////////////////////////////////////
    // Test a ktensor created from an array of cyclic maps 
    // and an array of Kokkos views

    // Create array of maps
    std::vector<const map_t *> maps(nModes);
    for (pt::mode_t m = 0; m < nModes; m++) {
      maps[m] = buildCyclicMap(modeSizes[m]);
    }
  
    // Create array of views
    std::vector<host_view_type> views(nModes);
    for (pt::mode_t m = 0; m < nModes; m++) {
      auto *map = maps[m];
      size_t nrows = map->getNodeNumElements();

      char label[20];
      sprintf(label, "mode%02d", m);
      views[m] = host_view_type(label, nrows, rank);

      auto v = views[m];
      for (size_t i = 0; i < nrows; i++)
        for (pt::rank_t r = 0; r < rank; r++)
          v(i,r) = buildValueFromLocalId(map, i, r);
    }
   
    // Create and test ktensor
    ktensorFromCyclicMaps = new ktensor_t(rank, maps, views, comm);

    ierr += checkKtensor("ktensorFromCyclicMaps", ktensorFromCyclicMaps, maps);
  }

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

  int ierr = 0;
  pt::mode_t nModes;
  pt::rank_t rank;

  { // A small ktensor
    nModes = 3; 
    rank = 2;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 9;
    modeSizes[1] = 6;
    modeSizes[2] = 3;

    if (me == 0) {
      std::cout << std::endl
                << "TESTING WITH <DOUBLE> AND RANK=" 
                << rank << ", NMODES=" << nModes << " (";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
      std::cout << ")" << std::endl;
    }

    testKtensor<double> test(comm, modeSizes, rank);
    ierr += test.run();
  }

  { // A larger ktensor
    nModes = 10; 
    rank = 6;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 90;
    modeSizes[1] = 60;
    modeSizes[2] = 30;
    modeSizes[3] = 10;
    modeSizes[4] = 3;
    modeSizes[5] = 1;
    modeSizes[6] = 10;
    modeSizes[7] = 20;
    modeSizes[8] = 40;
    modeSizes[9] = 80;

    if (me == 0) {
      std::cout << std::endl
                << "TESTING WITH <DOUBLE> AND RANK=" 
                << rank << ", NMODES=" << nModes << " (";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
      std::cout << ")" << std::endl;
    }

    testKtensor<double> test(comm, modeSizes, rank);
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
