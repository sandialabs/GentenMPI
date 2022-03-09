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
 
// Unit test for distFactorMatrix and distKTensor classes: 
// test whether classes can accept Kokkos::View of data to be used in
// warm-start scenarios or with Chapel arrays 

#include "pt_factormatrix.hpp"
#include "pt_test_compare.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Teuchos_FancyOStream.hpp"

#define SMALL 25
#define SCALAR 123. 
#define ABSSCALAR (SCALAR >= 0 ? SCALAR : -1. * SCALAR)


template <typename scalar_t>
class testFactorMatrix {

public:

  typedef typename pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename factormatrix_t::gno_t gno_t;
  typedef typename factormatrix_t::lno_t lno_t;
  typedef typename factormatrix_t::map_t map_t;
  typedef typename factormatrix_t::valueview_t valueview_t;


  // Constructor:  initializes values
  testFactorMatrix(const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
                   size_t nGlobal_,
                   pt::rank_t rank_):
    comm(comm_),
    me(comm->getRank()),
    np(comm->getSize()),
    nGlobal(nGlobal_),
    rank(rank_),
    defaultMap(new map_t(nGlobal_, 0, comm_)),
    cyclicMap(buildCyclicMap())
  { }


  // Destructor:  frees allocated pointers
  ~testFactorMatrix() 
  {
    if (defaultMap != NULL) delete defaultMap;
    if (cyclicMap != NULL) delete cyclicMap;
  }

  // A method for testing basic properties of a single factor matrix
  int checkFactor(const std::string &msg,
                  factormatrix_t *factor, map_t *map);

  // A method for testing doImport and doExport
  int checkImportExport(const std::string &msg,
                        factormatrix_t *factor, map_t *othermap);

  // How to run the tests within testFactorMatrix
  int runWithKokkosView();
  int runWithSeparatePointerPerRank();

  // A constructed, easy to test data value
  inline scalar_t buildValueFromLocalId(lno_t lid, pt::rank_t factorrank) 
  {
    return scalar_t((cyclicMap->getGlobalElement(lid)+1)*1000 + factorrank);
  }
  inline scalar_t buildValueFromGlobalId(gno_t gid, pt::rank_t factorrank) 
  {
    return scalar_t((gid+1)*1000 + factorrank);
  }

  // Build a cyclic map (like dealing cards)
  map_t *buildCyclicMap()
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

private:
  const Teuchos::RCP<const Teuchos::Comm<int> > comm;
  const int me;
  const int np;

  const size_t nGlobal;
  const pt::rank_t rank;

  map_t *defaultMap;
  map_t *cyclicMap;
}; 


////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
int testFactorMatrix<scalar_t>::checkFactor(
  const std::string &msg,
  factormatrix_t *factor,
  map_t *map
)
{
  int ierr = 0;
  if (me == 0) std::cout << "checkFactor with " << msg << std::endl;

  // View of this factor's Kokkos data; used in many tests for both setting
  // and checking values.
  valueview_t data = factor->getLocalView();

  // Check sizes
  if (factor->getLocalLength() != map->getNodeNumElements()) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in local lengths "
              << factor->getLocalLength() << " != "
              << map->getNodeNumElements() << std::endl;
  }

  if (factor->getFactorRank() != rank) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in ranks "
              << factor->getFactorRank() << " != " << rank << std::endl;
  }

  if (factor->getGlobalLength() != map->getGlobalNumElements()) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in global lengths "
              << factor->getGlobalLength() << " != "
              << map->getGlobalNumElements() << std::endl;
  }

  // Test setValues

  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      scalar_t expected = buildValueFromLocalId(j,r);
      if (data(j, r) != expected) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in setValues "
                  << data(j, r) << " != " << expected
                  << std::endl;
      }
    }
  }

  // Check norms
  Kokkos::View<scalar_t *> result("result", factor->getFactorRank());

  factor->normInf(result);
  
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    scalar_t expected = buildValueFromGlobalId(nGlobal-1, r);
    if (result(r) != expected) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in normInf for " << r << ": "
                << result(r) << " != " << expected << std::endl;
    }
  }

  // Test scaling method
  factor->setValues(SCALAR);
  Kokkos::View<scalar_t *> scalevalues("scalevalues", factor->getFactorRank());
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) scalevalues(r) = r;
  factor->scale(scalevalues);

  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      if (factor->getLocalEntry(j, r) != SCALAR * r) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in scale "
                  << factor->getLocalEntry(j, r) << " != " 
                  << SCALAR * r << std::endl;
      }
    }
  }

  // Set values directly in view
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      data(j, r) = buildValueFromLocalId(j, r);
    }
  }

  // Check the values 
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {

      scalar_t expected = buildValueFromLocalId(j, r);

      // test accessor method
      if (factor->getLocalEntry(j, r) != expected) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in getLocalEntry "
                  << factor->getLocalEntry(j, r) << " != " 
                  << expected << std::endl;
      }

      // test overloaded operator
      if ((*factor)(j, r) != expected) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in "
                  << "overloaded operator fetch "
                  << (*factor)(j, r) << " != "
                  << expected << std::endl;
      }
    }
  }

  if (nGlobal < SMALL) factor->print(msg);

  // Set values using overloaded operator; test them
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      (*factor)(j, r) *= 2.;
    }
  }

  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      scalar_t expected = 2.*buildValueFromLocalId(j,r);
      if (data(j, r) != expected) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in overloaded operator set "
                  << data(j, r) << " != " << expected << std::endl;
      }
    }
  }

  // Set values using replaceGlobalValue; test them
  for (size_t i = 0; i < nGlobal; i++) {
    gno_t gid = gno_t(i);
    for (pt::rank_t r = 0; r < rank; r++) {
      scalar_t val = 0.;
      if (factor->getMap()->isNodeGlobalElement(gid))
        val = buildValueFromGlobalId(gid, r);
      factor->replaceGlobalValue(gid, r, val);
    }
  }

  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      scalar_t expected = buildValueFromLocalId(j,r);
      if (data(j, r) != expected) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in overloaded operator set "
                  << data(j, r) << " != " << expected << std::endl;
      }
    }
  }

  if (nGlobal < SMALL) factor->print(msg);

  // Test copy constructor
  factormatrix_t *copy = new factormatrix_t(*factor);
  data = factor->getLocalView();
  valueview_t copydata = copy->getLocalView();
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      if (copydata(j,r) != data(j,r)) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in copy constructor "
                  << copydata(j, r) << " != " << data(j,r) << std::endl;
      }
    }
  }

  // Test copyData operation
  copy->randomize();
  factor->copyData(copy);
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      if (copydata(j,r) != data(j,r)) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in copyData method"
                  << copydata(j, r) << " != " << data(j,r) << std::endl;
      }
    }
  }

  delete copy;

  // Exercise the print() method
  if (nGlobal < SMALL) factor->print(msg);

  return ierr;
}

////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
int testFactorMatrix<scalar_t> ::checkImportExport(
  const std::string &msg,
  factormatrix_t *infactor,
  map_t *othermap
)
{
  int ierr = 0;
  if (me == 0) std::cout << "checkImportExport with " << msg << std::endl;

  // create factor matrix using othermap
  factormatrix_t *otherfactor = new factormatrix_t(rank, othermap);

  // create importer with othermap and factor's map
  typedef typename factormatrix_t::import_t import_t;
  import_t importer(Teuchos::rcp(infactor->getMap(), false),
                    Teuchos::rcp(othermap, false));

  // import infactor into otherfactor
  otherfactor->doImport(infactor, &importer, Tpetra::INSERT);

  // create newfactor matrix with factor's map
  const map_t *newmap = infactor->getMap();
  factormatrix_t *newfactor = new factormatrix_t(rank, newmap);

  // create exporter with othermap and newfactor's map
  typedef typename factormatrix_t::export_t export_t;
  export_t exporter(Teuchos::rcp(othermap, false),
                    Teuchos::rcp(newmap, false));

  // export other factor into newfactor
  newfactor->doExport(otherfactor, &exporter, Tpetra::INSERT);

  // Make sure newfactor and infactor are sized the same
  if ((newfactor->getLocalLength() != infactor->getLocalLength()) ||
      (newfactor->getFactorRank() != infactor->getFactorRank())) {

    ierr++;
    std::cout << me << " " << msg << " incompatible sizes for import/export "
              << newfactor->getLocalLength() << " vs " 
              << infactor->getLocalLength() << "; "
              << newfactor->getFactorRank() << " vs "
              << infactor->getFactorRank() << std::endl;

  }
  else {

    // all entries of newfactor should match those of infactor; compare them
    valueview_t inview = infactor->getLocalView();
    valueview_t newview = newfactor->getLocalView();

    for (pt::rank_t r = 0; r < infactor->getFactorRank(); r++) {
      for (size_t j = 0; j < infactor->getLocalLength(); j++) {
        if (inview(j, r) != newview(j, r)) {
          ierr++;
          std::cout << me << " " << msg
                    << " comparison failed after import/export insert for ("
                    << j << ", " << r << "):  "
                    << inview(j, r) << " != " << newview(j, r) 
                    << std::endl;
        }
      }
    }
  }

  if (infactor->getMap()->isOneToOne() && othermap->isOneToOne()) {
    // Maps are one-to-one; can test Tpetra::ADD the same way as Tpetra::INSERT
    // Set newfactor to zero, and then use export to add entries.
    // Test using importer in reverse mode
    newfactor->setValues(0.);
    newfactor->doExport(otherfactor, &importer, Tpetra::ADD);

    // all entries of newfactor should match those of infactor; compare them
    valueview_t inview = infactor->getLocalView();
    valueview_t newview = newfactor->getLocalView();

    for (pt::rank_t r = 0; r < infactor->getFactorRank(); r++) {
      for (size_t j = 0; j < infactor->getLocalLength(); j++) {
        if (inview(j, r) != newview(j, r)) {
          ierr++;
          std::cout << me << " " << msg
                    << " comparison failed after import/export add for ("
                    << j << ", " << r << "):  "
                    << inview(j, r) << " != " << newview(j, r) 
                    << std::endl;
        }
      }
    }
  }

  delete otherfactor;
  delete newfactor;

  return ierr;
}

////////////////////////////////////////////////////////////////////////////////
// How to run the tests within testFactorMatrix using Kokkos view
template <typename scalar_t>
int testFactorMatrix<scalar_t>::runWithKokkosView()
{
  int ierr = 0;

  factormatrix_t *factorMatrix;
#ifdef PT_LAYOUTRIGHT
  using host_view_type = 
        Tpetra::LayoutRightMultiVector<double>::dual_view_type::t_host;
#else
  using host_view_type = 
        Tpetra::MultiVector<double>::dual_view_type::t_host;
#endif

  // Create a factor matrix from global IDs and a Kokkos::View on host
  {
    {
      auto nLocal = cyclicMap->getNodeNumElements();
      host_view_type host_view("host view", nLocal, rank);

      for (size_t i = 0; i < nLocal; i++) {
        for (pt::rank_t r = 0; r < rank; r++) {
          host_view(i,r) = buildValueFromLocalId(i, r);
        }
      }
   
      auto myGlobalIndices = cyclicMap->getMyGlobalIndices();

      factorMatrix = new factormatrix_t(myGlobalIndices, host_view, comm);
    }

    ierr += checkFactor("factorFromMapAndView",
                        factorMatrix, cyclicMap);

    ierr += checkImportExport("factorFromMapAndView",
                              factorMatrix, defaultMap);
    delete factorMatrix;
  }

  // Create a factor matrix from a Tpetra::Map and a Kokkos::View on host
  {
    {
      auto nLocal = cyclicMap->getNodeNumElements();
      host_view_type host_view("host view", nLocal, rank);

      for (size_t i = 0; i < nLocal; i++) {
        for (pt::rank_t r = 0; r < rank; r++) {
          host_view(i,r) = buildValueFromLocalId(i, r);
        }
      }
   
      factorMatrix = new factormatrix_t(cyclicMap, host_view);
    }

    ierr += checkFactor("factorFromMapAndView",
                        factorMatrix, cyclicMap);

    ierr += checkImportExport("factorFromMapAndView",
                              factorMatrix, defaultMap);
    delete factorMatrix;
  }

  return ierr;
}

////////////////////////////////////////////////////////////////////////////////
// How to run the tests within testFactorMatrix using separate pointers
template <typename scalar_t>
int testFactorMatrix<scalar_t>::runWithSeparatePointerPerRank()
{
#ifdef PT_LAYOUTRIGHT
  // Separate pointer per rank doesn't make sense for LayoutRight
  return 0;
#endif

  int ierr = 0;

  factormatrix_t *factorMatrix;
  // Create a factor matrix from a Tpetra::Map and separate memory
  // pointer for each rank of the factor matrix
  auto nLocal = cyclicMap->getNodeNumElements();

  std::vector<scalar_t *> vectorOfPtrs(rank);
  
#undef KDD_WANT_THIS_EVENTUALLY
#ifdef KDD_WANT_THIS_EVENTUALLY

  std::cout << "KDD SEPARATE VECTORS -- NEED THEM EQUALLY SPACED" << std::endl;
  for (pt::rank_t r = 0; r < rank; r++) {
    vectorOfPtrs[r] = new scalar_t[nLocal];
    scalar_t *ptr = vectorOfPtrs[r];
    for (size_t i = 0; i < nLocal; i++) {
      ptr[i] = buildValueFromLocalId(i, r);
    }
  }

#else 

  std::cout << "KDD -- ONE BIG ALLOC; WILL NOT WORK FOR CHAPEL" << std::endl;
  vectorOfPtrs[0] = new scalar_t[nLocal * rank];
  for (pt::rank_t r = 0; r < rank; r++) {
    if (r > 0) vectorOfPtrs[r] = vectorOfPtrs[0] + r * nLocal;
    scalar_t *ptr = vectorOfPtrs[r];
    for (size_t i = 0; i < nLocal; i++) {
      ptr[i] = buildValueFromLocalId(i, r);
    }
  }

#endif
 
  const auto tmp = cyclicMap->getMyGlobalIndices();
  std::vector<gno_t> myGlobalIndices(tmp.data(), tmp.data() + tmp.size());

  factorMatrix = new factormatrix_t(rank, myGlobalIndices, vectorOfPtrs, comm);

  ierr += checkFactor("factorFromMapAndVectorOfPtrs",
                      factorMatrix, cyclicMap);

  ierr += checkImportExport("factorFromMapAndVectorOfPtrs",
                            factorMatrix, defaultMap);

  delete factorMatrix;

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
  size_t nGlobal;
  pt::rank_t rank;

  { // Test ability to give a managed host view to factor matrix constructor
    nGlobal = 10;  // A tiny factor matrix
    rank = 2;
    
    if (me == 0) 
      std::cout << std::endl
                << "TESTING WITH <DOUBLE> AND NGLOBAL=" 
                << nGlobal << ", RANK=" << rank << std::endl;
    testFactorMatrix<double> test(comm, nGlobal, rank);
    ierr += test.runWithKokkosView();
    ierr += test.runWithSeparatePointerPerRank();
  }

  {
    nGlobal = 100;  // A medium factor matrix
    rank = 4;

    if (me == 0) 
      std::cout << std::endl
                << "TESTING WITH <DOUBLE> AND NGLOBAL=" 
                << nGlobal << ", RANK=" << rank << std::endl;
    testFactorMatrix<double> test(comm, nGlobal, rank);
    ierr += test.runWithKokkosView();
    ierr += test.runWithSeparatePointerPerRank();
  }

  {
    nGlobal = 4560; // A big factor matrix
    rank = 12;

    if (me == 0) 
      std::cout << std::endl
                << "TESTING WITH <DOUBLE> AND NGLOBAL=" 
                << nGlobal << ", RANK=" << rank << std::endl;
    testFactorMatrix<double> test(comm, nGlobal, rank);
    ierr += test.runWithKokkosView();
    ierr += test.runWithSeparatePointerPerRank();
  }

  if (ierr) std::cout << me << ":  " << ierr << " errors detected." << std::endl;

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
