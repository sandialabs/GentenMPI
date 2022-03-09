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
 
// Unit test for distFactorMatrix class

#include "pt_factormatrix.hpp"
#include "pt_test_compare.hpp"
#include "Tpetra_Core.hpp"

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
    defaultMap(NULL),
    cyclicMap(NULL),
    factorFromDefaultMap(NULL),
    factorFromSizes(NULL),
    factorFromCyclicMap(NULL)
  { }


  // Destructor:  frees allocated pointers
  ~testFactorMatrix() 
  {
    if (defaultMap != NULL) delete defaultMap;
    if (cyclicMap != NULL) delete cyclicMap;
    if (factorFromDefaultMap != NULL) delete factorFromDefaultMap;
    if (factorFromSizes != NULL) delete factorFromSizes;
    if (factorFromCyclicMap != NULL) delete factorFromCyclicMap;
  }

  // A method for testing basic properties of a single factor matrix
  int checkFactor(const std::string &msg,
                  factormatrix_t *factor, map_t *map);

  // A method for testing doImport and doExport
  int checkImportExport(const std::string &msg,
                        factormatrix_t *factor, map_t *othermap);

  // How to run the tests within testFactorMatrix
  int run();

  // A constructed, easy to test data value
  inline scalar_t buildValue(lno_t lid, pt::rank_t factorrank) 
  {
    return scalar_t(me*10000 + factorrank*1000 + lid);
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
  factormatrix_t *factorFromDefaultMap;
  factormatrix_t *factorFromSizes;
  factormatrix_t *factorFromCyclicMap;
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
  factor->setValues(SCALAR);

  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      if (data(j, r) != SCALAR) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in setValues "
                  << data(j, r) << " != " << SCALAR << std::endl;
      }
    }
  }

  // Check norms
  Kokkos::View<scalar_t *> result("result", factor->getFactorRank());
  factor->norm1(result);
  scalar_t expected = factor->getGlobalLength() * ABSSCALAR;
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    if (result(r) != expected) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in norm1 for " << r << ": "
                << result(r) << " != " << expected << std::endl;
    }
  }

  factor->norm2(result);
  expected = std::sqrt(factor->getGlobalLength() * (SCALAR * SCALAR));
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    if (result(r) != expected) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in norm2 for " << r << ": "
                << result(r) << " != " << expected << std::endl;
    }
  }

  factor->normInf(result);
  expected = SCALAR;
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    if (result(r) != expected) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in normInf for " << r << ": "
                << result(r) << " != " << expected << std::endl;
    }
  }

  // Check normalization routine (which uses norm2)
  Kokkos::View<scalar_t *> norms("norms", factor->getFactorRank());
  factor->normalize(norms);
  expected = std::sqrt(factor->getGlobalLength() * (SCALAR * SCALAR));
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    if (norms(r) != expected) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in normalize "
                << "return argument for " << r << ": "
                << norms(r) << " != " << expected << std::endl;
    }
  }

  expected = SCALAR / expected;
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      if (factor->getLocalEntry(j, r) != expected) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in normalize values "
                  << factor->getLocalEntry(j, r) << " != " 
                  << expected << std::endl;
      }
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

  // Test scaleInverse method; should catch an error here since trying to 
  // divide by zero with scalevalues(0).
  bool caughtit = false;
  try {
    factor->scaleInverse(scalevalues);
  }
  catch (std::exception &e) {
    caughtit = true;
  }
  if (!caughtit) {
    std::cout << me << " " << msg << ":  Error in scaleInverse; "
              << "didn't throw on division by zero" << std::endl;
    ierr++;
  }

  // Now get rid of zero-valued scalevalue and try inverse again
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) scalevalues(r) = r+1;
  try {
    factor->scaleInverse(scalevalues);
  }
  catch (std::exception &e) {
    std::cout << me << " " << msg << ":  Error in scaleInverse; "
              << "error thrown despite good scalevalues" << std::endl;
    ierr++;
  }

  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      if (!pt::nearlyEqual<scalar_t>(factor->getLocalEntry(j, r), 
                                     SCALAR * r / (r+1))) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in scaleInverse "
                  << factor->getLocalEntry(j, r) << " != " 
                  << SCALAR * r / (r+1) << std::endl;
      }
    }
  }
  
  // Set values directly in view
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      data(j, r) = buildValue(j, r);
    }
  }

  // Check the values 
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {

      // test accessor method
      if (factor->getLocalEntry(j, r) != buildValue(j, r)) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in getLocalEntry "
                  << factor->getLocalEntry(j, r) << " != " 
                  << buildValue(j, r) << std::endl;
      }

      // test overloaded operator
      if ((*factor)(j, r) != buildValue(j, r)) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in "
                  << "overloaded operator fetch "
                  << (*factor)(j, r) << " != "
                  << buildValue(j, r) << std::endl;
      }
    }
  }

  // Set values using overloaded operator; test them
  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      (*factor)(j, r) *= 2.;
    }
  }

  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      if (data(j, r) != 2.*buildValue(j,r)) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in overloaded operator set "
                  << data(j, r) << " != " << SCALAR << std::endl;
      }
    }
  }

  // Set values using replaceGlobalValue; test them
  for (size_t i = 0; i < nGlobal; i++) {
    gno_t gid = gno_t(i);
    for (pt::rank_t r = 0; r < rank; r++) {
      scalar_t val = 0.;
      if (factor->getMap()->isNodeGlobalElement(gid))
        val = buildValue(factor->getMap()->getLocalElement(gid), r);
      factor->replaceGlobalValue(gid, r, val);
    }
  }

  for (pt::rank_t r = 0; r < factor->getFactorRank(); r++) {
    for (size_t j = 0; j < factor->getLocalLength(); j++) {
      if (data(j, r) != buildValue(j,r)) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in overloaded operator set "
                  << data(j, r) << " != " << buildValue(j,r) << std::endl;
      }
    }
  }

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
// How to run the tests within testFactorMatrix
template <typename scalar_t>
int testFactorMatrix<scalar_t>::run()
{
  int ierr = 0;

  // Create a factor matrix from a default Tpetra::Map
  defaultMap = new map_t(nGlobal, 0, comm);
  factorFromDefaultMap = new factormatrix_t(rank, defaultMap);

  ierr += checkFactor("factorFromDefaultMap",
                      factorFromDefaultMap, defaultMap);

  // Create a factor matrix from sizes
  factorFromSizes = new factormatrix_t(rank, nGlobal, comm);

  ierr += checkFactor("factorFromSizes",
                      factorFromSizes, defaultMap);

  // Create a factor matrix from a cyclic map (like dealing cards)
  cyclicMap = buildCyclicMap();
  factorFromCyclicMap = new factormatrix_t(rank, cyclicMap);

  ierr += checkFactor("factorFromCyclicMap",
                      factorFromCyclicMap, cyclicMap);
  
  // Test doImport and doExport

  ierr += checkImportExport("factorFromDefaultMap",
                            factorFromDefaultMap, cyclicMap);

  ierr += checkImportExport("factorFromCyclicMap",
                            factorFromCyclicMap, defaultMap);

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

  {
    nGlobal = 10;  // A tiny factor matrix
    rank = 2;

    if (me == 0) 
      std::cout << std::endl
                << "TESTING WITH <DOUBLE> AND NGLOBAL=" 
                << nGlobal << ", RANK=" << rank << std::endl;
    testFactorMatrix<double> test(comm, nGlobal, rank);
    ierr += test.run();
  }

  {
    nGlobal = 100;  // A medium factor matrix
    rank = 4;

    if (me == 0) 
      std::cout << std::endl
                << "TESTING WITH <DOUBLE> AND NGLOBAL=" 
                << nGlobal << ", RANK=" << rank << std::endl;
    testFactorMatrix<double> test(comm, nGlobal, rank);
    ierr += test.run();
  }

  {
    nGlobal = 4560; // A big factor matrix
    rank = 12;

    if (me == 0) 
      std::cout << std::endl
                << "TESTING WITH <DOUBLE> AND NGLOBAL=" 
                << nGlobal << ", RANK=" << rank << std::endl;
    testFactorMatrix<double> test(comm, nGlobal, rank);
    ierr += test.run();
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
