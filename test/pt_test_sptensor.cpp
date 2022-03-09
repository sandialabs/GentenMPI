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

#include "pt_sptensor.hpp"
#include "pt_ktensor.hpp"
#include "pt_test_compare.hpp"
#include "Tpetra_Core.hpp"

template <typename scalar_t>
class testSptensor {

public:

  typedef typename pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename pt::distKtensor<factormatrix_t> ktensor_t;
  typedef typename pt::distSptensor<scalar_t> sptensor_t;

  typedef typename factormatrix_t::map_t map_t;
  typedef typename factormatrix_t::valueview_t valueview_t;
  typedef typename factormatrix_t::gno_t gno_t;
  typedef typename factormatrix_t::lno_t lno_t;

  // Constructor:  initializes values
  testSptensor(const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
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

  // Constructor:  initializes values
  testSptensor(const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
               std::vector<size_t> &modeSizes_,
               pt::rank_t rank_,
               typename sptensor_t::gnoview_t &globalIndices_,
               typename sptensor_t::valueview_t &values_,
               std::vector<gno_t> &minIds_,
               std::vector<size_t> &lens_
  ):
    comm(comm_),
    me(comm->getRank()),
    np(comm->getSize()),
    modeSizes(modeSizes_),
    nModes(modeSizes_.size()),
    rank(rank_),
    globalIndices(globalIndices_),
    values(values_),
    minIds(minIds_),
    lens(lens_)
  { }

  // Destructor:  frees allocated pointers
  ~testSptensor() {}

  // A method for testing basic properties of a single sptensor
  int checkSptensor(const std::string &msg, sptensor_t &sptensor);

  // How to run the tests within testSptensor
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

  inline scalar_t testVal(gno_t gid, pt::rank_t r) { return scalar_t(gid*r+1); }

  std::vector<gno_t> minIds;  // local box first ID in each mode
  std::vector<size_t> lens;   // local box mode length
}; 


////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
int testSptensor<scalar_t>::checkSptensor(
  const std::string &msg,
  sptensor_t &sptensor
)
{
  int ierr = 0;
  if (me == 0) std::cout << "checkSptensor with " << msg << std::endl;

  // Check sizes
  if (sptensor.getNumModes() != nModes) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in nModes "
              << sptensor.getNumModes() << " != "
              << nModes << std::endl;
  }

  size_t nnz = sptensor.getLocalNumNonZeros();
  if (nnz != globalIndices.extent(0)) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in nnz "
              << nnz << " != " << globalIndices.extent(0) << std::endl;
  }

  // Testing non-sampled sptensor; 
  // nnz should equal num indices; nzeros should be 0
  size_t nidx = sptensor.getLocalNumIndices();
  if (nnz != nidx) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in nIndices "
              << nnz << " != " << nidx << std::endl;
  }

  if (sptensor.getLocalNumZeros() != 0) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in nZeros "
              << sptensor.getLocalNumZeros() << " != 0 " << std::endl;
  }

  // Check local-to-global mapping
  typedef std::map<lno_t, gno_t> localtoglobal_t;
  std::vector<localtoglobal_t> localToGlobal(nModes);

  std::vector<gno_t> maxgid(nModes, std::numeric_limits<gno_t>::min());
  std::vector<gno_t> mingid(nModes, std::numeric_limits<gno_t>::max());
  std::vector<lno_t> maxlid(nModes, std::numeric_limits<lno_t>::min());
  std::vector<lno_t> minlid(nModes, std::numeric_limits<lno_t>::max());

  typename sptensor_t::gnoview_t gids = sptensor.getGlobalIndices();
  typename sptensor_t::lnoview_t lids = sptensor.getLocalIndices();
  
  for (size_t nz = 0; nz < nnz; nz++) {
    for (pt::mode_t m = 0; m < nModes; m++) {

      gno_t gm = gids(nz, m);

      // were global indices stored correctly?
      if (gm != globalIndices(nz, m)) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in global indices"
                  << "nz " << nz << " m " << m 
                  << " gid is " << gm 
                  << " gid was " << globalIndices(nz, m) << std::endl;
      }

      // build std::map of local index to global, and make sure local index
      // always maps to the same global index

      lno_t lm = lids(nz, m);
      if (localToGlobal[m].find(lm) == localToGlobal[m].end()) {
        // local index not yet in map; add it
        localToGlobal[m][lm] = gm;
      }

      if (localToGlobal[m][lm] != gm) {
        // local index has different global ID associated with it in std::map
        ierr++;
        std::cout << me << " " << msg << ":  Error in local-to-global index "
                  << "nz " << nz << " m " << m << " lid " << lm
                  << " gid is " << gm
                  << " gid was " << localToGlobal[m][lm] << std::endl;
      }

      if (localToGlobal[m][lm] != sptensor.getMap(m)->getGlobalElement(lm)) {
        // local index has different global ID associated with it in Tpetra::map
        ierr++;
        std::cout << me << " " << msg << ":  Error in Tpetra::Map "
                  << "nz " << nz << " m " << m << " lid " << lm
                  << " gid is " << sptensor.getMap(m)->getGlobalElement(lm)
                  << " gid was " << localToGlobal[m][lm] << std::endl;
      }

      // gather max/min gid/lid
      maxgid[m] = ((gm > maxgid[m]) ? gm : maxgid[m]);
      mingid[m] = ((gm < mingid[m]) ? gm : mingid[m]);
      maxlid[m] = ((lm > maxlid[m]) ? lm : maxlid[m]);
      minlid[m] = ((lm < minlid[m]) ? lm : minlid[m]);
    }
  }

  // compute global max/min global indices
  std::vector<gno_t> gmaxgid(nModes);
  std::vector<gno_t> gmingid(nModes);

  Teuchos::reduceAll<int,gno_t>(*comm, Teuchos::REDUCE_MAX,
                                nModes, &maxgid[0], &gmaxgid[0]);
  Teuchos::reduceAll<int,gno_t>(*comm, Teuchos::REDUCE_MIN,
                                nModes, &mingid[0], &gmingid[0]);

  // Test properties of the maps
  for (pt::mode_t m = 0; m < nModes; m++) {
    const map_t * const map = sptensor.getMap(m);

    // check map size
    if (localToGlobal[m].size() != map->getNodeNumElements()) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in map size " << m 
                << " " << localToGlobal[m].size()
                << " != getNodeNumElements " << map->getNodeNumElements()
                << std::endl;
    }

    // check index base:  should be smallest global index
    if (gmingid[m] != map->getIndexBase()) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in indexbase " << m << ": "
                << gmingid[m]
                << " != getIndexBase " << map->getIndexBase()
                << std::endl;
    }

    // check global max/min global indices
    if (gmaxgid[m] != map->getMaxAllGlobalIndex()) {
      ierr++;
      std::cout << me << " " << msg << ":  Error global maxgid[" << m << "] "
                << gmaxgid[m]
                << " != getMaxAllGlobalIndex " << map->getMaxAllGlobalIndex()
                << std::endl;
    }
    if (gmingid[m] != map->getMinAllGlobalIndex()) {
      ierr++;
      std::cout << me << " " << msg << ":  Error global mingid[" << m << "] "
                << gmingid[m]
                << " != getMinAllGlobalIndex " << map->getMinAllGlobalIndex()
                << std::endl;
    }

    // check local max/min global/local indices
    if (nnz > 0) {
      // Tpetra::Map returns meaningful local values for max/min global/local
      // indices only when getNodeNumElements > 0

      if (maxgid[m] != map->getMaxGlobalIndex()) {
        ierr++;
        std::cout << me << " " << msg << ":  Error maxgid[" << m << "] "
                  << maxgid[m]
                  << " != getMaxGlobalIndex " << map->getMaxGlobalIndex()
                  << std::endl;
      }  
  
      if (maxlid[m] != map->getMaxLocalIndex()) {
        ierr++;
        std::cout << me << " " << msg << ":  Error maxlid[" << m << "] "
                  << maxlid[m]
                  << " != getMaxLocalIndex " << map->getMaxLocalIndex()
                  << std::endl;
      }  
  
      if (mingid[m] != map->getMinGlobalIndex()) {
        ierr++;
        std::cout << me << " " << msg << ":  Error mingid[" << m << "] "
                  << mingid[m]
                  << " != getMinGlobalIndex " << map->getMinGlobalIndex()
                  << std::endl;
      }  
  
      if (minlid[m] != map->getMinLocalIndex()) {
        ierr++;
        std::cout << me << " " << msg << ":  Error minlid[" << m << "] "
                  << minlid[m]
                  << " != getMinLocalIndex " << map->getMinLocalIndex()
                  << std::endl;
      }  
    }

    // Make sure that each maps' entries are among the tensor's nonzeros
    for (size_t i = 0; i < map->getNodeNumElements(); i++) {
      if (map->getGlobalElement(i) != localToGlobal[m][i]) {
        ierr++;
        std::cout << me << " " << msg << ":  Error in map " << m 
                  << " entries: " << map->getGlobalElement(i)
                  << " != localToGlobal" << localToGlobal[m][i]
                  << std::endl;
      }
    }

    // For small maps, print some map info
    if (map->getGlobalNumElements() < 10) {
      std::cout << me << " " << msg 
                << "   Map " << m << " number of global elements="
                << map->getGlobalNumElements()
                << "; number of local elements="
                << map->getNodeNumElements() << std::endl;
      std::cout << me << "   Local map " << m << " elements: ";
      for (size_t i = 0; i < map->getNodeNumElements(); i++) 
        std::cout << map->getMyGlobalIndices()[i] << " ";
      std::cout << std::endl;
    }
  }

  // Test the Frobenius norm.  Assuming all nonzero values are 1, the
  // Frobenius norm should be sqrt(gnnz);
  size_t lnnz = sptensor.getLocalNumNonZeros();
  size_t gnnz;
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, 1, &lnnz, &gnnz);
  scalar_t fnorm = sptensor.frobeniusNorm();
  if (!pt::nearlyEqual<scalar_t>(fnorm, sqrt(gnnz))) {
    std::cout << me << " " << msg 
              << "  Error in Frobenius norm with gnnz =  " << gnnz << ": "
              << fnorm << " != " << sqrt(gnnz) << std::endl;
    ierr++;
  }

  // Test the bounding box, if provided
  if (minIds.size() > 0) {

    if (!sptensor.haveBoundingBox()) {
      std::cout << me << " " << msg 
                << "  Error:  sptensor does not have bounding box " 
                << std::endl; 
      ierr++;
    }

    for (pt::mode_t m = 0; m < nModes; m++) {
      gno_t min; 
      size_t len;
      sptensor.getLocalIndexRangeForMode(m, min, len);
      if (min != minIds[m] || len != lens[m]) {
        std::cout << me << " " << msg 
                  << "  Error:  sptensor.getLocalRangeForMode " << m 
                  << " returned wrong result " 
                  << " minID " << min << " != " << minIds[m] << " or "
                  << " len " << len << " != " << lens[m]
                  << std::endl; 
        ierr++;
      }
    }

    std::vector<gno_t> minv;
    std::vector<size_t> lenv;
    sptensor.getLocalIndexRange(minv, lenv);
    for (pt::mode_t m = 0; m < nModes; m++) {
      if (minv[m] != minIds[m] || lenv[m] != lens[m]) {
        std::cout << me << " " << msg 
                  << "  Error:  sptensor.getLocalRangeForMode " << m 
                  << " returned wrong result " 
                  << " minID " << minv[m] << " != " << minIds[m] << " or "
                  << " len " << lenv[m] << " != " << lens[m]
                  << std::endl; 
        ierr++;
      }
    }

    size_t product = 1;
    for (pt::mode_t m = 0; m < nModes; m++) product *= lens[m];
    if (sptensor.getLocalTensorSize() != product) {
      std::cout << me << " " << msg 
                << "  Error:  sptensor.getLocalTensorSize " 
                << " returned wrong result "
                << sptensor.getLocalTensorSize() << " != " << product
                << std::endl;
      ierr++;
    }
  }

  // Exercise the print() method
  if (gnnz < 25)
    sptensor.print(msg);

  return ierr;
}

////////////////////////////////////////////////////////////////////////////////
// How to run the tests within testSptensor
template <typename scalar_t>
int testSptensor<scalar_t>::run()
{
  int ierr = 0;

  // Build sptensor 

  sptensor_t *distTensor;
  if (minIds.size() > 0)
    distTensor = new sptensor_t(nModes, modeSizes, globalIndices, values, comm,
                                minIds, lens);
  else
    distTensor = new sptensor_t(nModes, modeSizes, globalIndices, values, comm);

  ierr += checkSptensor("distTensor", *distTensor);

  delete distTensor;

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


  { // A very small sptensor; all nonzeros initially on proc zero
    // Example from TTB test
    typedef double scalar_t;
    typedef typename testSptensor<scalar_t>::gno_t gno_t;

    nModes = 3; 
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 2;
    modeSizes[1] = 3;
    modeSizes[2] = 4;

    nnz = (me == 0 ? 1 : 0);
    Kokkos::View<gno_t **> indices("indices", nnz, nModes);
    Kokkos::View<scalar_t *> values("values", nnz);

    if (me == 0) {
      indices(0,0) = 1;
      indices(0,1) = 1;
      indices(0,2) = 1;
      values(0) = 1.;
    }

    rank = 1;

    if (me == 0) {
      std::cout << std::endl
                << "TESTING WITH <DOUBLE> AND RANK=" 
                << rank << ", NMODES=" << nModes << " (";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
      std::cout << ")" << std::endl;
    }

    testSptensor<scalar_t> test(comm, modeSizes, rank, indices, values);
    ierr += test.run();
  }

  { // A generated tensor with 10 nonzeros per processor
    typedef float scalar_t;
    typedef typename testSptensor<scalar_t>::gno_t gno_t;
    nnz = 10;

    nModes = 4;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = np;
    modeSizes[1] = 2 * (np - 1) + (nnz - 1) + 1;
    modeSizes[2] = 2 * (np - 1) + 2 * (nnz - 1) + 1;
    modeSizes[3] = nnz;
   
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

    testSptensor<scalar_t> test(comm, modeSizes, rank, indices, values);
    ierr += test.run();
  }

  { // A generated tensor with block-based random input
    typedef float scalar_t;
    typedef typename testSptensor<scalar_t>::gno_t gno_t;

    nnz = 1000;
    nModes = 3;
    std::vector<size_t> modeSizes(nModes);
    const size_t mult[3] = {100, 200, 300};
    modeSizes[0] = mult[0] * np;
    modeSizes[1] = mult[1] * np;
    modeSizes[2] = mult[2] * np;

    Kokkos::View<gno_t **> indices("indices", nnz, nModes);
    Kokkos::View<scalar_t *> values("values", nnz);

    srand(me);

    for (size_t nz = 0; nz < nnz; nz++) {
      for (pt::mode_t m = 0; m < nModes; m++)
        indices(nz, m) = mult[m] * me + rand() % mult[m];
      values(nz) = 1.;
    }

    std::vector<gno_t> minIds(nModes);
    std::vector<size_t> lens(nModes);
    for (pt::mode_t m = 0; m < nModes; m++) {
      minIds[m] = mult[m] * me;
      lens[m] = mult[m];
    }

    rank = 10;

    if (me == 0) {
      std::cout << std::endl
                << "TESTING WITH <FLOAT, INT, INT> AND RANK=" 
                << rank << ", NMODES=" << nModes << " (";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
      std::cout << ")" << std::endl;
    }

    testSptensor<scalar_t> test(comm, modeSizes, rank,
                                indices, values, minIds, lens);
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
