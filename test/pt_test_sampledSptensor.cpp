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
 
// Unit test for distSptensor class with sampled Sptensor

#include "pt_sptensor.hpp"
#include "pt_ktensor.hpp"
#include "pt_test_compare.hpp"
#include "Tpetra_Core.hpp"

template <typename scalar_t>
class testSampledSptensor {

public:

  typedef typename pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename pt::distKtensor<factormatrix_t> ktensor_t;
  typedef typename pt::distSptensor<scalar_t> sptensor_t;
  typedef typename factormatrix_t::gno_t gno_t;
  typedef typename factormatrix_t::lno_t lno_t;
  typedef typename factormatrix_t::map_t map_t;
  typedef typename factormatrix_t::valueview_t valueview_t;

  // Constructor:  initializes values
  testSampledSptensor(const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
               std::vector<size_t> &modeSizes_,
               typename sptensor_t::gnoview_t &globalIndices_,
               typename sptensor_t::valueview_t &values_,
               std::vector<gno_t> &bbMinGid_,
               std::vector<size_t> &bbModeSizes_
  ):
    comm(comm_),
    me(comm->getRank()),
    np(comm->getSize()),
    modeSizes(modeSizes_),
    nModes(modeSizes_.size()),
    globalIndices(globalIndices_),
    values(values_),
    bbMinGid(bbMinGid_),
    bbModeSizes(bbModeSizes_)
  { }

  // Destructor:  frees allocated pointers
  ~testSampledSptensor() {}

  // A method for testing basic properties of a single sptensor
  int checkSampledSptensor(const std::string &msg, 
                           sptensor_t &sampledSptensor,
                           sptensor_t &origSptensor,
                           size_t nzSample, size_t zSample);

  // A method for resampling tensor and testing its properties
  int  resampleSptensor(const std::string &msg, sptensor_t &sptensor);

  // How to run the tests within testSampledSptensor
  int run();

private:
  const Teuchos::RCP<const Teuchos::Comm<int> > comm;
  const int me;
  const int np;

  std::vector<size_t> modeSizes;
  const pt::mode_t nModes;

  typename sptensor_t::gnoview_t globalIndices;
  typename sptensor_t::valueview_t values;

  std::vector<gno_t> bbMinGid;
  std::vector<size_t> bbModeSizes;

  inline size_t computeSampleSize(size_t orig) {
    return std::max(size_t(1), size_t(0.001 * orig));
  }
}; 


////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
int testSampledSptensor<scalar_t>::checkSampledSptensor(
  const std::string &msg,
  sptensor_t &sampledSptensor,
  sptensor_t &origSptensor,
  size_t nzSamples,
  size_t zSamples
)
{
  int ierr = 0;
  if (me == 0) std::cout << "checkSampledSptensor with " << msg << std::endl;

  // Check sizes
  if (sampledSptensor.getNumModes() != nModes) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in nModes "
              << sampledSptensor.getNumModes() << " != "
              << nModes << std::endl;
  }

  size_t nidx = sampledSptensor.getLocalNumIndices();

  size_t nnz = sampledSptensor.getLocalNumNonZeros();
  if (origSptensor.getLocalNumNonZeros()) {
    if (nnz != nzSamples) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in nonzero sample size "
                << nnz << " !=  " << nzSamples << std::endl;
    }
  }
  else {
    // origSptensor had no nonzeros to sample; sampledSptensor should not
    // have any nonzeros then.
    if (nnz != 0) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in nonzero sample size "
                << nnz << " !=  0" 
                << std::endl;
    }
  }

  size_t nz = sampledSptensor.getLocalNumZeros();
  if (sampledSptensor.getLocalTensorSize() > 0 && nz != zSamples) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in zero sample size "
              << nz << " !=  " << zSamples << std::endl;
  }
  else if (sampledSptensor.getLocalTensorSize() == 0 && nz != 0) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in zero sample size "
              << nz << " zeros sampled from empty bounding box " << std::endl;
  }

  if (nnz + nz != nidx) {
    ierr++;
    std::cout << me << " " << msg << ":  Error in nnz + nZeros "
              << nnz << " + " << nz << " != " << nidx << std::endl;
  }

  // Check local-to-global mapping
  typedef std::map<lno_t, gno_t> localtoglobal_t;
  std::vector<localtoglobal_t> localToGlobal(nModes);

  std::vector<gno_t> maxgid(nModes, std::numeric_limits<gno_t>::min());
  std::vector<gno_t> mingid(nModes, std::numeric_limits<gno_t>::max());
  std::vector<lno_t> maxlid(nModes, std::numeric_limits<lno_t>::min());
  std::vector<lno_t> minlid(nModes, std::numeric_limits<lno_t>::max());

  typename sptensor_t::gnoview_t gids = sampledSptensor.getGlobalIndices();
  typename sptensor_t::lnoview_t lids = sampledSptensor.getLocalIndices();
  
  for (size_t lnz = 0; lnz < nidx; lnz++) {
    for (pt::mode_t m = 0; m < nModes; m++) {

      gno_t gm = gids(lnz, m);

      // TODO:  should make sure every sampled nonzero is actually an 
      // TODO:  original nonzero
      // TODO:  requires search of globalIndices array

      // build std::map of local index to global, and make sure local index
      // always maps to the same global index

      lno_t lm = lids(lnz, m);
      if (localToGlobal[m].find(lm) == localToGlobal[m].end()) {
        // local index not yet in map; add it
        localToGlobal[m][lm] = gm;
      }

      if (localToGlobal[m][lm] != gm) {
        // local index has different global ID associated with it in std::map
        ierr++;
        std::cout << me << " " << msg << ":  Error in local-to-global index "
                  << "nz " << lnz << " m " << m << " lid " << lm
                  << " gid is " << gm
                  << " gid was " << localToGlobal[m][lm] << std::endl;
      }

      if (localToGlobal[m][lm] != 
          sampledSptensor.getMap(m)->getGlobalElement(lm)) {
        // local index has different global ID associated with it in Tpetra::map
        ierr++;
        std::cout << me << " " << msg << ":  Error in Tpetra::Map "
                  << "nz " << lnz << " m " << m << " lid " << lm
                  << " gid is " 
                  << sampledSptensor.getMap(m)->getGlobalElement(lm)
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
    const map_t * const map = sampledSptensor.getMap(m);

    // check map size
    if (localToGlobal[m].size() != map->getNodeNumElements()) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in map size " << m 
                << " " << localToGlobal[m].size()
                << " != getNodeNumElements " << map->getNodeNumElements()
                << std::endl;
    }

    // check index base:  should be same as original tensor
    if (origSptensor.getMap(m)->getIndexBase() != map->getIndexBase()) {
      ierr++;
      std::cout << me << " " << msg << ":  Error in indexbase " << m << ": "
                << origSptensor.getMap(m)->getIndexBase()
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
    if (nidx > 0) {
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
                  << " != localToGlobal " << localToGlobal[m][i]
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
  size_t lnnz = sampledSptensor.getLocalNumNonZeros();
  size_t gnnz;
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, 1, &lnnz, &gnnz);
  scalar_t fnorm = sampledSptensor.frobeniusNorm();
  if (!pt::nearlyEqual<scalar_t>(fnorm, sqrt(gnnz))) {
    std::cout << me << " " << msg 
              << "  Error in Frobenius norm with gnnz =  " << gnnz << ": "
              << fnorm << " != " << sqrt(gnnz) << std::endl;
    ierr++;
  }

  // Test the bounding box:  all global indices should be within the box
  for (size_t i = 0; i < sampledSptensor.getLocalNumIndices(); i++) {
    for (pt::mode_t m = 0; m < nModes; m++) {
      if (gids(i,m) < bbMinGid[m] || 
          gids(i,m) >= bbMinGid[m] + static_cast<gno_t>(bbModeSizes[m])) {
        std::cout << me << " " << msg
                  << "  Invalid sampled index in mode " << m << ": " 
                  << gids(i,m) << " not in range "
                  << bbMinGid[m] << "-" << bbMinGid[m]+bbModeSizes[m]-1
                  << std::endl;
        ierr++;
      }
    }
  }

  // Exercise the print() method
  size_t lnz = sampledSptensor.getLocalNumZeros();
  size_t gnz;
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, 1, &lnz, &gnz);
  if (gnnz + gnz < 25)
    sampledSptensor.print(msg);

  return ierr;
}

////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
int  testSampledSptensor<scalar_t>::resampleSptensor(
  const std::string &msg, 
  sptensor_t &sptensor
)
{
  int ierr = 0;
  std::vector<gno_t> keepFirstIndex(nModes); 
  std::vector<gno_t> keepLastIndex(nModes);
  size_t nIndices = sptensor.getLocalNumIndices();

  if (nIndices) {
    // Store first and last indices to confirm that we have resampled
    // Pretty unlikely (though not impossible) that we'd resample the 
    // same indices.

    auto globalIndices = sptensor.getGlobalIndices();

    for (pt::mode_t m = 0; m < nModes; m++) {
      keepFirstIndex[m] = globalIndices(0,m);
      keepLastIndex[m] = globalIndices(nIndices-1, m);
    }
  }

  sptensor.resample(123);

  // Compare new indices to the saved ones
  globalIndices = sptensor.getGlobalIndices();
  bool samefirst = true;
  bool samelast = true;

  if (nIndices) {
    for (pt::mode_t m = 0; m < nModes; m++) {
      if (globalIndices(0,m) != keepFirstIndex[m]) samefirst = false;
      if (globalIndices(nIndices-1,m) != keepLastIndex[m]) samelast = false;
    }
   
    if (samefirst && sptensor.getLocalNumNonZeros() > 1) {
      std::cout << me << " " << msg 
                << ":  Potential error -- same resampled first index"
                << std::endl;
    }
    if (samelast && sptensor.getLocalNumZeros() > 1) {
      std::cout << me << " " << msg 
                << ":  Potential error -- same resampled last index"
                << std::endl;
    }
    if ((samefirst && sptensor.getLocalNumNonZeros() > 1) &&
        (samelast && sptensor.getLocalNumZeros() > 1)) {
      std::cout << me << " " << msg 
                << ":  Error -- first and last indices unchanged in sampling"
                << std::endl;
      ierr++;
    }
  }
  return ierr;
}

////////////////////////////////////////////////////////////////////////////////
// How to run the tests within testSampledSptensor
template <typename scalar_t>
int testSampledSptensor<scalar_t>::run()
{
  int ierr = 0;

  {
    // Build sptensor without bounding box; sampling should not work;
    // catch errors
    sptensor_t distTensor(nModes, modeSizes, globalIndices, values, comm);

    size_t nzSample = computeSampleSize(distTensor.getLocalNumNonZeros());

    size_t globalTensorSize = 1;
    for (pt::mode_t m = 0; m < nModes; m++) globalTensorSize *= modeSizes[m];
    size_t zSample = computeSampleSize(globalTensorSize);

    bool caught = false;
    try {
      sptensor_t *semiStratTensor = 
                  distTensor.semiStratSampledTensor(nzSample, zSample);
      delete semiStratTensor;
    }
    catch (std::exception &e) {
      if (me == 0) 
        std::cout << "semiStrat bounding box existence check OK" << std::endl;
      caught = true;
    }
    if (!caught) {
      std::cout << "Error: semi-strat sampling ran without bounding box"
                << std::endl;
      ierr += 1;
    }

    caught = false;
    try {
      sptensor_t *stratTensor = 
                  distTensor.stratSampledTensor(nzSample, zSample);
      delete stratTensor;
    }
    catch (std::exception &e) {
      if (me == 0) 
        std::cout << "strat bounding box existence check OK" << std::endl;
      caught = true;
    }

    if (!caught) {
      std::cout << "Error:  strat sampling ran without bounding box"
                << std::endl;
      ierr += 1;
    }
  }

  {
    // Build sptensor with bounding box; sampling and resampling should work
    sptensor_t distTensor(nModes, modeSizes, globalIndices, values, comm, 
                          bbMinGid, bbModeSizes);

    size_t nzSample = computeSampleSize(distTensor.getLocalNumNonZeros());
    size_t zSample = computeSampleSize(distTensor.getLocalTensorSize());

    if (me == 0) std::cout << "Creating semiStrat tensor" << std::endl;
    sptensor_t *semiStratTensor = 
                distTensor.semiStratSampledTensor(nzSample, zSample);

    ierr += checkSampledSptensor("semiStratTensor", *semiStratTensor,
                                 distTensor, nzSample, zSample);

    ierr += resampleSptensor("semiStratTensor", *semiStratTensor);

    ierr += checkSampledSptensor("resampled_semiStratTensor", *semiStratTensor,
                                 distTensor, nzSample, zSample);
    delete semiStratTensor;

    if (me == 0) std::cout << "Creating strat tensor" << std::endl;
    sptensor_t *stratTensor = distTensor.stratSampledTensor(nzSample, zSample);

    ierr += checkSampledSptensor("stratTensor", *stratTensor, distTensor,
                                 nzSample, zSample);

    ierr += resampleSptensor("stratTensor", *stratTensor);

    ierr += checkSampledSptensor("resampled_stratTensor", *stratTensor,
                                 distTensor, nzSample, zSample);

    delete stratTensor;
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
  int np = comm->getSize();

  int ierr = 0;
  pt::mode_t nModes;
  size_t nnz;

  ////////////////////////
  { // A very small sptensor; all nonzeros initially on proc zero
    typedef double scalar_t;
    typedef typename testSampledSptensor<scalar_t>::gno_t gno_t;

    nModes = 3; 
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 2;
    modeSizes[1] = 3;
    modeSizes[2] = 4;

    nnz = (me == 0 ? 1 : 0);
    Kokkos::View<gno_t **> indices("indices", nnz, nModes);
    Kokkos::View<scalar_t *> values("values", nnz);

    if (me == 0) {
      indices(0,0) = 0;
      indices(0,1) = 0;
      indices(0,2) = 0;
      values(0) = 1.;
    }
    
    // bounding boxes split in mode 2 (1D); 
    // with np > modeSizes[2], some procs' boxes will be empty
    std::vector<size_t> bbModeSizes(3);
    bbModeSizes[0] = modeSizes[0];
    bbModeSizes[1] = modeSizes[1];
    bbModeSizes[2] = modeSizes[2] / np + (me < gno_t(modeSizes[2]) % np);

    std::vector<gno_t> bbMinGid(3);
    bbMinGid[0] = 0;
    bbMinGid[1] = 0;
    bbMinGid[2] = me * (modeSizes[2] / np) 
                + std::min<gno_t>(me, gno_t(modeSizes[2])%np);
    
    if (me == 0) {
      std::cout << std::endl
                << "TESTING WITH <DOUBLE, INT, INT> " 
                << " NMODES=" << nModes << " (";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
      std::cout << ")" << std::endl;
    }

    testSampledSptensor<scalar_t> test(comm, modeSizes, indices, values,
                                       bbMinGid, bbModeSizes);
    ierr += test.run();
  }

  ////////////////////////
  { // A generated tensor with 10 nonzeros per processor
    typedef float scalar_t;
    typedef typename testSampledSptensor<scalar_t>::gno_t gno_t;
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

    // bounding boxes split 1D in mode 0
    std::vector<size_t> bbModeSizes(4);
    bbModeSizes[0] = 1;
    bbModeSizes[1] = modeSizes[1];
    bbModeSizes[2] = modeSizes[2];
    bbModeSizes[3] = modeSizes[3];

    std::vector<gno_t> bbMinGid(4);
    bbMinGid[0] = me;
    bbMinGid[1] = 0;
    bbMinGid[2] = 0;
    bbMinGid[3] = 0;

    if (me == 0) {
      std::cout << std::endl
                << "TESTING WITH <FLOAT> "
                << " NMODES=" << nModes << " (";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
      std::cout << ")" << std::endl;
    }

    testSampledSptensor<scalar_t> test(comm, modeSizes, indices, values,
                                       bbMinGid, bbModeSizes);
    ierr += test.run();
  }

  ////////////////////////
  { // A generated tensor with block-based random input
    typedef float scalar_t;
    typedef typename testSampledSptensor<scalar_t>::gno_t gno_t;

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

    // bounding boxes 
    std::vector<size_t> bbModeSizes(3);
    bbModeSizes[0] = modeSizes[0];
    bbModeSizes[1] = mult[1];
    bbModeSizes[2] = modeSizes[2];

    std::vector<gno_t> bbMinGid(3);
    bbMinGid[0] = 0;
    bbMinGid[1] = me * mult[1];
    bbMinGid[2] = 0;

    if (me == 0) {
      std::cout << std::endl
                << "TESTING WITH <FLOAT> " 
                << "NMODES=" << nModes << " (";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
      std::cout << ")" << std::endl;
    }

    testSampledSptensor<scalar_t> test(comm, modeSizes, indices, values,
                                    bbMinGid, bbModeSizes);
    ierr += test.run();
  }

  ////////////////////////
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
