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
 
#ifndef PT_SPTENSOR_
#define PT_SPTENSOR_

#include "pt_shared.h"
#include "pt_factormatrix.hpp"
#include "pt_sptensor_boundingbox.hpp"
#include "pt_samplingstrategies.hpp"

#include <Tpetra_Map.hpp>
#include <Tpetra_Details_makeOptimizedColMap.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <unordered_set>

namespace pt {

#ifdef TIME_RESAMPLE
static Teuchos::RCP<Teuchos::Time>
      timeResample(Teuchos::TimeMonitor::getNewTimer("RESAMPLE 01 sampleTheTensor")),
      timeBuildMaps(Teuchos::TimeMonitor::getNewTimer("RESAMPLE 03 BuildMaps")),
      timeWaitOne(Teuchos::TimeMonitor::getNewTimer("RESAMPLE 00 WaitOne")),
      timeWaitTwo(Teuchos::TimeMonitor::getNewTimer("RESAMPLE 02 WaitTwo")),
      timeWaitThree(Teuchos::TimeMonitor::getNewTimer("RESAMPLE 04 WaitThree"));
#endif

template <typename SCALAR>
class distSptensor 
{
public:

  typedef SCALAR scalar_t;

  typedef distSptensor<scalar_t> sptensor_t;
  typedef distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename factormatrix_t::gno_t gno_t;
  typedef typename factormatrix_t::lno_t lno_t;

  typedef typename factormatrix_t::map_t map_t;
  typedef typename factormatrix_t::import_t import_t;
  typedef typename factormatrix_t::export_t export_t;

  typedef Kokkos::View<scalar_t *> valueview_t;
  typedef Kokkos::View<gno_t **> gnoview_t;
  typedef Kokkos::View<lno_t **> lnoview_t;
  
  typedef TensorHashMap<scalar_t,gno_t> hashmap_t;

  // Destructor
  ~distSptensor() { 
    for (mode_t m = 0; m < nModes; m++) delete maps[m]; 
  }

  // Constructor for which user provides the Kokkos views representing
  // the nonzeros in coordinate format. 
  distSptensor(mode_t nModes,
               const std::vector<size_t> &modeSizesGuess_,
               const Kokkos::View<gno_t **> &myGlobalIndices_,
               const Kokkos::View<scalar_t *> &values_,
               const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
               const std::vector<gno_t> &bbMinIndex = std::vector<gno_t>(),
               const std::vector<size_t> &bbModeSizes = std::vector<size_t>());

  // Constructor for which user provides the Kokkos views representing
  // the nonzeros in coordinate format; allow user to distinguish between
  // zeros and nonzeros
  distSptensor(mode_t nModes,
               const std::vector<size_t> &modeSizesGuess_,
               const Kokkos::View<gno_t **> &myGlobalIndices_,
               const Kokkos::View<scalar_t *> &values_,
               const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
               const size_t numNonzeros_, 
               const size_t numZeros_,
               const std::vector<gno_t> &bbMinIndex = std::vector<gno_t>(),
               const std::vector<size_t> &bbModeSizes = std::vector<size_t>());

  // Constructor which sptensor uses to create sampled tensors
  distSptensor(
    mode_t nModes,
    const std::vector<size_t> &modeSizesGuess_,
    const Kokkos::View<gno_t **> &myGlobalIndices_,
    const Kokkos::View<scalar_t *> &values_,
    const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
    const size_t numNonzeros_, 
    const size_t numZeros_,
    const Teuchos::RCP<const SamplingStrategy<sptensor_t> > &samplingStrategy_,
    const Teuchos::RCP<const distSptensorBoundingBox> &boundingBox_);
     
  // Construct sampled version of existing sparse tensor using semi-stratified 
  // sampling
  // - numNonzeros nonzeros are sampled
  // - numZeros zeros are sampled (but some zeros may actually be nonzeros)
  // - seed is seed for random number generator
  sptensor_t *semiStratSampledTensor( 
    size_t numNonzeros, 
    size_t numZeros, 
    unsigned int seed = 1)
  {
    typedef SemiStratifiedSamplingStrategy<sptensor_t> sampler_t;
    typedef SamplingStrategy<sptensor_t> base_sampler_t;
    Teuchos::RCP<base_sampler_t> sampler = Teuchos::rcp(new sampler_t(this));
    return getSampledTensor(sampler, numNonzeros, numZeros, seed);
  }
               
  // Construct sampled version of existing sparse tensor using stratified 
  // sampling
  // - numNonzeros nonzeros are sampled
  // - numZeros zeros are sampled (and sampled zeros are actually zeros)
  // - seed is seed for random number generator
  sptensor_t *stratSampledTensor( 
    size_t numNonzeros, 
    size_t numZeros, 
    unsigned int seed = 1)
  {
    typedef StratifiedSamplingStrategy<sptensor_t> sampler_t;
    typedef SamplingStrategy<sptensor_t> base_sampler_t;
    Teuchos::RCP<base_sampler_t> sampler = Teuchos::rcp(new sampler_t(this));
    return getSampledTensor(sampler, numNonzeros, numZeros, seed);
  }
  
  // Construct sampled version of existing sparse tensor using provided
  // samplingStrategy
  // - numNonzeros nonzeros are sampled according to samplingStrategy
  // - numZeros zeros are sampled according to samplingStrategy
  // - seed is seed for random number generator
  sptensor_t *getSampledTensor(
    Teuchos::RCP<SamplingStrategy<sptensor_t> > &sampler,
    size_t nRequestedNonZeros, 
    size_t nRequestedZeros, 
    unsigned int seed = 1)
  {
    // Allocate memory for the newly sampled indices
    // Zeros and nonzeros are both stored explicitly
     
    size_t numSamples = nRequestedNonZeros + nRequestedZeros;

    Kokkos::View<gno_t **> inds("inds", numSamples, nModes);
    Kokkos::View<scalar_t *> vals("vals", numSamples);
  
    // Sample the tensor, giving new inds, vals, nzSamples, zSamples
    size_t nzSamples, zSamples;  // actual number of samples drawn
    sampler->sampleTheTensor(nRequestedNonZeros, nRequestedZeros,
                             nzSamples, zSamples,
                             inds, vals, seed);
  
    // Construct and return sampled tensor
    return new sptensor_t(nModes, modeSizes, inds, vals, comm,
                          nzSamples, zSamples, sampler,
                          this->boundingBox);
  }

  ////////////
  // Modifiers 
  void resample(unsigned int seed)
  {
    // Sample the same number of nonzeros and zeros as previously sampled.

#ifdef TIME_RESAMPLE
    timeWaitOne->start();
    comm->barrier();
    timeWaitOne->stop();
    timeResample->start();
#endif
    size_t nzSamples, zSamples;
    samplingStrategy->sampleTheTensor(nNonZeros, nZeros, nzSamples, zSamples,
                                      globalIndices, values, seed);

    // Update the relevant constants based on nzSamples, zSamples.
    nNonZeros = nzSamples;
    nZeros = zSamples;
    nIndices = nNonZeros + nZeros;

#ifdef TIME_RESAMPLE
    timeResample->stop();
    timeWaitTwo->start();
    comm->barrier();
    timeWaitTwo->stop();
    timeBuildMaps->start();
#endif

    // Build the maps and reassign the localIndices using the new globalIndices
    if (boundingBox != Teuchos::null)
      buildMapsWithBB();
    else 
      buildMaps();

#ifdef TIME_RESAMPLE
    timeBuildMaps->stop();
    timeWaitThree->start();
    comm->barrier();
    timeWaitThree->stop();
#endif
  }

  inline void setValues(scalar_t val) { Kokkos::deep_copy(values, val); }

  import_t *optimizeMapAndBuildImporter(mode_t m, const map_t *fmMap) {
    bool err = false;
    // Use Tpetra to optimize map/importer for fewer copies during communication
    std::pair<Teuchos::RCP<const map_t>, Teuchos::RCP<import_t> > result = 
      Tpetra::Details::makeOptimizedColMapAndImport<map_t>(
                           std::cerr, err, *fmMap, *(maps[m]));
    delete maps[m];
    map_t *newMap = const_cast<map_t*>(result.first.getRawPtr());
    maps[m] = newMap;
    result.first.release();

    // Redo local indexing in SpTensor to agree with map
    for (size_t i = 0; i < nIndices; i++)
      localIndices(i,m) = newMap->getLocalElement(globalIndices(i,m));

    // return importer
    import_t *retval = result.second.getRawPtr();
    result.second.release();

    return retval;
  }

  // Accessors
  // The number of sptensor non-zeros on this processor
  inline size_t getLocalNumNonZeros() const { return nNonZeros; }

  // The number of sampled zero-valued tensor indices on this processor
  inline size_t getLocalNumZeros() const { return nZeros; }

  // The number of sptensor non-zeros plus sampled zero-valued tensor 
  // indices on this processor
  inline size_t getLocalNumIndices() const { return nIndices; }

  // In a sampled tensor, pointer to the tensor from which it was sampled.
  // In a non-sampled tensor, returns this.
  inline const sptensor_t *getSourceTensor() const { 
    if (samplingStrategy != Teuchos::null) 
      return samplingStrategy->getSourceTensor(); 
    else
      return this;
  }

  // If a bounding box of tensor indices is available (e.g., through generation
  // or through medium-grained or RCB decomposition), the number of indices
  // possible within the bounding box.  Used in sampling zeros.
  inline double getLocalTensorSize() const { 
    if (boundingBox != Teuchos::null) return boundingBox->getBoxSize();
    else throw std::runtime_error("No bounding box information available");
  }

  inline bool canSampleTensor() const { 
    if (boundingBox != Teuchos::null) return boundingBox->goodForSampling(); 
    else return false;
  }

  inline mode_t getNumModes() const { return nModes; }

  inline gnoview_t getGlobalIndices() const { return globalIndices; }

  inline lnoview_t getLocalIndices() const { return localIndices; }

  inline valueview_t getValues() const { return values; }

  inline size_t getModeSize(size_t m) const { return modeSizes[m]; }

  inline const std::vector<size_t> &getModeSizes() const { return modeSizes; }

  inline const map_t *getMap(mode_t m) const { return maps[m]; }

  inline const Teuchos::RCP<const Teuchos::Comm<int> > &getComm() const {
    return comm;
  }
  
  // Frobenius norm (sqrt of sum of squares of all values)
  // This is a global operation, requiring synchronization
  scalar_t frobeniusNorm() const
  {
    scalar_t gsum, sum = 0.;
    for (size_t i = 0; i < nNonZeros; i++) sum += (values(i) * values(i));
    Teuchos::reduceAll<int, scalar_t>(*comm, Teuchos::REDUCE_SUM, 1, 
                                      &sum, &gsum);
    return sqrt(gsum);
  }
  
  // Query whether bounding box info is available
  inline bool haveBoundingBox() const { return (boundingBox != Teuchos::null); }

  // Return bounding box info in given mode
  // Since the union of all processors' bounding boxes are intended to cover
  // the complete index space of the tensor, this function may return values
  // less than map[m].getNodeMinGlobalIndex and 
  // greater than map[m].getNodeMaxGlobalIndex.
  inline void getLocalIndexRangeForMode(
    const mode_t m, 
    gno_t &min, 
    size_t &len) const 
  {
    if (boundingBox != Teuchos::null) boundingBox->getRangeInMode(m, min, len);
    else throw std::runtime_error("No bounding box information available");
  }

  // Return all bounding box coordinates
  // Since the union of all processors' bounding boxes are intended to cover
  // the complete index space of the tensor, this function may return values
  // less than map[m].getNodeMinGlobalIndex and 
  // greater than map[m].getNodeMaxGlobalIndex.
  inline void getLocalIndexRange(
    std::vector<gno_t> &min, 
    std::vector<size_t> &len) const 
  {
    if (boundingBox != Teuchos::null) boundingBox->getRange(min, len);
    else throw std::runtime_error("No bounding box information available");
  }

  // Printing high-level stats about the tensor; uses AllReduce communication
  void printStats(const std::string &msg, std::ostream &ostr=std::cout) const;

  // Debugging output
  void print(const std::string &msg, std::ostream &ostr = std::cout) const;

private:

  // Constructor functions for building maps and localIndices
  // Can be called in initial constructor or resampling routine
  void buildMaps();        // With no bounding box provided
  void buildMapsWithBB();  // With a bounding box provided

  const Teuchos::RCP<const Teuchos::Comm<int> > comm;

  // tensor storage
  const mode_t nModes;                    // Number of modes; should be 
                                          // modeSizes.size()
  std::vector<size_t> modeSizes;          // range of each mode; indices for
                                          // mode m can be in [0,modeSizes[m]).
                                          // this is a global value

  size_t nNonZeros;                       // Number of local nonzeros in the 
                                          // tensor; may be sampled or not
  size_t nZeros;                          // Number of local sampled zeros in 
                                          // the tensor (for stratified and
                                          // semi-stratified sampling)

  size_t nIndices;                        // Number of local entries
                                          // in tensor; should be <=
                                          // globalIndices.extent(0)

  const gnoview_t globalIndices;          // Nonzeros' and Zeros' indices in 
                                          // coordinate format using global 
                                          // indexing; nonzeros listed first,
                                          // then zeros.
  const lnoview_t localIndices;           // Nonzeros' and Zeros' indices in 
                                          // coordinate format using local 
                                          // indexing wrt map in each mode;
                                          // nonzeros listed first, then zeros
  const valueview_t values;               // Nonzeros' and Zeros' values  

  // tensor distribution
  std::vector<map_t*> maps;           // Map describing tensor distribution 
                                      // in each mode.

  Teuchos::RCP<const SamplingStrategy<sptensor_t> > samplingStrategy;
                                      // Sampling technique used in this tensor
                                      // Can be Teuchos::null for unsampled
                                      // tensors

  Teuchos::RCP<const distSptensorBoundingBox> boundingBox; 
                                      // Tensor bounding box -- 
                                      // in each mode, max and min indices 
                                      // of tensor for which this processor 
                                      // is responsible; needed by Sampled 
                                      // tensors.
                                      // Made boundingBox an RCP so that 
                                      // sampled tensors can use their 
                                      // source tensor's boundingBox.
};

//////////////////////////////////////////////////////////////////////////////
template <typename SCALAR>
distSptensor<SCALAR>::distSptensor(
  mode_t nModes_,
  const std::vector<size_t> &modeSizes_,
  const Kokkos::View<gno_t **> &myGlobalIndices_,
  const Kokkos::View<scalar_t *> &values_,
  const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
  const std::vector<gno_t> &bbMinIndex,
  const std::vector<size_t> &bbModeSizes
) : 
  distSptensor<SCALAR>(nModes_, modeSizes_, myGlobalIndices_, values_, comm_, 
                       myGlobalIndices_.extent(0), 0, bbMinIndex, bbModeSizes)
{}

//////////////////////////////////////////////////////////////////////////////
template <typename SCALAR>
distSptensor<SCALAR>::distSptensor(
  mode_t nModes_,
  const std::vector<size_t> &modeSizes_,
  const Kokkos::View<gno_t **> &myGlobalIndices_,
  const Kokkos::View<scalar_t *> &values_,
  const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
  const size_t numNonzeros_,
  const size_t numZeros_,
  const std::vector<gno_t> &bbMinIndex,
  const std::vector<size_t> &bbModeSizes
) :
  comm(comm_),
  nModes(nModes_),
  modeSizes(nModes),
  nNonZeros(numNonzeros_),
  nZeros(numZeros_),
  nIndices(numNonzeros_+numZeros_),
  globalIndices(myGlobalIndices_),
  localIndices("distSptensor.localIndices", myGlobalIndices_.extent(0),
                                            myGlobalIndices_.extent(1)),
  values(values_),
  maps(nModes, NULL),
  samplingStrategy(Teuchos::null),
  boundingBox(Teuchos::null)
{
  // Build the maps for this Sptensor;
  for (mode_t m = 0; m < nModes; m++) {
    modeSizes[m] = modeSizes_[m];
  }

  // If user provides bounding box information, use it.
  if (static_cast<mode_t>(bbMinIndex.size()) == nModes && 
      static_cast<mode_t>(bbModeSizes.size()) == nModes) {
    boundingBox = 
        Teuchos::rcp(new distSptensorBoundingBox(bbMinIndex, bbModeSizes,
                                                 modeSizes, comm));
  }

  if (boundingBox != Teuchos::null)
    buildMapsWithBB();
  else 
    buildMaps();
}

//////////////////////////////////////////////////////////////////////////////
template <typename SCALAR>
distSptensor<SCALAR>::distSptensor(
  mode_t nModes_,
  const std::vector<size_t> &modeSizes_,
  const Kokkos::View<gno_t **> &myGlobalIndices_,
  const Kokkos::View<scalar_t *> &values_,
  const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
  const size_t numNonzeros_,
  const size_t numZeros_,
  const Teuchos::RCP<const SamplingStrategy<sptensor_t> > &samplingStrategy_,
  const Teuchos::RCP<const distSptensorBoundingBox> &boundingBox_
) :
  comm(comm_),
  nModes(nModes_),
  modeSizes(nModes),
  nNonZeros(numNonzeros_),
  nZeros(numZeros_),
  nIndices(numNonzeros_+numZeros_),
  globalIndices(myGlobalIndices_),
  localIndices("distSptensor.localIndices", myGlobalIndices_.extent(0),
                                            myGlobalIndices_.extent(1)),
  values(values_),
  maps(nModes, NULL),
  samplingStrategy(samplingStrategy_),
  boundingBox(boundingBox_)
{
  // Build the maps for this Sptensor;
  for (mode_t m = 0; m < nModes; m++) {
    modeSizes[m] = modeSizes_[m];
  }

  if (boundingBox != Teuchos::null)
    buildMapsWithBB();
  else 
    buildMaps();
}
/////////////////////////////////////////////////////////////////////////////
template <typename SCALAR>
void distSptensor<SCALAR>::buildMapsWithBB()
{
#ifdef TIME_IMBALANCE
  Teuchos::RCP<Teuchos::Time>
    timeUniquify(Teuchos::TimeMonitor::getNewTimer("SPTENSOR 00 Uniquify GIDS BB")),
    timeReduce(Teuchos::TimeMonitor::getNewTimer("SPTENSOR 00 AllReduce")),
    timeTpetraMap(Teuchos::TimeMonitor::getNewTimer("SPTENSOR 02 TpetraMap Constructor BB"));
#endif

  // Given the set of global indices of nonzeros stored on this processor,
  // build maps describing this distribution of nonzeros.

#ifdef TIME_IMBALANCE
  comm->barrier();
  timeUniquify->start();
#endif

  // create unordered map to remove duplicates and store local indices
  std::vector<std::vector<lno_t> > uniqueIndicesMap(nModes);

  // Need views of unique indices to create Tpetra::Maps
  // Made uniqueIndicesView a std::vector of views because couldn't get
  // Kokkos::subview to work if it was a 2D array of gno_t.
  std::vector<Kokkos::View<gno_t *> > uniqueIndicesView(nModes);

  std::vector<gno_t> bbmin;
  std::vector<size_t> bbsize;
  boundingBox->getRange(bbmin, bbsize);

  for (mode_t m = 0; m < nModes; m++) {

    // Initialize uniqueIndicesView.
    // These lists are likely too large; we'll subview them before sending them
    // to the map constructor.
    char name[24];
    sprintf(name, "uniqueIndicesView%02d", m);
    uniqueIndicesView[m] = Kokkos::View<gno_t *>(name, nIndices);

    // Provide upper-bound on capacity of uniqueIndicesMap[m]
    // TODO:  This value may be way too large.
    uniqueIndicesMap[m].reserve(bbsize[m]);
    for (size_t i = 0; i < bbsize[m]; i++) uniqueIndicesMap[m][i] = -1;
  }

  // Keep track of number of unique gids in each mode
  std::vector<size_t> nunique(nModes, 0);

  for (size_t nz = 0; nz < nIndices; nz++) {

    // remove duplicates of global indices in mode m
    // create list of unique indices to provide to map
    // translate global to local indices.

    for (mode_t m = 0; m < nModes; m++) {

      gno_t gid = globalIndices(nz, m);

      // look for the global index in the map
      auto idx = gid - bbmin[m];

      if (uniqueIndicesMap[m][idx] == -1) {
        // gid has not been seen before in this mode

        lno_t lidx = (nunique[m])++;          // next available local index

        // insert gid into map
        uniqueIndicesMap[m][idx] = lidx;
        uniqueIndicesView[m](lidx) = gid;     // put gid in list of gids for Map
        localIndices(nz, m) = lidx;           // update local index for nonzero
      }
      else {
        // gid exists already in this mode;
        // update local index for nonzero using gid's local value
        localIndices(nz, m) = uniqueIndicesMap[m][idx];
      }
    }
  }

#ifdef TIME_IMBALANCE
  timeUniquify->stop();
  comm->barrier();
  timeReduce->start();
#endif

  // global smallest gid in each mode; these are indexBase for Tpetra::Map
  std::vector<gno_t> gidxbase(nModes); 

  if (maps[0] != NULL) {
    // Resampling -- reuse the index base
    // Not sure if a true min is needed by Tpetra or if a lower 
    // bound sufficies; this is a lower bound
    for (mode_t m = 0; m < nModes; m++) 
      gidxbase[m] = maps[m]->getIndexBase();
  }
  else if (getSourceTensor() != this) {
    // Sampled tensor -- use the source tensor's index base
    // Not sure if a true min is needed by Tpetra or if a lower 
    // bound sufficies; this is a lower bound
    for (mode_t m = 0; m < nModes; m++) 
      gidxbase[m] = getSourceTensor()->getMap(m)->getIndexBase();
  }
  else {
    // Original tensor -- have to reduceAll to get min GIDs for maps
    Teuchos::reduceAll<int, gno_t>(*comm, Teuchos::REDUCE_MIN, nModes, 
                                    &bbmin[0], &gidxbase[0]);
  }

#ifdef TIME_IMBALANCE
  timeReduce->stop();

  comm->barrier();
  timeTpetraMap->start();
#endif

  // Build maps from the unique indices
  for (mode_t m = 0; m < nModes; m++) {

    // Get a subview of just the unique indices for mode m
    typedef typename Kokkos::View<gno_t *>::size_type ksize_t;
    Kokkos::View<gno_t *> resizedUniqueIndicesView = 
       Kokkos::subview(uniqueIndicesView[m],
                       std::pair<ksize_t, ksize_t>(0,nunique[m]));

    // build the map (or rebuild it when resampling)
    if (maps[m] != NULL) delete maps[m];  // delete old maps if resampling 

    const Tpetra::global_size_t INV =
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();
    maps[m] = new map_t(INV, resizedUniqueIndicesView, gidxbase[m], comm);
  }

#ifdef TIME_IMBALANCE
  timeTpetraMap->stop();
#endif
}

/////////////////////////////////////////////////////////////////////////////
template <typename SCALAR>
void distSptensor<SCALAR>::buildMaps()
{
#ifdef TIME_IMBALANCE
  Teuchos::RCP<Teuchos::Time>
    timeUniquify(Teuchos::TimeMonitor::getNewTimer("SPTENSOR 00 Uniquify GIDS")),
    timeReduce(Teuchos::TimeMonitor::getNewTimer("SPTENSOR 01 AllReduce")),
    timeTpetraMap(Teuchos::TimeMonitor::getNewTimer("SPTENSOR 02 TpetraMap Constructor"));
#endif

  // Given the set of global indices of nonzeros stored on this processor,
  // build maps describing this distribution of nonzeros.

#ifdef TIME_IMBALANCE
  timeUniquify->start();
#endif
  // create unordered map to remove duplicates and store local indices
  std::vector<Kokkos::UnorderedMap<gno_t, lno_t> > uniqueIndicesMap(nModes);

  // Need views of unique indices to create Tpetra::Maps
  // Made uniqueIndicesView a std::vector of views because couldn't get
  // Kokkos::subview to work if it was a 2D array of gno_t.
  std::vector<Kokkos::View<gno_t *> > uniqueIndicesView(nModes);

  for (mode_t m = 0; m < nModes; m++) {

    // Initialize uniqueIndicesView.
    // These lists are likely too large; we'll subview them before sending them
    // to the map constructor.
    char name[24];
    sprintf(name, "uniqueIndicesView%02d", m);
    uniqueIndicesView[m] = Kokkos::View<gno_t *>(name, nIndices);

    // Provide upper-bound on capacity of uniqueIndicesMap[m]
    // TODO:  This value may be way too large.
    uniqueIndicesMap[m].rehash(modeSizes[m]);
  }

  // Keep track of number of unique gids in each mode, and min gid in each mode
  std::vector<size_t> nunique(nModes, 0);
  std::vector<gno_t> mingid(nModes, std::numeric_limits<gno_t>::max());

  for (size_t nz = 0; nz < nIndices; nz++) {

    // remove duplicates of global indices in mode m
    // create list of unique indices to provide to map
    // translate global to local indices.

    for (mode_t m = 0; m < nModes; m++) {

      gno_t gid = globalIndices(nz, m);
      if (gid < mingid[m]) mingid[m] = gid;

      // look for the global index in the map
      auto idx = uniqueIndicesMap[m].find(gid);

      if (!uniqueIndicesMap[m].valid_at(idx)) {
        // gid has not been seen before in this mode

        lno_t lidx = (nunique[m])++;          // next available local index

        // insert gid into map; check that insert succeeded
        if (uniqueIndicesMap[m].insert(gid, lidx).failed()) {
          std::cout << "Kokkos::UnorderedMap.insert failed with mode " << m
                    << " gid " << gid << " lid " << lidx << std::endl;
          throw std::runtime_error("Kokkos::UnorderedMap insert failed.");
        }

        uniqueIndicesView[m](lidx) = gid;     // put gid in list of gids for Map
        localIndices(nz, m) = lidx;           // update local index for nonzero
      }
      else {
        // gid exists already in this mode;
        // update local index for nonzero using gid's local value
        localIndices(nz, m) = uniqueIndicesMap[m].value_at(idx);
      }
    }
  }

#ifdef TIME_IMBALANCE
  timeUniquify->stop();
  timeReduce->start();
#endif

  // global smallest gid in each mode; these are indexBase for Tpetra::Map
  std::vector<gno_t> gidxbase(nModes); 
  if (maps[0] != NULL) {
    // Resampling -- reuse the index base
    // Not sure if a true min is needed by Tpetra or if a lower 
    // bound sufficies; this is a lower bound
    for (mode_t m = 0; m < nModes; m++)
      gidxbase[m] = maps[m]->getIndexBase();
  }
  else if (getSourceTensor() != this) {
    // Sampled tensor -- use the source tensor's index base
    // Not sure if a true min is needed by Tpetra or if a lower 
    // bound sufficies; this is a lower bound
    for (mode_t m = 0; m < nModes; m++)
      gidxbase[m] = getSourceTensor()->getMap(m)->getIndexBase();
  }
  else {
    // Original tensor -- have to reduceAll to get min GIDs for maps
    Teuchos::reduceAll<int, gno_t>(*comm, Teuchos::REDUCE_MIN, nModes,
                                   &mingid[0], &gidxbase[0]);
  }

  // Build maps from the unique indices
    
#ifdef TIME_IMBALANCE
  timeReduce->stop();
  timeTpetraMap->start();
#endif

  for (mode_t m = 0; m < nModes; m++) {

    // Get a subview of just the unique indices for mode m
    typedef typename Kokkos::View<gno_t *>::size_type ksize_t;
    Kokkos::View<gno_t *> resizedUniqueIndicesView = 
       Kokkos::subview(uniqueIndicesView[m],
                       std::pair<ksize_t, ksize_t>(0,nunique[m]));

    // build the map (or rebuild it when resampling)
    if (maps[m] != NULL) delete maps[m];  // delete old maps if resampling 

    const Tpetra::global_size_t INV =
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();
    maps[m] = new map_t(INV, resizedUniqueIndicesView, gidxbase[m], comm);
  }
#ifdef TIME_IMBALANCE
  timeTpetraMap->stop();
#endif
}

/////////////////////////////////////////////////////////////////////////////

template <typename SCALAR>
void distSptensor<SCALAR>::printStats(
  const std::string &msg,
  std::ostream &ostr
) const
{
  size_t gmax, gmin, gsum;
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, 1,
                                 &nNonZeros, &gsum);
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MAX, 1,
                                 &nNonZeros, &gmax);
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MIN, 1,
                                 &nNonZeros, &gmin);

  if (comm->getRank() == 0) {
    ostr << std::endl;
    ostr << "SPSTATS Distributed Sparse Tensor: " << msg << std::endl;
    ostr << "SPSTATS   Number of processors: " << comm->getSize() << std::endl;

    ostr << std::endl;
    ostr << "SPSTATS   Number of modes: " << nModes << std::endl;
    ostr << "SPSTATS   Mode sizes:      ";
    for (mode_t m = 0; m < nModes; m++) ostr << modeSizes[m] << " ";
    ostr << std::endl << std::endl;
  }

  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, 1,
                                 &nNonZeros, &gsum);
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MAX, 1,
                                 &nNonZeros, &gmax);
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MIN, 1,
                                 &nNonZeros, &gmin);

  if (comm->getRank() == 0) {
    ostr << "SPSTATS   Global number of nonzeros:  " << gsum << std::endl;
    ostr << "SPSTATS   Max number of nonzeros:     " << gmax << std::endl;
    ostr << "SPSTATS   Min number of nonzeros:     " << gmin << std::endl;

    double gavg = double(gsum) / comm->getSize();
    ostr << "SPSTATS   Avg number of nonzeros:     " << gavg << std::endl;
    if (gavg > 0.)
      ostr << "SPSTATS   Imbalance (max/avg):        " << gmax/gavg<< std::endl;
  }

  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, 1,
                                 &nZeros, &gsum);
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MAX, 1,
                                 &nZeros, &gmax);
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MIN, 1,
                                 &nZeros, &gmin);

  if (comm->getRank() == 0) {
    ostr << "SPSTATS   Global number of zeros:  " << gsum << std::endl;
    ostr << "SPSTATS   Max number of zeros:     " << gmax << std::endl;
    ostr << "SPSTATS   Min number of zeros:     " << gmin << std::endl;

    double gavg = double(gsum) / comm->getSize();
    ostr << "SPSTATS   Avg number of zeros:     " << gavg << std::endl;
    if (gavg > 0.)
      ostr << "SPSTATS   Imbalance (max/avg):     " << gmax/gavg << std::endl;
  }

  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, 1,
                                 &nIndices, &gsum);
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MAX, 1,
                                 &nIndices, &gmax);
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MIN, 1,
                                 &nIndices, &gmin);

  if (comm->getRank() == 0) {
    ostr << "SPSTATS   Global number of indices:  " << gsum << std::endl;
    ostr << "SPSTATS   Max number of indices:     " << gmax << std::endl;
    ostr << "SPSTATS   Min number of indices:     " << gmin << std::endl;

    double gavg = double(gsum) / comm->getSize();
    ostr << "SPSTATS   Avg number of indices:     " << gavg << std::endl;
    if (gavg > 0.)
      ostr << "SPSTATS   Imbalance (max/avg):        " << gmax/gavg<< std::endl;
  }

  // Map stats
  std::vector<size_t> mapSize(nModes);
  for (mode_t m = 0; m < nModes; m++) 
    mapSize[m] = maps[m]->getNodeNumElements();
  
  std::vector<size_t> gvmin(nModes), gvmax(nModes), gvsum(nModes);
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, nModes, 
                                 &(mapSize[0]), &(gvsum[0]));
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MAX, nModes,
                                 &(mapSize[0]), &(gvmax[0]));
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MIN, nModes,
                                 &(mapSize[0]), &(gvmin[0]));

  if (comm->getRank() == 0) {
    ostr << std::endl;
    for (mode_t m = 0; m < nModes; m++) 
      ostr << "SPSTATS   Mode " << m << " map local elements min/max/avg: " 
           << gvmin[m] << " / " << gvmax[m] << " / " 
           << gvsum[m] / comm->getSize()
           << std::endl;
    for (mode_t m = 0; m < nModes; m++) 
      ostr << "SPSTATS   Mode " << m << " global min/max GID "
           << maps[m]->getMinAllGlobalIndex() << " / " 
           << maps[m]->getMaxAllGlobalIndex() 
           << " index base " << maps[m]->getIndexBase()
           << std::endl;
  }
}



//////////////////////////////////////////////////////////////////////////////
template <typename SCALAR>
void distSptensor<SCALAR>::print(
  const std::string &msg,
  std::ostream &ostr
) const
{
  // A horribly serial print function; for debugging only.
  // Please do not use this function in production.

  int me = comm->getRank();
  int np = comm->getSize();

  for (int p = 0; p < np; p++) {
    comm->barrier();  // try to synchronize the output across processors
    if (p == me) {
      if (me == 0) {
        ostr << "Distributed Sparse Tensor: " << msg << std::endl
             << "    Number of Modes: " << nModes << std::endl
             << "    Mode Sizes: ";
        for (mode_t m = 0; m < nModes; m++) ostr << modeSizes[m] << " ";
        ostr << std::endl << std::endl;
      }
      ostr << me << "   Number of local nonzeros: " << nNonZeros << std::endl;
      ostr << me << "   Number of local zeros:    " << nZeros << std::endl;
      ostr << me << "   Number of local indices:  " << nIndices << std::endl;
      for (size_t j = 0; j < nIndices; j++) {
        ostr << me << "    (";
        for (mode_t i = 0; i < nModes; i++) ostr << globalIndices(j, i) << " ";
        ostr << ") (";
        for (mode_t i = 0; i < nModes; i++) ostr << localIndices(j, i) << " ";
        ostr << ") = " << values(j) << std::endl;
      }
      ostr << std::endl;
    }
  }

  if (haveBoundingBox()) boundingBox->print();
}

}
#endif
