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
 
#ifndef PT_SAMPLERS_
#define PT_SAMPLERS_

#include "pt_sptensor.hpp"
#include "pt_tensorhash.hpp"


namespace pt {

// Forward declaration
template<typename scalar_t>
class distSptensor;

// Base class for sampling functions
// All sampling functions can derive from this base class.
template <typename sptensor_t>
class SamplingStrategy 
{
public:

  typedef typename sptensor_t::gno_t gno_t;
  typedef typename sptensor_t::scalar_t scalar_t;
  static constexpr scalar_t zero = 0.;

  //////  To create your own SamplingStrategy...
  //////
  //////  If desired, implement constructor specific to the sampling technique
  //////  Otherwise, implement minimal constructor that invokes this one
  SamplingStrategy(sptensor_t * tensorToSample) 
  : originalSptensor(tensorToSample)
  { }

  //////  Required:  Indicate whether sampling is 
  //////    stratified (zeros sampled are guaranteed to be zeros), or
  //////    semi-stratified (zeros sampled may actually be nonzeros or zeros)
  virtual bool isStratified() const = 0;

  //////  Required:  Provide a name for the sampling strategy
  virtual const std::string name() const = 0;

  //////  Required:  Implement sampling specific to the sampling technique
  virtual void sampleTheTensor( 
    size_t nRequestedNonzeros,   // in: # of samples of nonzeros to 
                                 //     draw on this proc
    size_t nRequestedZeros,      // in: # of samples of zeros to 
                                 //     draw on this proc
    size_t &nSampledNonzeros,    // out: # of samples of nonzeros actually
                                 //      drawn on this processor
    size_t &nSampledZeros,       // out: # of samples of zeros actually
                                 //      drawn on this processor
    const Kokkos::View<gno_t **> &inds,
                                    // in/out: global indices of drawn samples;
                                    //         must already be allocated
                                    //         Storing both nonzeros and
                                    //         zeros explicitly
    const Kokkos::View<scalar_t *> &vals,
                                    // in/out: values of drawn samples;
                                    //         must already be allocated
                                    //         Storing both nonzeros and
                                    //         zeros explicitly
    unsigned int seed = 1           // in: used to set random seed
  ) const = 0;

  //////  Use these base class methods directly; no need to reimplement

  const sptensor_t * getSourceTensor() const { return originalSptensor; }

  //////  Virtual destructor needed
  virtual ~SamplingStrategy() = default;

protected:

  // Look for errors in the bounding box or nonzeros that would cause 
  // problems for sampling.  May want to reduce the amount of checking later.
  void checkBoundingBox()
  {
    if (!(originalSptensor->haveBoundingBox())) {
      std::cout << "Error for " << this->name() << std::endl;
      throw std::runtime_error("This sampling requires tensor "
                               "to be constructed with a bounding box.");
    }
    if (!(originalSptensor->canSampleTensor())) {
      std::cout << "Error for " << this->name() << std::endl;
      throw std::runtime_error("Bounding box is "
                               "not good for sampling");
    }

    // Make sure bounding box volume is not zero if the processor has nonzeros
    size_t nNonZeros = originalSptensor->getLocalNumNonZeros();
    if (originalSptensor->getLocalTensorSize() == 0 && nNonZeros) {
      std::cout << "Error on rank " << originalSptensor->getComm()->getRank()
                << " for " << this->name() 
                << ":  Bounding box has size zero but  "
                << "processor has nonzeros" << std::endl;
      throw std::runtime_error("Bounding box has size zero but  "
                               "processor has nonzeros");
    }

    // Make sure all nonzeros are in the bounding box
    mode_t nModes = originalSptensor->getNumModes();
    auto globalIndices = originalSptensor->getGlobalIndices();
    std::vector<gno_t> bbMinGid;
    std::vector<size_t> bbModeSizes;
    originalSptensor->getLocalIndexRange(bbMinGid, bbModeSizes);

    size_t ierr = 0;
    for (size_t i = 0; i < nNonZeros; i++) {
      for (mode_t m = 0; m < nModes; m++) {
        typename sptensor_t::gno_t gid = globalIndices(i,m);
        if (gid < bbMinGid[m] || 
            gid >= bbMinGid[m]+static_cast<gno_t>(bbModeSizes[m]))
          ierr++;
      }
    }
    if (ierr > 0) {
      std::cout << "Error on rank " << originalSptensor->getComm()->getRank()
                << " for " << this->name() << ": " << ierr 
                << " nonzeros are outside the bounding box" << std::endl;
      throw std::runtime_error("Nonzeros found outside bounding box");
    }
  }

  sptensor_t * const originalSptensor;
};

///////////////////////////////////////////////////////////////////////////
// Semi-stratified sampling function
template <typename sptensor_t>
class SemiStratifiedSamplingStrategy : public SamplingStrategy<sptensor_t>
{
public:
  typedef typename SamplingStrategy<sptensor_t>::gno_t gno_t;
  typedef typename SamplingStrategy<sptensor_t>::scalar_t scalar_t;

  SemiStratifiedSamplingStrategy(sptensor_t *sptensor)
  : SamplingStrategy<sptensor_t>(sptensor)
  {
    // KDD Technically, could do semi-stratified sampling without a bounding
    // KDD box.  Since we don't need to check whether a sampled zero is really
    // KDD a nonzero, we can sample anywhere in the domain.  However, sampling
    // KDD within a bounding box will likely result in less communication.
    // KDD And our error computations assume bounding-box sizes in scaling
    // KDD zero and nonzero contributions to error.
    // KDD So we'll require a bounding box for semi-strat sampling.
    this->checkBoundingBox();
  }

  inline bool isStratified() const { return false; }

  inline const std::string name() const { return "semiStratified"; }

  void sampleTheTensor(
    size_t nRequestedNonzeros,
    size_t nRequestedZeros,  
    size_t &nSampledNonzeros,
    size_t &nSampledZeros,
    const Kokkos::View<gno_t **> &inds,
    const Kokkos::View<scalar_t *> &vals,
    unsigned int seed = 1
  ) const
  {
    const sptensor_t * const sptensor = this->getSourceTensor();

    // Set random seed based on processor id
    srand( seed * 12345 * (sptensor->getComm()->getRank()+1) );
  
    gno_t s;

    // sample nonzeros uniformly
    nSampledNonzeros = 0;
    size_t nNonZeros = sptensor->getLocalNumNonZeros();
    mode_t nModes = sptensor->getNumModes();

    if (nNonZeros) { // can sample nonzeros only if have some
      auto globalIndices = sptensor->getGlobalIndices();
      auto values = sptensor->getValues();

      for (size_t i = 0; i < nRequestedNonzeros; i++) {
        // choose random nonzero index
        s = rand() % nNonZeros;
        for (mode_t m = 0; m < nModes; m++) {
          inds(i,m) = globalIndices(s,m);
        }
        vals(i) = values(s);
        nSampledNonzeros++;
      }
    }
  
    // sample entries uniformly from tensor, assume they are zero
    nSampledZeros = 0;
    if (sptensor->getLocalTensorSize() > 0) { // can sample indices 
                                              // only if have some
      std::vector<gno_t> bbMinGid;
      std::vector<size_t> bbModeSizes;
      sptensor->getLocalIndexRange(bbMinGid, bbModeSizes);
      for (size_t i = 0; i < nRequestedZeros; i++) {
        for (mode_t m = 0; m < nModes; m++) {
          // choose random index for mode m
          s = rand() % bbModeSizes[m];
          inds(i+nSampledNonzeros,m) = bbMinGid[m] + s;
        }
        vals(i+nSampledNonzeros) = this->zero;
        nSampledZeros++;
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////
// Stratified sampling function
template <typename sptensor_t>
class StratifiedSamplingStrategy : public SamplingStrategy<sptensor_t>
{
public:
  typedef typename SamplingStrategy<sptensor_t>::gno_t gno_t;
  typedef typename SamplingStrategy<sptensor_t>::scalar_t scalar_t;

  typedef TensorHashMap<scalar_t,gno_t> hashmap_t;

  ~StratifiedSamplingStrategy() { delete nonzerosMap; }

  StratifiedSamplingStrategy(sptensor_t *sptensor)
  : SamplingStrategy<sptensor_t>(sptensor),
    nonzerosMap(NULL)
  {
    this->checkBoundingBox();

    // Stratified sampling needs a hash map of the sptensor's global indices
    auto globalIndices = sptensor->getGlobalIndices();
    auto values = sptensor->getValues();
    size_t nNonZeros = sptensor->getLocalNumNonZeros();
    mode_t nModes = sptensor->getNumModes();
    
    nonzerosMap = new hashmap_t(nModes, nNonZeros);
    // insert global indices into hash map
    for (size_t i = 0; i < nNonZeros; i++)
      nonzerosMap->insert(subview(globalIndices, i, Kokkos::ALL),values(i));
  }

  inline bool isStratified() const { return true; }

  inline const std::string name() const { return "stratified"; }

  void sampleTheTensor(
    size_t nRequestedNonzeros,
    size_t nRequestedZeros,  
    size_t &nSampledNonzeros,
    size_t &nSampledZeros,
    const Kokkos::View<gno_t **> &inds,
    const Kokkos::View<scalar_t *> &vals,
    unsigned int seed = 1
  ) const
  {
    const sptensor_t * const sptensor = this->getSourceTensor();
    gno_t s;
  
    // Set random seed based on processor id
    srand( seed * 12345 * (sptensor->getComm()->getRank()+1) );
  
    // sample nonzeros uniformly
    nSampledNonzeros = 0;
    size_t nNonZeros = sptensor->getLocalNumNonZeros();
    mode_t nModes = sptensor->getNumModes();

    if (nNonZeros) { // can sample nonzeros only if have some
      auto globalIndices = sptensor->getGlobalIndices();
      auto values = sptensor->getValues();

      for (size_t i = 0; i < nRequestedNonzeros; i++) {
        // choose random nonzero index
        s = rand() % nNonZeros;
        for (mode_t m = 0; m < nModes; m++) {
          inds(i,m) = globalIndices(s,m);
        }
        vals(i) = values(s);
        nSampledNonzeros++;
      }
    }
  
    // sample entries uniformly from tensor, make sure they are zero
    nSampledZeros = 0;
    if (sptensor->getLocalTensorSize() > 0) { // can sample indices 
                                              // only if have some
      std::vector<gno_t> bbMinGid;
      std::vector<size_t> bbModeSizes;
      sptensor->getLocalIndexRange(bbMinGid, bbModeSizes);

      bool found;
      for (size_t i = 0; i < nRequestedZeros; i++) {
        found = true;
        size_t idx = i + nSampledNonzeros;
        while(found) {
          for (mode_t m = 0; m < nModes; m++) {
            // choose random index for mode m
            s = rand() % bbModeSizes[m];
            inds(idx,m) = bbMinGid[m] + s;
          } 
          found = nonzerosMap->exists(subview(inds, idx, Kokkos::ALL));    
        }
        vals(idx) = 0;
        nSampledZeros++;
      }
    }
  }

private:
  hashmap_t *nonzerosMap;       // Hashmap of global indices for fast 
                                // determination if entry is nonzero,
                                // used for stratified sampling
};

///////////////////////////////////////////////////////////////////////////
// For testing:  sampling function that returns the entire tensor, 
// including both zeros and nonzeros
// Note that the resulting tensor will be dense!  
// You'll hate yourself if you try it on big tensors.
template <typename sptensor_t>
class FullTensorSamplingStrategy : public SamplingStrategy<sptensor_t>
{
public:
  typedef typename SamplingStrategy<sptensor_t>::gno_t gno_t;
  typedef typename SamplingStrategy<sptensor_t>::scalar_t scalar_t;

  typedef TensorHashMap<scalar_t,gno_t> hashmap_t;

  ~FullTensorSamplingStrategy() { delete nonzerosMap; }

  FullTensorSamplingStrategy(sptensor_t *sptensor)
  : SamplingStrategy<sptensor_t>(sptensor),
    nonzerosMap(NULL)
  {
    if (sptensor->getNumModes() > 4) {
      throw std::runtime_error("FullTensorSamplingStrategy is not supported "
                               "for tensors with more than four modes.");
    }

    this->checkBoundingBox();

    // FullTensor sampling needs a hash map of the sptensor's global indices
    auto globalIndices = sptensor->getGlobalIndices();
    auto values = sptensor->getValues();
    size_t nNonZeros = sptensor->getLocalNumNonZeros();
    mode_t nModes = sptensor->getNumModes();
    
    nonzerosMap = new hashmap_t(nModes, nNonZeros);
    // insert global indices into hash map
    for (size_t i = 0; i < nNonZeros; i++)
      nonzerosMap->insert(subview(globalIndices, i, Kokkos::ALL),values(i));
  }

  // fullTensor sampling is stratified
  inline bool isStratified() const { return true; }  

  inline const std::string name() const { return "fullTensor"; }

  void sampleTheTensor(
    size_t nRequestedNonzeros,
    size_t nRequestedZeros,  
    size_t &nSampledNonzeros,
    size_t &nSampledZeros,
    const Kokkos::View<gno_t **> &inds,
    const Kokkos::View<scalar_t *> &vals,
    unsigned int seed = 1
  ) const
  {
    // This sampling ignores the requested number of nonzeros and zeros,
    // and instead returns the entire sptensor, including all nonzeros and 
    // zeros.
    // It assumes the inds and vals arrays are big enough.
    const sptensor_t * const sptensor = this->getSourceTensor();
  
    // Copy all nonzeros
    nSampledNonzeros = 0;
    size_t nNonZeros = sptensor->getLocalNumNonZeros();
    mode_t nModes = sptensor->getNumModes();

    auto globalIndices = sptensor->getGlobalIndices();
    auto values = sptensor->getValues();
    for (size_t i = 0; i < nNonZeros; i++) {
      for (mode_t m = 0; m < nModes; m++) {
        inds(i,m) = globalIndices(i,m);
      }
      vals(i) = values(i);
      nSampledNonzeros++;
    }
  
    // Insert all zeros
    // Loop over bounding box dimensions, test whether index is a nonzero,
    // if not, insert it as a zero
    std::vector<gno_t> bbMinGid;
    std::vector<size_t> bbModeSizes;
    sptensor->getLocalIndexRange(bbMinGid, bbModeSizes);

    nSampledZeros = 0;
    size_t idx = nSampledNonzeros;
    double localSize = sptensor->getLocalTensorSize();

    switch (nModes) {
    case 2:
      for (size_t i = 0; i < bbModeSizes[0] && idx < localSize; i++) {
        size_t i_idx = i + bbMinGid[0];
        for (size_t j = 0; j < bbModeSizes[1] && idx < localSize; j++) {
          size_t j_idx = j + bbMinGid[1];
          inds(idx, 0) = i_idx;
          inds(idx, 1) = j_idx;
          if (!(nonzerosMap->exists(subview(inds, idx, Kokkos::ALL)))) {
            vals(idx) = this->zero;
            idx++;
            nSampledZeros++;
          }
        }
      }
      break;
    case 3:
      for (size_t i = 0; i < bbModeSizes[0] && idx < localSize; i++) {
        size_t i_idx = i + bbMinGid[0];
        for (size_t j = 0; j < bbModeSizes[1] && idx < localSize; j++) {
          size_t j_idx = j + bbMinGid[1];
          for (size_t k = 0; k < bbModeSizes[2] && idx < localSize; k++) {
            size_t k_idx = k + bbMinGid[2];
            inds(idx, 0) = i_idx;
            inds(idx, 1) = j_idx;
            inds(idx, 2) = k_idx;
            if (!(nonzerosMap->exists(subview(inds, idx, Kokkos::ALL)))) {
              vals(idx) = this->zero;
              idx++;
              nSampledZeros++;
            }
          }
        }
      }
      break;
    case 4:
      for (size_t i = 0; i < bbModeSizes[0] && idx < localSize; i++) {
        size_t i_idx = i + bbMinGid[0];
        for (size_t j = 0; j < bbModeSizes[1] && idx < localSize; j++) {
          size_t j_idx = j + bbMinGid[1];
          for (size_t k = 0; k < bbModeSizes[2] && idx < localSize; k++) {
            size_t k_idx = k + bbMinGid[2];
            for (size_t p = 0; p < bbModeSizes[3] && idx < localSize; p++) {
              size_t p_idx = p + bbMinGid[3];
              inds(idx, 0) = i_idx;
              inds(idx, 1) = j_idx;
              inds(idx, 2) = k_idx;
              inds(idx, 3) = p_idx;
              if (!(nonzerosMap->exists(subview(inds, idx, Kokkos::ALL)))) {
                vals(idx) = this->zero;
                idx++;
                nSampledZeros++;
              }
            }
          }
        }
      }
      break;
    }

    // Sanity check
    if (nSampledZeros + nSampledNonzeros != sptensor->getLocalTensorSize()) {
      throw std::runtime_error("FullTensorSamplingStrategy failed; "
                               "nSamples != local tensor size");
    }
  }

private:
  hashmap_t *nonzerosMap;       // Hashmap of global indices for fast 
                                // determination if entry is nonzero,
                                // used for stratified sampling
};


}
#endif
