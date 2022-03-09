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
 
#ifndef PT_SPTENSOR_BOUNDINGBOX_
#define PT_SPTENSOR_BOUNDINGBOX_

#include "pt_shared.h"
#include "Teuchos_CommHelpers.hpp"

namespace pt {

// Tensor bounding box -- in each mode, max and min indices of tensor for 
// which this processor is responsible.  
// A processor's bounding box includes all globalIndices on the processor.
// It may also include global indices for which the processor does not have
// nonzeros.
// 
// Currently, bounding boxes are required only for sampled tensors, 
// but we may someday exploit them in determining factormatrix maps in 
// distSptensor.
// 
// The union of all processors' tensor bounding boxes must cover the entire
// index space of the tensor.
//
// For correct functionality in distSptensor, the intersection of the 
// bounding boxes may or may not be null; distSptensor allows bounding boxes
// to be overlapping.
// However, for stratified sampling to work correctly, bounding boxes must
// not overlap, as the test to determine whether a sampled zero is actually
// a nonzero depends on having a single processor store all nonzeros from a 
// box's index space.
//
// TODO:  Perhaps remove comm from class and pass it only to functions that
// TODO:  use it (e.g., overlapping()) to emphasize that they do communication

class distSptensorBoundingBox
{
public:

  typedef pt::global_ordinal_type gno_t;

  // Constructor 
  distSptensorBoundingBox(
               const std::vector<gno_t> &bbMinIndex_,
               const std::vector<size_t> &bbModeSizes_,
               const std::vector<size_t> &modeSizes_,
               const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
    bbMinIndex(bbMinIndex_),
    bbModeSizes(bbModeSizes_),
    boxSize(0.),
    modeSizes(modeSizes_),
    comm(comm_)
  {
    // error check the bounding box:  must have values within modeSizes
    bool ok = true;
    mode_t nModes = modeSizes.size();

    for (mode_t m = 0; m < nModes; m++) {

      gno_t bbidx = bbMinIndex[m];
      gno_t bblen = static_cast<gno_t>(bbModeSizes[m]);
      gno_t modelen = static_cast<gno_t>(modeSizes[m]);

      if (bblen > 0) {

        if (bblen > modelen) {
          std::cout << comm->getRank() << " Error in mode " << m 
                    << ": bounding box mode size " << bblen
                    << " > tensor mode size " << modelen << std::endl;
          ok = false;
        }

        if (bbidx < 0) {
          std::cout << comm->getRank() << " Error in mode " << m
                    << ": negative bounding box min index provided; "
                    << " min = " << bbidx << std::endl;
          ok = false;
        }

        if (bbidx+bblen-1 >= modelen) {
          std::cout << comm->getRank() << " Error in mode " << m
                    << ": bounding box min index exceeds mode size"
                    << " min = " << bbidx
                    << " modesize = " << modelen << std::endl;
          ok = false;
        }
      }
    }

    if (!ok) throw std::runtime_error("distSptensor bounding box failed");

    boxSize = 1.;
    for (size_t m = 0; m < bbMinIndex.size(); m++) boxSize *= bbModeSizes[m];
  }

  // Return bounding box coordinates in given mode
  inline void getRangeInMode(const mode_t m, gno_t &min, size_t &num) const {
    min = bbMinIndex[m];
    num = bbModeSizes[m];
  }

  // Return all bounding box coordinates
  inline void getRange(std::vector<gno_t> &min, std::vector<size_t> &num) const
  {
    min = bbMinIndex;
    num = bbModeSizes;
  }

  // Return number of (nonzero or zero) indices in local bounding box
  inline double getBoxSize() const { return boxSize; }

  // Return true if the bounding boxes in the communicator are good for 
  // stratified sampling; that is, the bounding boxes should not overlap
  // and should cover the entire index range of tensor
  // This function requires collective communication operations
  inline bool goodForSampling() const {
#if 0
// Coverage test here worked well when boxSize and gBoxSize were size_t.
// But with both being double, round-off error can cause the test to fail.
// For now, I'll disable it.  
// TODO Add a better coverage test.
    double gBoxSize;
    Teuchos::reduceAll<int, double>(*comm, Teuchos::REDUCE_SUM, 1,
                                    &boxSize, &gBoxSize);

    double prod = 1.;
    size_t nModes = modeSizes.size();
    for (size_t m = 0; m < nModes; m++) prod *= modeSizes[m];
    
    return((std::abs(gBoxSize-prod) < 1.) && !overlapping());
#endif
    return !overlapping();
  }
 
  // Return true if bounding boxes in communicator overlap, false otherwise
  // Stratified sampling requires non-overlapping bounding boxes
  // This function requires collective communication operations
  // TODO:  If we end up calling overlapping() frequently, we should move 
  // TODO:  this code to the constructor and save the result, rather than
  // TODO:  doing the communication in each call.
  bool overlapping() const {

    int me = comm->getRank();
    int np = comm->getSize();

    mode_t nModes = bbMinIndex.size();
    if (nModes == 0) return false;

    // Get all processors' bounding boxes
    size_t allLen = np * nModes;
    std::vector<gno_t> allMin(allLen);
    Teuchos::gatherAll<int, gno_t>(*comm, nModes, &bbMinIndex[0], 
                                    allLen, &allMin[0]);

    std::vector<size_t> allModeSizes(allLen);
    std::vector<size_t> emptyBox(nModes, 0);  // Empty boxes don't overlap
    Teuchos::gatherAll<int, size_t>(*comm, nModes, 
                                    (boxSize ? &bbModeSizes[0] : &emptyBox[0]), 
                                     allLen, &allModeSizes[0]);

    // Check whether my box overlaps with any others
    int overlapping = false;

    if (boxSize) { // only non-empty boxes look for overlap
      for (int p = 0; p < np && !overlapping; p++) {

        if (p == me) continue;

        int pidx = p * nModes;
        if (allModeSizes[pidx] == 0) continue;  // empty box -- skip it

        mode_t noverlapmodes = 0;
        for (mode_t m = 0; m < nModes; m++) {
          gno_t myMin = bbMinIndex[m];
          gno_t myMax = myMin + bbModeSizes[m] - 1;
          gno_t pMin = allMin[pidx+m];
          gno_t pMax = pMin + allModeSizes[pidx+m] - 1;
          if ((myMin >= pMin && myMin <= pMax) || 
              (myMax >= pMin && myMax <= pMax))
            noverlapmodes++;
        } 
        if (noverlapmodes == nModes)
          overlapping = true;
      }
    }

    // Gather global result
    int gOverlapping;
    Teuchos::reduceAll<int, int>(*comm, Teuchos::REDUCE_MAX, 1,
                                 &overlapping, &gOverlapping);
    return gOverlapping;
  }


  // Print method for debugging
  void print() const {
    std::cout << comm->getRank() << " bounding box: (";
    for (size_t m = 0; m < bbMinIndex.size(); m++)
      if (bbModeSizes[m] > 0)
        std::cout << bbMinIndex[m] << " ";
      else 
        std::cout << "- ";
    std::cout << ") -> (";
    for (size_t m = 0; m < bbMinIndex.size(); m++)
      if (bbModeSizes[m] > 0)
        std::cout << bbMinIndex[m]+bbModeSizes[m]-1 << " ";
      else 
        std::cout << "- ";
    std::cout << ")" << std::endl;
  }

private:
  const std::vector<gno_t> bbMinIndex;
  const std::vector<size_t> bbModeSizes;
  double boxSize;  // Product of bbModeSizes
  const std::vector<size_t> modeSizes;
  const Teuchos::RCP<const Teuchos::Comm<int> > comm;
};

}
#endif
