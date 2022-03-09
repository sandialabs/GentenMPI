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
 
// Unit test for distSptensorBoundingBox

#include "pt_sptensor_boundingbox.hpp"
#include <Tpetra_Core.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>

class testBoundingBox {

public:

  typedef pt::global_ordinal_type gno_t;

  // Constructor:  initializes values
  testBoundingBox(const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
    comm(comm_),
    me(comm->getRank()),
    np(comm->getSize()),
    nModes(4),
    modeSizes(nModes),
    bbmin(nModes),
    bbmodesize(nModes)
  { 
    // Tiny little tensor
    modeSizeProduct = 1;
    for (pt::mode_t m = 0; m < nModes; m++) {
      modeSizes[m] = N * np * (m+1);
      modeSizeProduct *= modeSizes[m];
    }
  }

  // How to run the tests within testBoundingBox
  int run();

  // Output for debugging
  void printBox() {
    std::cout << "  " << comm->getRank() << "  Box: (";
    for (pt::mode_t m = 0; m < nModes; m++)
      std::cout << bbmin[m] << " ";
    std::cout << ") -> (";
    for (pt::mode_t m = 0; m < nModes; m++)
      std::cout << (bbmodesize[m] ? bbmin[m]+gno_t(bbmodesize[m])-1 : -2) 
                << " ";
    std::cout << ")" << std::endl;
  }

private:
  const Teuchos::RCP<const Teuchos::Comm<int> > comm;
  const int me;
  const int np;
  static const size_t N = 2;

  const pt::mode_t nModes;
  std::vector<size_t> modeSizes;
  size_t modeSizeProduct;

  std::vector<gno_t> bbmin;
  std::vector<size_t> bbmodesize;
}; 


////////////////////////////////////////////////////////////////////////////////
// How to run the tests within testBoundingBox
int testBoundingBox::run()
{
  int ierr = 0;

  {
    // Create a bounding box with bad indices on process zero
    // Rank zero (and only process zero) should catch an error;
    // if not, the test fails.
  
    if (me == 0) std::cout << "TEST BAD INDICES" << std::endl;

    if (me == 0) {
      bbmin[0] = 0;  
      bbmodesize[0] = 0;  // test one empty mode size
      for (pt::mode_t m = 1; m < nModes; m++) {
        bbmin[m] = -1;            // test invalid indices
        bbmodesize[m] = modeSizes[m];
      }
    }
    else {
      for (pt::mode_t m = 0; m < nModes; m++) {
        bbmin[m] = 0;
        bbmodesize[m] = modeSizes[m]-1;
      }
    }
    printBox();
  
    bool caught = 0;
    try {
      pt::distSptensorBoundingBox bb(bbmin, bbmodesize, modeSizes, comm);
    }
    catch (std::exception &e) {
      if (me == 0) {
        std::cout << "  CORRECT:  BAD INDEX TEST THREW AN ERROR" << std::endl;
        caught = 1;
      }
      else {
        std::cout << "  FAIL:  BAD INDEX TEST THREW UNEXPECTED ERROR " 
                  << e.what() << std::endl;
        ierr++;
      }
    }
    if (me == 0 && caught == 0) {
      std::cout << "  FAIL:  BAD INDICES NOT DETECTED IN BAD INDEX TEST "
                << std::endl;
      ierr++;
    }
  }

  {
    // Create overlapping bounding boxes -- identical on all procs; 
    // overlapping except when np == 1.
    // good for sampling only when np == 1; 
    // overlapping boxes are not good for sampling.

    if (me == 0) std::cout << "TEST IDENTICAL BOXES" << std::endl;

    for (pt::mode_t m = 0; m < nModes; m++) {
      bbmin[m] = 0;
      bbmodesize[m] = modeSizes[m];
    }
    printBox();

    pt::distSptensorBoundingBox bb(bbmin, bbmodesize, modeSizes, comm);

    bool overlapping = bb.overlapping();
    if (me == 0) {
      if (((np == 1) && !overlapping) || ((np > 1) && overlapping)) {
        std::cout << "  CORRECT:  OVERLAP TEST WITH IDENTICAL BOXES" 
                  << std::endl;
      }
      else {
        std::cout << "  FAIL:  OVERLAP TEST WITH IDENTICAL BOXES" << std::endl;
        ierr++;
      }
    }

    bool goodForSampling = bb.goodForSampling();
    if (me == 0) {
      if (((np == 1) && goodForSampling) || ((np > 1) && !goodForSampling)) {
        std::cout << "  CORRECT:  SAMPLING TEST WITH IDENTICAL BOXES"
                  << std::endl;
      }
      else {
        std::cout << "  FAIL:  SAMPLING TEST WITH IDENTICAL BOXES" << std::endl;
        ierr++;
      }
    }
  }

  {
    // Create overlapping bounding boxes -- different on all procs; 
    // overlapping except when np == 1.
    // not good for sampling because boxes do not cover the entire domain and
    // boxes overlap.

    if (me == 0) std::cout << "TEST DIFFERING BOXES" << std::endl;

    for (pt::mode_t m = 0; m < nModes; m++) {
      bbmin[m] = me * (m+1);
      bbmodesize[m] = m+2;
    }
    printBox();

    pt::distSptensorBoundingBox bb(bbmin, bbmodesize, modeSizes, comm);

    bool overlapping = bb.overlapping();
    if (me == 0) {
      if (((np == 1) && !overlapping) || ((np > 1) && overlapping)) {
        std::cout << "  CORRECT:  OVERLAP TEST WITH DIFFERING BOXES" 
                  << std::endl;
      }
      else {
        std::cout << "  FAIL:  OVERLAP TEST WITH DIFFERING BOXES" << std::endl;
        ierr++;
      }
    }

    bool goodForSampling = bb.goodForSampling();
    if (me == 0) {
      if (!goodForSampling) {
        std::cout << "  CORRECT:  SAMPLING TEST WITH DIFFERING BOXES"
                  << std::endl;
      }
      else {
        std::cout << "  FAIL:  SAMPLING TEST WITH DIFFERING BOXES" << std::endl;
        ierr++;
      }
    }
  }

  {
    // Create non-overlapping bounding boxes; check for overlapping
    // non-overlapping for all np.
    // with np == 1, good for sampling, as non-overlapping and box covers
    // entire domain.
    // with np > 1, not good for sampling because boxes do not cover the 
    // entire domain 

    if (me == 0) std::cout << "TEST NONOVERLAP BOXES" << std::endl;

    for (pt::mode_t m = 0; m < nModes; m++) {
      bbmin[m] = me * (m+1) * N;
      bbmodesize[m] = (m+1) * N;
    }
    printBox();

    pt::distSptensorBoundingBox bb(bbmin, bbmodesize, modeSizes, comm);

    bool overlapping = bb.overlapping();
    if (me == 0) {
      if (!overlapping) {
        std::cout << "  CORRECT:  NONOVERLAP TEST " << std::endl;
      }
      else {
          std::cout << "  FAIL:  NONOVERLAP TEST " << std::endl;
        ierr++;
      }
    }

    bool goodForSampling = bb.goodForSampling();
    if (me == 0) {
      if ((np == 1 && goodForSampling) || (np > 1 && !goodForSampling)) {
        std::cout << "  CORRECT:  SAMPLING TEST WITH NONOVERLAP BOXES"
                  << std::endl;
      }
      else {
        std::cout << "  FAIL:  SAMPLING TEST WITH NONOVERLAP BOXES" 
                  << std::endl;
        ierr++;
      }
    }
  }

  {
    // Create non-overlapping bounding boxes with one empty box; 
    // non-overlapping for all np
    // not good for sampling because boxes do not cover the domain

    if (me == 0) std::cout << "TEST NONOVERLAP WITH EMPTY BOX" << std::endl;

    if (me == comm->getSize()-1) {
      for (pt::mode_t m = 0; m < nModes; m++) {
        bbmin[m] = 0;
        bbmodesize[m] = 0;
      }
    }
    else {
      for (pt::mode_t m = 0; m < nModes; m++) {
        bbmin[m] = me * (m+1) * N;
        bbmodesize[m] = (m+1) * N;
      }
    }
    printBox();

    pt::distSptensorBoundingBox bb(bbmin, bbmodesize, modeSizes, comm);

    bool overlapping = bb.overlapping();
    if (me == 0) {
      if (!overlapping) {
        std::cout << "  CORRECT:  NONOVERLAP WITH EMPTY BOX TEST " << std::endl;
      }
      else {
          std::cout << "  FAIL:  NONOVERLAP WITH EMPTY BOX TEST " << std::endl;
        ierr++;
      }
    }

    bool goodForSampling = bb.goodForSampling();
    if (me == 0) {
      if (!goodForSampling) {
        std::cout << "  CORRECT:  SAMPLING TEST WITH EMPTY BOX" << std::endl;
      }
      else {
        std::cout << "  FAIL:  SAMPLING TEST WITH EMPTY BOX" << std::endl;
        ierr++;
      }
    }
  }

  {
    // Create non-overlapping bounding boxes that cover the tensor index space
    // non-overlapping for all np
    // good for sampling because nonoverlapping and cover the tensor index space

    if (me == 0) std::cout << "TEST GOOD FOR SAMPLING" << std::endl;

    for (pt::mode_t m = 0; m < nModes-1; m++) {
        bbmin[m] = 0;
        bbmodesize[m] = modeSizes[m];
    }
    // split last mode among processors
    int lastModeSize = modeSizes[nModes-1];
    bbmin[nModes-1] = me * (lastModeSize / np) + 
                      std::min<int>(me, lastModeSize % np);
    bbmodesize[nModes-1] = (lastModeSize / np) + (me < (lastModeSize % np));
    printBox();

    pt::distSptensorBoundingBox bb(bbmin, bbmodesize, modeSizes, comm);

    bool overlapping = bb.overlapping();
    if (me == 0) {
      if (!overlapping) {
        std::cout << "  CORRECT:  NONOVERLAP GOOD FOR SAMPLING" << std::endl;
      }
      else {
          std::cout << "  FAIL:  NONOVERLAP GOOD FOR SAMPLING " << std::endl;
        ierr++;
      }
    }

    bool goodForSampling = bb.goodForSampling();
    if (me == 0) {
      if (goodForSampling) {
        std::cout << "  CORRECT:  SAMPLING TEST GOOD FOR SAMPLING" << std::endl;
      }
      else {
        std::cout << "  FAIL:  SAMPLING TEST GOOD FOR SAMPLING" << std::endl;
        ierr++;
      }
    }
  }

  return ierr;
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

int main(int narg, char **arg)
{
  Tpetra::ScopeGuard scopeguard(&narg, &arg);
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int me = comm->getRank();

  int ierr = 0;

  {
    if (me == 0) 
      std::cout << std::endl << "TESTING " << std::endl;
    testBoundingBox test(comm);
    ierr += test.run();
  }

  if (ierr) 
    std::cout << me << ":  " << ierr << " errors detected." << std::endl;

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
