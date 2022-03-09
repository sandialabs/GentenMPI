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
 
#ifndef DISTFROMGENERATORS_
#define DISTFROMGENERATORS_

#include "pt_test_generators.hpp"
#include "pt_ktensor.hpp"
#include "pt_sptensor.hpp"
#include <numeric>


//////////////////////////////////////////////////////////////////////////////
// Build a distributed Ktensor and initialize its values 
// using the Ktensor generator

template <typename ktensorGenerator_t>
class generatedDistKtensor {
public:

  typedef typename ktensorGenerator_t::gno_t gno_t;
  typedef typename ktensorGenerator_t::scalar_t scalar_t;

  typedef typename pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename pt::distKtensor<factormatrix_t> ktensor_t;

  generatedDistKtensor(
    KtensorGenerator<scalar_t> &ktensorGenerator,
    const Teuchos::RCP<const Teuchos::Comm<int> > &comm, bool norm = true)
  { 
    // Build ktensor with same global dimensions/sizes as ktensorGenerator
    pt::rank_t rank = ktensorGenerator.getFactorRank();
    pt::mode_t nModes = ktensorGenerator.getNumModes();

    std::vector<size_t> modeSizes(nModes);
    for (pt::mode_t m = 0; m < nModes; m++) 
      modeSizes[m] = ktensorGenerator.getSize(m);

    ktensor = new ktensor_t(rank, modeSizes, comm);

    // Initialize local entries with values from ktensorGenerator
    for (pt::mode_t m = 0; m < nModes; m++) {

      factormatrix_t *factormatrix = ktensor->getFactorMatrix(m);
      typename factormatrix_t::valueview_t data = 
                                                factormatrix->getLocalView();

      size_t len = factormatrix->getLocalLength();
      for (size_t i = 0; i < len; i++) {
        gno_t gidx = factormatrix->getMap()->getGlobalElement(i);
        for (pt::rank_t r = 0; r < rank; r++) {
          data(i, r) = ktensorGenerator.getGlobalValue(m, gidx, r);
        }
      }
    }

    if (norm) ktensor->normalize();
  }

  ~generatedDistKtensor() { delete ktensor; }

  inline void print(const std::string &msg) { ktensor->print(msg); }

  inline ktensor_t *getKtensor() { return ktensor; }

private:
  ktensor_t *ktensor;
};

//////////////////////////////////////////////////////////////////////////////
// Build a distributed Sptensor and initialize its values 
// using the Sptensor generator

template <typename sptensorGenerator_t>
class generatedDistSptensor {
public:

  typedef typename sptensorGenerator_t::gno_t gno_t;
  typedef typename sptensorGenerator_t::scalar_t scalar_t;

  typedef typename pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename pt::distKtensor<factormatrix_t> ktensor_t;
  typedef typename pt::distSptensor<scalar_t> sptensor_t;

  generatedDistSptensor(
    sptensorGenerator_t &sptensorGenerator,
    ktensor_t *ktensor,
    const Teuchos::RCP<const Teuchos::Comm<int> > &comm,
    int verbosity_=1) : verbosity(verbosity_)
  {
    pt::mode_t nModes = ktensor->getNumModes();
    std::vector<size_t> modeSizes(nModes);
    for (pt::mode_t m = 0; m < nModes; m++) 
      modeSizes[m] = ktensor->getModeSize(m);

    // Build distribution of processors to the tensor
    std::vector<int> nProc(nModes);    // Number of procs in each mode; 
                                       // product over modes = comm->getSize()
    std::vector<int> myProc(nModes);   // Index of my proc in the 
                                       // nMode-dimension layout
    buildProcLayout(nProc, myProc, modeSizes, comm);

    // Build distribution of the tensor to the processors
    std::vector<size_t> myModeSizes(nModes, 0); 
                                       // Length of each mode on my proc
    std::vector<gno_t> myFirstGid(nModes, 0);   
                                       // First GID for which my proc is 
                                       // responsible in each mode
    buildTensorLayout(myModeSizes, myFirstGid, nProc, myProc, modeSizes, comm);
 
    // Set up data to convert from nz index to mode-based index
    std::vector<size_t> prefixSizes(nModes, 1);
    size_t maxNNZ = myModeSizes[0];
    for (pt::mode_t m = 1; m < nModes; m++) {
      maxNNZ *= myModeSizes[m];
      prefixSizes[m] = prefixSizes[m-1] * myModeSizes[m-1];
    }

    // Count the actual nonzeros in my block
    size_t myNNZ = 0;
    std::vector<gno_t> idx(nModes);

    for (size_t nz = 0; nz < maxNNZ; nz++) {
      for (pt::mode_t m = 0; m < nModes; m++) {
        idx[m] = myFirstGid[m] + ((nz / prefixSizes[m]) % myModeSizes[m]);
      }
      if (sptensorGenerator.getGlobalValue(idx) != 0.) myNNZ++;
    }
    
    // Allocate storage for nonzeros
    Kokkos::View<gno_t **> myIndices("myindices", myNNZ, nModes);
    Kokkos::View<scalar_t *> myValues("myvalues", myNNZ);

    // Set indices and values for nonzeros to match generated ones
    myNNZ = 0;
    for (size_t nz = 0; nz < maxNNZ; nz++) {

      for (pt::mode_t m = 0; m < nModes; m++) 
        idx[m] = myFirstGid[m] + ((nz / prefixSizes[m]) % myModeSizes[m]);

      scalar_t value = sptensorGenerator.getGlobalValue(idx);
      if (value != 0.) {
        for (pt::mode_t m = 0; m < nModes; m++) myIndices(myNNZ, m) = idx[m];
        myValues(myNNZ) = value;
        myNNZ++;
      }

    }

    // Create the distributed Sptensor, including bounding box
    sptensor = new sptensor_t(nModes, ktensor->getModeSizes(),
                              myIndices, myValues, comm,
                              myFirstGid, myModeSizes);
  }
  
  ~generatedDistSptensor() { delete sptensor; }

  inline void print(const std::string &msg) { sptensor->print(msg); }

  inline sptensor_t *getSptensor() { return sptensor; }

private:
  int verbosity;

  sptensor_t *sptensor;
                                 
  void buildProcLayout(
    std::vector<int> &nProc,           // Number of procs in each mode; 
                                       // product over modes = comm->getSize()
    std::vector<int> &myProc,          // Index of my proc in the 
                                       // nMode-dimension layout
    std::vector<size_t> &modeSizes,    // Global modeSizes
    const Teuchos::RCP<const Teuchos::Comm<int> > &comm
  )
  {
    // Create a nMode-dimensional cartesion processor layout
    // Assign processors to modes in decreasing size of modes
    int me = comm->getRank();
    int np = comm->getSize();

    // Sort the modes in decreasing order, creating index sorted
    pt::mode_t nModes = modeSizes.size();
    std::vector<size_t> sorted(nModes);
    std::iota(sorted.begin(), sorted.end(), 0);
    auto comparator = [&modeSizes](pt::mode_t a, pt::mode_t b){ 
         return modeSizes[a] > modeSizes[b]; 
    };
    std::sort(sorted.begin(), sorted.end(), comparator);

    // Sum the modes
  
    size_t totalModes = modeSizes[0];
    for (pt::mode_t m = 1; m < nModes; m++) totalModes += modeSizes[m];

    // Determine number of procs in each mode
    size_t procProd = 1;
    int npRemaining = np;

    for (pt::mode_t m = 0; m < nModes-1; m++) {
      pt::mode_t mode = sorted[m];

      // get good guess at nProc[mode], proportional to modeSizes[mode]
      int guess = std::ceil((double)npRemaining *
                            ((double)modeSizes[mode]/(double)totalModes));
      nProc[mode] = std::max(1, guess);  // Have to have at least one

      // find nearest divisor of npRemaining
      int tmpUp = nProc[mode];
      int tmpDown = (nProc[mode] > 1 ? nProc[mode]-1 : 1);
      while (npRemaining % tmpUp) tmpUp++;
      while (npRemaining % tmpDown) tmpDown--;
 
      nProc[mode] = ((tmpUp-nProc[mode]) <= (nProc[mode]-tmpDown) ? tmpUp 
                                                                  : tmpDown);
      procProd *= nProc[mode];
      npRemaining /= nProc[mode];
      totalModes -= modeSizes[mode];
    }

    // give last dimension the remaining procs
    nProc[sorted[nModes-1]] = np / procProd;

    // Compute myProc index in the layout
    procProd = 1;
    for (pt::mode_t m = 0; m < nModes; m++) {
      myProc[m] = (me / procProd) % nProc[m];
      procProd *= nProc[m];
    }

    if (verbosity > 0) {
      std::cout << me << " NPROC/MODE ";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << nProc[m] << " ";
      std::cout << std::endl;
      std::cout << me << " MYPROC ";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << myProc[m] << " ";
      std::cout << std::endl;
    }
  }

  void buildTensorLayout(
    std::vector<size_t> &myModeSizes,  // Length of each mode on my proc
    std::vector<gno_t> &myFirstGid,    // First GID for which my proc is 
                                       // responsible in each mode
    std::vector<int> &nProc,           // Number of procs in each mode; 
                                       // product over modes = comm->getSize()
    std::vector<int> &myProc,          // Index of my proc in the 
                                       // nMode-dimension layout
    std::vector<size_t> &modeSizes,    // Global modeSizes
    const Teuchos::RCP<const Teuchos::Comm<int> > &comm
  )
  {
    // Assign blocks of the tensor to the processor layout
    // Compute number of potential GIDs in each mode for this proc
    // Compute first GID in each mode for this proc

    pt::mode_t nModes = modeSizes.size();
    for (pt::mode_t m = 0; m < nModes; m++) {

      int frac = int(double(modeSizes[m]) / double(nProc[m])); 
      int mod = modeSizes[m] % nProc[m];
      int remainder = ((myProc[m] < mod) ? 1 : 0);
      int remainderOffset = std::min(mod, myProc[m]);

      myModeSizes[m] = frac + remainder;
      myFirstGid[m] = myProc[m] * frac + remainderOffset;
     
      if (verbosity > 0) {
        std::cout << comm->getRank() << " TENSORLAYOUT mode " << m 
                  << " mysize " << myModeSizes[m] 
                  << " myfirst " << myFirstGid[m] << std::endl;
      }
    }
  }
};

#endif
