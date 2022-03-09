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
 
#ifndef TTBFROMGENERATORS_
#define TTBFROMGENERATORS_

#include "pt_test_generators.hpp"
#include <TTB_Ktensor.h>
#include <TTB_Sptensor.h>

//////////////////////////////////////////////////////////////////////////////
// Build a TTB Ktensor and initialize its values 
// using the Ktensor generator

template <typename ktensorGenerator_t>
class generatedTTBKtensor {
public:

  typedef ttb_indx lno_t;
  typedef ttb_indx gno_t;
  typedef ttb_real scalar_t;
  typedef typename TTB::Ktensor ktensor_t;

  generatedTTBKtensor(ktensorGenerator_t &ktensorGenerator, bool norm = true)
  { 
    // Build ktensor with same global dimensions/sizes as ktensorGenerator
    pt::rank_t rank = ktensorGenerator.getFactorRank();
    pt::mode_t nModes = ktensorGenerator.getNumModes();

    ktensor = new ktensor_t(rank, nModes);

    TTB::IndxArray dims;
    dims.resize(nModes);

    for (pt::mode_t m = 0; m < nModes; m++) {
      dims[m] = ktensorGenerator.getSize(m);
    }
    ktensor->setMatrices(dims, rank, 0.0);

    // Initialize local entries with values from ktensorGenerator
    for (pt::mode_t m = 0; m < nModes; m++) {
      for (size_t i = 0; i < dims[m]; i++) {
        for (pt::rank_t r = 0; r < rank; r++) {
          (*ktensor)[m].entry(i,r) = ktensorGenerator.getGlobalValue(m, i, r);
        }
      }
    }
    if (norm) ktensor->normalize(TTB::NormTwo);
  }

  ~generatedTTBKtensor() { delete ktensor; }

  inline ktensor_t *getKtensor() { return ktensor; }

private:
  ktensor_t *ktensor;
};

//////////////////////////////////////////////////////////////////////////////
// Build a TTB Sptensor and initialize its values 
// using the Sptensor generator

template <typename sptensorGenerator_t>
class generatedTTBSptensor {
public:

  typedef typename sptensorGenerator_t::lno_t lno_t;
  typedef typename sptensorGenerator_t::gno_t gno_t;
  typedef typename sptensorGenerator_t::scalar_t scalar_t;

  typedef typename TTB::Ktensor ktensor_t;
  typedef typename TTB::Sptensor sptensor_t;

  generatedTTBSptensor(
    sptensorGenerator_t &sptensorGenerator,
    ktensor_t *ktensor,
    int verbosity_=1) : verbosity(verbosity_)
  {

    pt::mode_t nModes = sptensorGenerator.getNumModes();

    std::vector<ttb_indx> dims(nModes);
    for (pt::mode_t m = 0; m < nModes; m++)
      dims[m] = sptensorGenerator.getSize(m);

    // Set up data to convert from nz index to mode-based index
    std::vector<size_t> prefixSizes(nModes, 1);
    size_t maxNNZ = dims[0];
    for (pt::mode_t m = 1; m < nModes; m++) {
      maxNNZ *= dims[m];
      prefixSizes[m] = prefixSizes[m-1] * dims[m-1];
    }

    // Count the actual nonzeros 
    ttb_indx NNZ = 0;
    std::vector<gno_t> idx(nModes);

    for (size_t nz = 0; nz < maxNNZ; nz++) {
      for (pt::mode_t m = 0; m < nModes; m++) {
        idx[m] = (nz / prefixSizes[m]) % dims[m];
      }
      if (sptensorGenerator.getGlobalValue(idx) != 0.) NNZ++;
    }

    // Allocate storage for nonzeros
    std::vector<double> values(NNZ);
    std::vector<ttb_indx> subscripts(NNZ*nModes);
    
    // Set indices and values for nonzeros to match generated ones
    NNZ = 0;
    for (size_t nz = 0; nz < maxNNZ; nz++) {

      for (pt::mode_t m = 0; m < nModes; m++) 
        idx[m] = (nz / prefixSizes[m]) % dims[m];

      ttb_real value = sptensorGenerator.getGlobalValue(idx);
      if (value != 0.) {
        for (pt::mode_t m = 0; m < nModes; m++) 
          subscripts[NNZ*nModes+m] = idx[m];
        values[NNZ] = value;
        NNZ++;
      }

    }

    // Create the TTB Sptensor
    sptensor = new sptensor_t(nModes, &(dims[0]), NNZ,
                              &(values[0]), &(subscripts[0]));
  }
  
  ~generatedTTBSptensor() { delete sptensor; }

  inline sptensor_t *getSptensor() { return sptensor; }

private:
  int verbosity;
  sptensor_t *sptensor;
                                 
};

#endif
