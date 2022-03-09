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
 
#ifndef TESTGENERATORS_
#define TESTGENERATORS_

#include "pt_factormatrix.hpp"

//////////////////////////////////////////////////////////////////////////////
// Class to generate Ktensor values according to a fixed formula

template <typename SC>
class KtensorGenerator
{
public:

  typedef SC scalar_t;
  typedef typename pt::distFactorMatrix<SC>::lno_t lno_t;
  typedef typename pt::distFactorMatrix<SC>::gno_t gno_t;

  KtensorGenerator(pt::mode_t nModes_, std::vector<size_t> &modeSizes_, 
                   pt::rank_t rank_) :
    nModes(nModes_), modeSizes(modeSizes_), rank(rank_)
  {}

  inline pt::rank_t getFactorRank() const { return rank; }
  
  inline pt::mode_t getNumModes() const { return nModes; }

  inline size_t getSize(pt::mode_t m) const { return modeSizes[m]; }

  scalar_t getGlobalValue(pt::mode_t m, gno_t idx, pt::rank_t r) const
  { return m * 10 + (idx+1) * std::pow(m+1, r); }

  void print(const std::string &msg) 
  {
    for (pt::mode_t m = 0; m < nModes; m++) {
      std::cout << msg << " mode " << m << std::endl;
      for (size_t i = 0; i < modeSizes[m]; i++) {
        std::cout << "IDX " << i << ": ";
        for (pt::rank_t r = 0; r < rank; r++)
          std::cout << getGlobalValue(m, i, r) << " ";   
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }

private:
  pt::mode_t nModes;
  std::vector<size_t> modeSizes;
  pt::rank_t rank;
};

//////////////////////////////////////////////////////////////////////////////
// Class to generate sptensor values according to a fixed formula

template <typename SC>
class SptensorGenerator
{
public:
  typedef SC scalar_t;
  typedef typename pt::distFactorMatrix<SC>::lno_t lno_t;
  typedef typename pt::distFactorMatrix<SC>::gno_t gno_t;

  typedef typename std::vector<gno_t> nonzero_t;

  SptensorGenerator(pt::mode_t nModes_, std::vector<size_t> &modeSizes_, 
                    pt::rank_t rank_, double density_) :
    nModes(nModes_), modeSizes(modeSizes_), rank(rank_),
    iteratorIdx(nModes, -1), prefixSizes(nModes, 1), maxNNZ(0), iteratorNZ(-1),
    density(density_)
  {
    maxNNZ = modeSizes[0];
    for (pt::mode_t m = 1; m < nModes; m++) {
      maxNNZ *= modeSizes[m];
      prefixSizes[m] = prefixSizes[m-1] * modeSizes[m-1];
    }
  }

  nonzero_t getFirstNonzero() 
  {
    iteratorNZ = -1;
    return getNextNonzero();
  }

  nonzero_t getNextNonzero() 
  { 
    iteratorNZ++;
    for (; iteratorNZ < maxNNZ; iteratorNZ++) {
      for (pt::mode_t m = 0; m < nModes; m++) {
        iteratorIdx[m] = (iteratorNZ / prefixSizes[m]) % modeSizes[m];
      }
      scalar_t val = getGlobalValue(iteratorIdx);
      if (val == 0.) continue;
      return iteratorIdx;
    }
    for (pt::mode_t m = 0; m < nModes; m++)
      iteratorIdx[m] = invalid;
    return iteratorIdx;
  }

  inline gno_t getIndex(nonzero_t nz, pt::mode_t mode) const { return nz[mode]; }

  inline bool validNonzero(nonzero_t nz) const { return !(nz[0] == invalid); }

  scalar_t getGlobalValue(nonzero_t &index) const { 
    scalar_t value = 0;
    for (pt::mode_t m = 0; m < nModes; m++) value += index[m];
    srand48(value);  // Set random number generator consistently for index
    for (gno_t i = 0; i < index[0]; i++) drand48();// skip some random numbers
    return ((drand48() <= density) ? value : 0.);
  }

  inline int64_t getMaxNNZ() const { return maxNNZ; }

  inline pt::mode_t getNumModes() const { return nModes; }

  inline size_t getSize(pt::mode_t m) const { return modeSizes[m]; }

  void print(const std::string &msg)
  {
    nonzero_t nz = getFirstNonzero();
    while (validNonzero(nz)) {
      std::cout << msg << " IDX: ";
      for (pt::mode_t m = 0; m < nModes; m++) 
        std::cout << nz[m] << " ";
      std::cout << "VAL: " << getGlobalValue(nz) << std::endl;
      nz = getNextNonzero();
    }
    std::cout << std::endl;
  }

private:
  static const gno_t invalid = -1;
  pt::mode_t nModes;
  std::vector<size_t> modeSizes;
  pt::rank_t rank;

  nonzero_t iteratorIdx;
  std::vector<size_t> prefixSizes;
  int64_t maxNNZ;
  int64_t iteratorNZ;
  double density;
};

#endif
