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
 
#ifndef DISTFROMRANDOM_
#define DISTFROMRANDOM_

#include "pt_ktensor.hpp"
#include "pt_sptensor.hpp"

//////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
class CDF {

  CDF(Kokkos::View<scalar_t *> lambda) :
    distValues("distValues", lambda.extent(0))
  {
    size_t len = lambda.extent(0);
    distValues(0) = lambda(0);
    for (size_t i = 1; i < len; i++) 
      distValues(i) = lambda(i) + distValues(i-1);
  }

  CDF(factormatrix_t *fm, pt::rank_t r) :
    distValues("distValues", fm->getModeSize())
  {

  }

private:
  Kokkos::View<scalar_t *> distValues;
};


//////////////////////////////////////////////////////////////////////////////
// onMyProc -- determine a layout for nonzeros among processors

template <typename ktensor_t>
class procDist {
public:
  procDist(const ktensor_t &ktensor) {
    int np = comm->getSize();
    nModes = ktensor->getNumModes();
    std::sort<>(ktensor->getModeSizes(), sortedModes, descending);
    std::vector<int> nProcPerMode(nModes);
    double maxGNZ = 1.;
    for (int m = 0; m < nModes; m++) maxGNZ *= ktensor->getModesSizes()(m);
    for (int m = 0, npProd = 1, remNp = np; m < nModes-1; m++) {
      int tmp = ceil(pow(
      nProcPerMode[m] = ceil(np * double(sortedModes(m)) / maxGNZ);
      npProd *= nProcPerMode[m];
    }
    nProcPerMode[nModes-1] = np / npProd;
  }
};



//////////////////////////////////////////////////////////////////////////////
// Build a random distributed sptensor

template <typename sptensor_t, typename ktensor_t>
sptensor_t *randomSptensor(
  size_t maxnnz,
  ktensor_t *ktensor,
  const Teuchos::RCP<const Teuchos::Comm<int> > &comm
) 
{
  pt::mode_t nModes = ktensor->getNumModes();
  pt::rank_t rank = ktensor->getFactorRank();

  // set ktensor to uniform values
  ktensor->setUniformRandom();

  // compute distribution of ktensor's lambda weights 
  pt::CDF weightDist(ktensor->getLambda());

  // compute number of samples from each rank
  std::vector<size_t> nSamplesPerRank(rank, 0);
  srand48(12345.);
  for (size_t nz = 0; nz < maxnnz, nz++) {
    double rnum = drand48();
    nSamplesPerRank[weightDist.getRandomSample(rnum)]++;
  }

  // Allocated CDF for each factor matrix and maps to track local coordinates
  pt::CDF *fmDist = new pt::CDF[nModes];
  
  for (pt::rank_t r = 0; r < rank; r++) {

    for (pt::mode_t m = 0; m < nModes; m++)
      fmDist[m].load(ktensor->getFactorMatrix(m), r);

    for (size_t nz = 0; nz < nSamplesPerRank[r]; nz++) {
      gno_t *coord = new gno_t[nModes];
      for (pt::mode_t m = 0; m < nModes; m++) {
        rnum = drand48();
        coord[m] = fmDist[m].getRandomSample(rnum);
      }

      if (onMyProc(coord)) {
        if ((auto it = coordMap.find(coord)) == coordMap.end()) {
          coordMap.insert(std::pair<gno_t *, int>(coord, 1));
        }
        else {
          it->second += 1;
          delete [] coord;
        }
      }
    }
  }

  // Form sparse tensor from the coordinates in the map
  Kokkos::View<gno_t **> myGlobalIndices("randomGlobalIndices",
                                          coordMap.size(), nModes);
  Kokkos::View<scalar_t *> myValues("randomValues", coordMap.size());
  size_t myCnt = 0;
  for (auto it = coordMap.begin(); it != coordMap.end(); it++) {
    gno_t *coord = it->first;
    for (pt::mode_t m = 0; m < nModes; m++) 
      myGlobalIndices(myCnt, m) = coord[m];
    myValues(myCnt) = it->second;
    delete it->first;
    myCnt++;
  }
  sptensor_t *sptensor = new sptensor_t(nModes, ktensor->getModeSizes(),
                                        myGlobalIndices, myValues, comm);

  return sptensor;
}

#endif
