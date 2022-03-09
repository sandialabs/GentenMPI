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
 
#ifndef READBYBOXES_HPP
#define READBYBOXES_HPP

#include "pt_sptensor.hpp"

#ifdef __cplusplus
/* if C++, include these header files as extern C */
extern "C" {
#endif
#define restrict __restrict__
#include "splatt_mpi.h"
#include "splatt.h"
#include "util.h"

#ifdef __cplusplus
/* if C++, include these header files as extern C */
}
#endif

// KDD Had to copy from SPLATT because this one function is static.  Ugh
static void p_get_best_mpi_dim(
  rank_info * const rinfo)
{
  int nprimes = 0;
  int * primes = get_primes(rinfo->npes, &nprimes);

  idx_t total_size = 0;
  for(idx_t m=0; m < rinfo->nmodes; ++m) {
    total_size += rinfo->global_dims[m];

    /* reset mpi dims */
    rinfo->dims_3d[m] = 1;
  }
  idx_t target = total_size / (idx_t)rinfo->npes;

  long diffs[MAX_NMODES];

  /* start from the largest prime */
  for(int p = nprimes-1; p >= 0; --p) {
    int furthest = 0;
    /* find dim furthest from target */
    for(idx_t m=0; m < rinfo->nmodes; ++m) {
      /* distance is current - target */
      idx_t const curr = rinfo->global_dims[m] / rinfo->dims_3d[m];
      /* avoid underflow */
      diffs[m] = (curr > target) ? (curr - target) : 0;

      if(diffs[m] > diffs[furthest]) {
        furthest = m;
      }
    }

    /* assign p processes to furthest mode */
    rinfo->dims_3d[furthest] *= primes[p];
  }

  free(primes);
}



// Let SPLATT do the reading and simple (naive) decomposition
// Compute desired uniform bounding boxes for each processor
// Assign indices to boxes
// Let SPLATT move the data

template <typename ptsptensor_t>
ptsptensor_t *readWithUniformBlocks(
  const std::string &filename, 
  const Teuchos::RCP<const Teuchos::Comm<int> > &comm
) 
{
  int me = comm->getRank();

  if (me == 0)
    std::cout << "Using readWithUniformBlocks " << filename << std::endl;

  // Make sure tensor file exists
  FILE *fin;
  if ((fin = fopen(filename.c_str(), "r")) == NULL) {
    if (me == 0) std::cout << "Error:  file " << filename
                           << " is not readable " << std::endl;
    exit(-1);
  }
  fclose(fin);

  // Use SPLATT to read tensor with naive distribution
  typedef sptensor_t splatt_sptensor_t;

  const Teuchos::MpiComm<int> *tmpicomm =
               dynamic_cast<const Teuchos::MpiComm<int> *>(comm.getRawPtr());
  MPI_Comm mpicomm = *(tmpicomm->getRawMpiComm());

  splatt_sptensor_t *simpleSplatt = mpi_simple_distribute(filename.c_str(),
                                                          mpicomm);

  typedef rank_info splatt_rank_info_t;
  splatt_rank_info_t rinfo;
  rinfo.rank = comm->getRank();
  rinfo.npes = comm->getSize();
  rinfo.decomp = (splatt_decomp_type) SPLATT_DECOMP_MEDIUM; // set & ignore
  rinfo.nmodes = simpleSplatt->nmodes;

  int nModes = simpleSplatt->nmodes;
  Teuchos::reduceAll<int, idx_t>(*comm, Teuchos::REDUCE_SUM, 1,
                                 &(simpleSplatt->nnz), &(rinfo.global_nnz));
  Teuchos::reduceAll<int, idx_t>(*comm, Teuchos::REDUCE_MAX, nModes,
                                 simpleSplatt->dims, rinfo.global_dims);

  // Find best medium grain processor layout a la SPLATT
  p_get_best_mpi_dim(&rinfo);
  mpi_setup_comms(&rinfo);

  // Set layer starts and layer ends
  std::vector<size_t> approxIdxPerProc(nModes);
  std::vector<int> remainIdxPerProc(nModes);

  for (int m = 0; m < nModes; m++) {
    int me = rinfo.coords_3d[m];
    int np = rinfo.dims_3d[m];
    size_t nidx = rinfo.global_dims[m];
    approxIdxPerProc[m] = nidx / np;
    remainIdxPerProc[m] = nidx % np;
    size_t myNumIdx = approxIdxPerProc[m] + (me < remainIdxPerProc[m]);
    size_t myFirstIdx = me * approxIdxPerProc[m] 
                      + std::min(me, remainIdxPerProc[m]);
    rinfo.layer_starts[m] = myFirstIdx;
    rinfo.layer_ends[m] = myFirstIdx + myNumIdx;
  }

  // create partition based on uniform boxes
  std::vector<int> parts(simpleSplatt->nnz);
  std::vector<int> procCoords(nModes);

  for (size_t n = 0; n < simpleSplatt->nnz; n++) {
    for (int m = 0; m < nModes; m++) {
      size_t approx = approxIdxPerProc[m];
      int remain = remainIdxPerProc[m];
      size_t idx = simpleSplatt->ind[m][n];
      int estProc = idx / approx;
      while (idx < (estProc * approx + std::min(estProc, remain)))
        estProc--;
      procCoords[m] = estProc;
    }
    MPI_Cart_rank(rinfo.comm_3d, &procCoords[0], &parts[n]);
  }

  // Move and reindex the data a la Splatt
  splatt_sptensor_t *boxedSplatt = 
    mpi_rearrange_by_part(simpleSplatt, (simpleSplatt->nnz ? &parts[0] : NULL),
                          rinfo.comm_3d);

  for (int m = 0; m < nModes; m++) {
    boxedSplatt->dims[m] = rinfo.layer_ends[m] - rinfo.layer_starts[m];
    for(size_t n = 0; n < boxedSplatt->nnz; n++) {
      assert(boxedSplatt->ind[m][n] >= rinfo.layer_starts[m]);
      assert(boxedSplatt->ind[m][n] < rinfo.layer_ends[m]);
      boxedSplatt->ind[m][n] -= rinfo.layer_starts[m];
    }
  }
  
  // Clean up simple splatt tensor
  tt_free(simpleSplatt);

  // Convert splatt tensor to DistSpTensor
  ptsptensor_t *newTensor = splattToPT<ptsptensor_t>(boxedSplatt, &rinfo,
                                                     false, comm);

  // Clean up splatt tensor
  tt_free(boxedSplatt);

  return newTensor;
}
#endif
