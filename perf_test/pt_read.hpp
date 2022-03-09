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
 
#include "pt_sptensor.hpp"
#include "pt_rebalance.hpp"

#ifdef __cplusplus
/* if C++, include these header files as extern C */
extern "C" {
#endif
#define restrict __restrict__
#include "splatt_mpi.h"
#include "splatt.h"

#ifdef __cplusplus
/* if C++, include these header files as extern C */
}
#endif


// Convert a distributed SPLATT tensor to a PT tensor
// May be a way to use Kokkos::Views with the SPLATT memory, but
// probably safer to copy the SPLATT data and then free it.

template <typename ptsptensor_t>
static ptsptensor_t *splattToPT(
  sptensor_t *splattTensor,
  rank_info *rinfo, 
  int rebalance, 
  const Teuchos::RCP<const Teuchos::Comm<int> > &comm
)
{
  typedef typename ptsptensor_t::gnoview_t gnoview_t;
  typedef typename ptsptensor_t::valueview_t valueview_t;
  
  pt::mode_t nModes = splattTensor->nmodes;

  // Need to compute global max id to get global mode sizes
  std::vector<size_t> maxid(nModes, 0);

  // indices
  size_t nnz = splattTensor->nnz;


  gnoview_t globalIndices("splattGlobalIndices", nnz, nModes);
  for (size_t n = 0; n < nnz; n++) {
    for (pt::mode_t m = 0; m < nModes; m++) {
      splatt_idx_t id = splattTensor->ind[m][n] + rinfo->layer_starts[m];
      globalIndices(n, m) = id;
      if (id > maxid[m]) maxid[m] = id;
    }
  }
    
  std::vector<typename ptsptensor_t::gno_t> bbMinGid(nModes);
  std::vector<size_t> bbModeSizes(nModes);

  for (pt::mode_t m = 0; m < nModes; m++) {
    bbMinGid[m] = rinfo->layer_starts[m];
    bbModeSizes[m] = rinfo->layer_ends[m] - rinfo->layer_starts[m];
  }

  // values
  valueview_t values("splattValues", nnz);
  for (size_t n = 0; n < nnz; n++) {
    values(n) = splattTensor->vals[n];
  }
  
  // mode sizes -- get global maxid
  std::vector<size_t> modeSizes(nModes);
  Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MAX, nModes, 
                                 &(maxid[0]), &(modeSizes[0]));
  // SPLATT returns ids using base 0, so max+1 gives mode size.
  for (pt::mode_t m = 0; m < nModes; m++) modeSizes[m]++;

  if (rebalance)
    zoltanRebalance<gnoview_t, valueview_t>(globalIndices, values, comm,
                                            rinfo->dims_3d);

  ptsptensor_t *newTensor = new ptsptensor_t(nModes, modeSizes, globalIndices,
                                             values, comm,
                                             bbMinGid, bbModeSizes);

  return newTensor;
}

template <typename ptsptensor_t>
ptsptensor_t *readUsingSplattIO(
  std::string &filename,   // COO-formatted file with tensor nonzeros
  std::string &distribute, // string with nModes entries IxJxKx... giving
                           // processor layout
  int rebalance,
  const Teuchos::RCP<const Teuchos::Comm<int> > &comm
)
{
  if (comm->getRank() == 0) 
    std::cout << "Using readUsingSplattIO " << filename << std::endl;

  // First, use SPLATT to read file and distribute tensor into a SPLATT tensor;
  // then convert SPLATT tensor to DistSpTensor.

  rank_info rinfo;
  rinfo.rank = comm->getRank();
  rinfo.npes = comm->getSize();
  rinfo.decomp = (splatt_decomp_type) DEFAULT_MPI_DISTRIBUTION;

  // Parse the processor layout in string distribute
  // std::istringstream iss(distribute);
  // std::string token;
  // splatt_idx_t d = 0;
  // while (std::getline(iss, token, 'x')) rinfo.dims_3d[d++] = std::stoi(token);
  // for (; d < SPLATT_MAX_NMODES; d++) {
  //   if (d == 0) rinfo.dims_3d[d] = comm->getSize();
  //   else        rinfo.dims_3d[d] = 1;
  // }

  // SPLATT picks its configuration automatically when DEFAULT_MPI_DISTRIBUTION
  // For now, set all dims_3d to one.
  for (splatt_idx_t d = 0; d < SPLATT_MAX_NMODES; d++) rinfo.dims_3d[d] = 1;

  // Call splatt reader; mostly copied from splatt:src/cmds/mpi_cmd_cpd.c
  sptensor_t *tt = splatt_mpi_tt_read(filename.c_str(), NULL, &rinfo);
  if(tt == NULL) {
    std::ostringstream sMsg;
    sMsg << "SPLATT file read failed " << filename << std::endl;
    throw std::runtime_error(sMsg.str());
  }

  // Convert splatt tensor to DistSpTensor
  ptsptensor_t *newTensor = splattToPT<ptsptensor_t>(tt, &rinfo, 
                                                     rebalance, comm);

  // Clean up splatt tensor
  tt_free(tt);

  return newTensor;
}
