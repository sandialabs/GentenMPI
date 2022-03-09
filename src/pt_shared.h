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
 

#ifndef PT_SHARED_
#define PT_SHARED_

#include <iostream>
#include <stdexcept>
#include <limits>
#include "Teuchos_Comm.hpp"
#include "Kokkos_Core.hpp"
#include "Tpetra_Map.hpp"


#define DBGASSERT(cond, msg) if (!(cond)) throw std::runtime_error(msg);

namespace pt {
  typedef int mode_t;    // Data type for number of modes; needs to be signed
  typedef int rank_t;    // Data type for rank of factor matrices

  // Data layout used in factor matrices and other dense data structures
#ifdef PT_LAYOUTRIGHT
  // pt_lrmv*.hpp:  LayoutRightMultiVector
  typedef Kokkos::LayoutRight layout_t;
#else
  // Tpetra::MultiVector default
  typedef Kokkos::LayoutLeft layout_t;
#endif

  typedef Tpetra::Map<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::Map<>::local_ordinal_type local_ordinal_type;
}

#endif

