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
 

#ifndef PT_GRAMIAN_
#define PT_GRAMIAN_

#include "pt_factormatrix.hpp"
#include "pt_squarelocalmatrix.hpp"

namespace pt{

// rank x rank matrix stored on every processor
template <typename factormatrix_t>
class gramianMatrix : 
      public squareLocalMatrix<typename factormatrix_t::scalar_t> {

public:

  typedef typename factormatrix_t::scalar_t scalar_t;
  typedef squareLocalMatrix<scalar_t> slm_t;
  typedef typename slm_t::valueview_t valueview_t;

  // Constructor that does the allocation and by default,
  // calls compute() to compute values
  gramianMatrix(factormatrix_t *fm, bool doCompute = true) : 
    squareLocalMatrix<scalar_t>(fm->getFactorRank())
  {
    if (doCompute) compute(fm);
  }

  // Constructor that does only the allocation and by default,
  // calls compute() to compute values
  gramianMatrix(rank_t rank) : squareLocalMatrix<scalar_t>(rank) { }

  ~gramianMatrix() {}

  // Computation of the gram matrix (fm^T * fm)
  // Assumes factormatrix has a one-to-one map; otherwise, duplicate entries
  // will be accrued into the sum.
  // TODO:  Lots of Kokkos potential here.  See Eric's code.
  void compute(factormatrix_t *fm)
  {
    size_t len = fm->getLocalLength();
    valueview_t fmvals = fm->getLocalView();
    valueview_t localdata("localdata", this->rank, this->rank);

    // Initialize memory to zero
    Kokkos::deep_copy(localdata, 0.0);

    // Accrue local sums
    // TODO:  Prefer change loop order through specialization or 
    //        Kokkos::paralle_for
#ifdef PT_LAYOUTRIGHT  
    for (size_t i = 0; i < len; i++) {
      for (rank_t r = 0; r < this->rank; r++) {
        for (rank_t rr = 0; rr < this->rank; rr++) {
#else
    for (rank_t r = 0; r < this->rank; r++) {
      for (rank_t rr = 0; rr < this->rank; rr++) {
        for (size_t i = 0; i < len; i++) {
#endif
          localdata(r, rr) += fmvals(i,r) * fmvals(i,rr);
        }
      }
    }


    // Allreduce results to all processors
    Teuchos::reduceAll<int, scalar_t>(*(fm->getComm()), Teuchos::REDUCE_SUM,
                                      this->rank * this->rank,
                                      localdata.data(),
                                      this->data.data());
  }

private:

};

}

#endif
