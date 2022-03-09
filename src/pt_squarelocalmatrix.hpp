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
 

#ifndef PT_SQUARELOCAL_
#define PT_SQUARELOCAL_

#include "pt_shared.h"  // defines pt::layout_t

namespace pt{

// rank x rank matrix stored on every processor
template <typename SCALAR>
class squareLocalMatrix {

public:

  typedef SCALAR scalar_t;
  typedef typename Kokkos::View<scalar_t **, typename pt::layout_t> valueview_t;

  // Constructor does the allocation and initializes to zero.
  squareLocalMatrix(rank_t rank_) :
    rank(rank_),
    data("squareLocal", rank, rank)
  {
    Kokkos::deep_copy(data, 0.0);
  }

  // Copy constructor
  squareLocalMatrix(squareLocalMatrix &slm) :
    rank(slm.getRank()),
    data("squareLocalCopy", rank, rank)
  {
    Kokkos::deep_copy(data, slm.getView());
  }

  ~squareLocalMatrix() {}

  // setValues:  set all values to a scalar
  void setValues(scalar_t value)
  {
    Kokkos::deep_copy(data, value);
  }

  // Sum of all matrix entries (without absolute value)
  scalar_t sum() const
  {
    scalar_t sum = 0;
    for (rank_t r = 0; r < rank; r++)
      for (rank_t rr = 0; rr < rank; rr++)
        sum += data(r, rr);
    return sum;
  }

  // Hadamard product of two local matrices
  // Element-wise product this = this .* slm
  void hadamard(squareLocalMatrix &slm) 
  {
    // TODO:  There has to be a KokkosKernel method for this operation!
    // TODO:  Or perhaps use GenTen here.
    if (slm.getRank() != rank) 
      throw std::runtime_error("Error:  incompatible inputs to hadamard");

    valueview_t slmview = slm.getView();

    for (rank_t r = 0; r < rank; r++)
      for (rank_t rr = 0; rr < rank; rr++)
        data(r, rr) *= slmview(r, rr);
  }

  // Hadamard product of squareLocalMatrix and outerproduct of vector x.
  // Element-wise product this = this .* (x * x^T)
  void hadamard(Kokkos::View<scalar_t *> &x)
  {
    // TODO:  There has to be a KokkosKernel method for this operation!
    // TODO:  Or perhaps use GenTen here.
    if (x.extent(0) != size_t(rank)) 
      throw std::runtime_error("Error:  incompatible inputs to hadamard");

    for (rank_t r = 0; r < rank; r++)
      for (rank_t rr = 0; rr < rank; rr++)
        data(r, rr) *= (x(r) * x(rr));
  }

  // Accessors
  inline valueview_t getView() { return data; }

  inline rank_t getRank() { return rank; }

  inline scalar_t &operator() (const rank_t r, const rank_t rr) 
  {
    return data(r, rr);
  }

  inline const scalar_t &operator() (const rank_t r, const rank_t rr) const
  {
    return data(r, rr);
  }

  void print(const std::string &msg, std::ostream &ostr = std::cout) const { 

    ostr << "squareLocalMatrix " << msg << std::endl;

    for (rank_t r = 0; r < rank; r++) {
      for (rank_t rr = 0; rr < rank; rr++)
        ostr << data(r, rr) << " ";
      ostr << std::endl;
    }
  }


protected:
  rank_t rank;
  valueview_t data;  // rank x rank storage for squareLocalMatrix values
};

}

#endif
