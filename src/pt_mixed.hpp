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
 

#ifndef PT_MIXED_
#define PT_MIXED_

#include <iomanip>
#include "pt_factormatrix.hpp"
#include "pt_gram.hpp"
#include <Teuchos_BLAS.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_LAPACK.hpp>

namespace pt{

  /////////////////////////////////////////////////////////////////////
  // Functions to copy Kokkos views.
  // If layouts are the same, a shallow copy is made and false is returned.
  // If layouts are different, a deep copy is made and true is returned.
  template <typename tgtView, typename srcView>
  inline bool copyView(tgtView &a, srcView &b) {
    a = tgtView("copiedView", b.extent(0), b.extent(1));
    Kokkos::deep_copy(a, b);
    return true;
  }

  template <typename tgtView, typename srcView>
  inline bool copyView(tgtView &a, tgtView &b) {
    a = b;
    return false;
  }

  /////////////////////////////////////////////////////////////////////
  // solveTransposeRHS:  solve upsilon * X = fm^Transpose for X, overwriting
  // fm with the result.

  template <typename squarelocal_t, typename factormatrix_t>
  void solveTransposeRHS(squarelocal_t &upsilon, factormatrix_t &fm)
  {
    // Note:  fm is distributed by rows, while upsilon is copied on each proc.
    // Since fm is distributed by rows, solving for X is a local operation

    // We'll try calling Genten to do it.
    // Wrap the views in Genten::FacMatrix
    typedef typename factormatrix_t::scalar_t scalar_t;
    typedef typename factormatrix_t::valueview_t valueview_t;

    int rank = upsilon.getRank();
    int len = fm.getLocalLength();  // LAPACK requires int

    if (len == 0) return;  // Without any rows of fm on proc, there's nothing to do.

    valueview_t fmView = fm.getLocalView();

    // LAPACK uses LayoutLeft.
    // We need the transpose of the factor matrix to effectively be LayoutLeft.
    // So if factor matrix is stored LayoutLeft, copying it to LayoutRight 
    // gives the transpose in the correct layout for LAPACK.
    // If the factor matrix is LayoutRight, it is already in the correct 
    // format for LAPACK to interpret it as the transpose.

    typedef Kokkos::View<scalar_t **, Kokkos::LayoutRight> transposeview_t;
    transposeview_t useView;
    bool deepCopiedFM = copyView<transposeview_t, valueview_t>(useView, fmView);
    
    // LAPACK wants upsilon to be LayoutLeft.
    // Need to copy upsilon as LAPACK overwrites it with LU factors, so no
    // harm in copying it to the preferred format, even though upsilon is
    // symmetric.
    // TODO:  Don't need to change format; can probably save time in copy 
    // TODO:  by using same format.
    typedef Kokkos::View<scalar_t **, Kokkos::LayoutLeft> upsilonview_t;
    upsilonview_t upsilonCopy("upsilonCopy", rank, rank);
    Kokkos::deep_copy(upsilonCopy, upsilon.getView());

    // Need temporary output array
    int *ipiv = new int[rank];

    // Call LAPACK
    int info = 0;

    Teuchos::LAPACK<int, scalar_t> lapack;
    lapack.GESV(rank, len, upsilonCopy.data(), rank, ipiv,
                           useView.data(), rank, &info);
/*
    int neq = rank;
    int nrhs = len;
    int lda = rank;
    int ldb = rank;
    dgesv(&neq, &nrhs, upsilonCopy.data(), &lda, ipiv, useView.data(), &ldb, &info);
*/

    delete [] ipiv;

    if (info < 0) 
      throw std::runtime_error("solveTransposeRHS:  invalid arguments");
    else if (info > 0)
      throw std::runtime_error("solveTransposeRHS:  dgesv failed");

    // If needed, copy result back to fm

    if (deepCopiedFM) Kokkos::deep_copy(fm.getLocalView(), useView);
    
    return;
  }
  
  /////////////////////////////////////////////////////////////////////
  // symmRectMatmul:  compute result = upsilon * fm - result, overwriting
  // result, where upsilon is a symmetric matrix.
  // (this is useful in computing CP gradients)
  
  template <typename squarelocal_t, typename factormatrix_t>
  void symmRectMatmul(squarelocal_t &upsilon, factormatrix_t &fm,
                        factormatrix_t &result)
  {
    // Note: fm/result are distributed by rows, while upsilon is copied on each proc.
    // Since fm/result are distributed by rows, computing result is a local operation

    typedef typename factormatrix_t::scalar_t scalar_t;
    typedef typename factormatrix_t::valueview_t valueview_t;
    typedef Kokkos::View<scalar_t **, Kokkos::LayoutLeft> view_t;

    int rank = upsilon.getRank();
    int len = fm.getLocalLength();  // LAPACK requires int

    if (len == 0) return;  // Without any rows of fm on proc, there's nothing to do.

    // LAPACK uses LayoutLeft
    // get views of matrices, and copy into LayoutLeft if necessary using copyView
    // note that upsilon is symmetric, so no copy is ever necessary
    valueview_t fmView = fm.getLocalView();
    view_t fmViewLAPACK;
    copyView<view_t, valueview_t>(fmViewLAPACK, fmView);
    valueview_t resView = result.getLocalView();
    view_t resViewLAPACK;
    bool deepCopiedFM = copyView<view_t, valueview_t>(resViewLAPACK, resView);
    valueview_t upsilonView = upsilon.getView();

    // Call LAPACK
    // use symmetric multiplication with symmetric matrix on right side,
    // set alpha to be 1 and beta to be -1 to subtract result from fm*upsilon
    Teuchos::BLAS<int, scalar_t> blas;
    blas.SYMM(Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, len, rank, 
                1., upsilonView.data(), rank, fmViewLAPACK.data(), len, 
                -1., resViewLAPACK.data(), len);

    // If needed, copy result back to result
    if (deepCopiedFM) Kokkos::deep_copy(resView, resViewLAPACK);
    
    return;
  }
}

#endif
