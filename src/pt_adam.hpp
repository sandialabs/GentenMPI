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
 
#ifndef PT_ADAM_HPP_
#define PT_ADAM_HPP_

namespace pt {
// Algorithm 5.1 GCP with Adam (GCP-Adam)
//1:  function GcpAdam(X, r, s, alpha, beta1, beta2, eps, tau, kappa, nu, el)
//    X = distSys->sptensor;
//    r = distSys->rank;
//    alpha = stepsize
//    nu = stepsize reduction factor on bad iteration
//    beta1, beta2
//    tau = nIterPerEpoch
//    kappa = maxBadEpochs
//    el = lowerBound

//////////////////////////////////////
//15:       Bk = beta1*Bk +(1-beta1)Gk
template <typename scalar_t, typename factormatrix_t>
static void gcp_adam_line_15(
  factormatrix_t *g, 
  factormatrix_t *b, 
  scalar_t beta1
)
{
  rank_t rank = b->getFactorRank();;
  size_t len = b->getLocalLength();
  auto bdata = b->getLocalView();
  auto gdata = g->getLocalView();
  scalar_t one_minus_beta1 = 1. - beta1;
  for (size_t i = 0; i < len; i++) {
    for (rank_t r = 0; r < rank; r++) {
      bdata(i,r) = beta1 * bdata(i,r) + one_minus_beta1 * gdata(i,r);
    }
  }
}

//////////////////////////////////////
//16:       Ck = beta2*Ck + (1 - beta2)(Gk^2)
template <typename scalar_t, typename factormatrix_t>
static void gcp_adam_line_16(
  factormatrix_t *g, 
  factormatrix_t *c, 
  scalar_t beta2
)
{
  rank_t rank = c->getFactorRank();;
  size_t len = c->getLocalLength();
  auto cdata = c->getLocalView();
  auto gdata = g->getLocalView();
  scalar_t one_minus_beta2 = 1. - beta2;
  for (size_t i = 0; i < len; i++) {
    for (rank_t r = 0; r < rank; r++) {
      cdata(i,r) = beta2 * cdata(i,r) 
                 + one_minus_beta2 * gdata(i,r) * gdata(i,r);
    }
  }
}

//////////////////////////////////////
//17:       Bhat_k = Bk/(1 - beta1^t)
//18:       Chat_k = Ck/(1 - beta2^t)
//19:       Ak = Ak - alpha * ( Bhat_k / sqrt( Chat_k + eps )
//20:       Ak = max{Ak, l}
template <typename scalar_t, typename factormatrix_t>
static void gcp_adam_lines_17_thru_20(
  factormatrix_t *a, 
  factormatrix_t *b, 
  factormatrix_t *c, 
  scalar_t alpha, 
  scalar_t eps, 
  scalar_t lowerBound,
  scalar_t divb,
  scalar_t divc
)
{
  scalar_t bhat, chat, tmp;
  rank_t rank = a->getFactorRank();;
  size_t len = a->getLocalLength();
  auto adata = a->getLocalView();
  auto bdata = b->getLocalView();
  auto cdata = c->getLocalView();

  for (size_t i = 0; i < len; i++) {
    for (rank_t r = 0; r < rank; r++) {
      bhat = bdata(i,r) / divb;
      chat = cdata(i,r) / divc;
      tmp = adata(i,r) - alpha * (bhat / (sqrt(chat) + eps));
      adata(i,r) = std::max<scalar_t>(lowerBound, tmp);
    }
  }
}

/*
template <typename sptensor_t, typename ktensor_t>
distSystem<sptensor_t, ktensor_t>::GCP_Adam(
  lossFunction<scalar_t> lossFn
)
{
  typedef typename distsystem_t::sptensor_t sptensor_t;
  typedef typename distsystem_t::ktensor_t ktensor_t;

  //2:  for k = 1,2,...,d do
  //3:    Ak = random matrix of size nk × r
  //4:    Bk,Ck = all-zero matrices of size nk ×r
  //5:  end for

  sptensor_t *X = getSptensor();

  ktensor_t *A = getKtensor();
  ktensor_t *Acopy = new ktensor(A);

  ktensor_t *B = new ktensor(A);     // Get A's structure
  B->putScalar(0.);                  // Store zero values
  ktensor_t *Bcopy = new ktensor(B);

  ktensor_t *C = new ktensor(B);
  ktensor_t *Ccopy = new ktensor(C);

  ktensor_t *grad = new ktensor(A);
  A->setRandomUniform();

  //6:  F = EstObj(X, { Ak })  // estimate loss with fixed set of samples
  const double fixedPercent = 0.01;
  size_t nFixedZeroSamples = fixedPercent * X->getNumLocalZeros();
  size_t nFixedNonZeroSamples = fixedPercent * X->getNumLocalNonZeros();
  sptensor_t fixedSampleTensor = X->stratSampledTensor(nFixedNonZeroSamples,
                                                       nFixedZeroSamples);
  distsystem_t fixedSystem(&fixedSampleTensor, A);
  double fixedError = fixedSystem.computeLossFn(lossFn);

  // Prepare for stochastic gradient computation
  const double stocGradPercent = 0.001;
  size_t nStocGradZeroSamples = stocGradPercent * X->getNumLocalZeros();
  size_t nStocGradNonZeroSamples = stocGradPercent * X->getNumLocalNonZeros();
  SemiStratifiedSamplingStrategy sampler(X);

  //7:  c = 0
  //8:  t = 0 //t=#ofAdamiterations
  int nBadEpochs = 0;
  int nAdamIterations = 0;

  //9:  while c <= kappa do //#=max#ofbadepochs
  while (nBadEpochs < maxBadEpochs) {
    //10:   Save copies of {Ak }, {Bk }, {Ck }
    Acopy->copyData(A);
    Bcopy->copyData(B);
    Ccopy->copyData(C);

    //11:   Fold = F
    double fixedErrorOld = fixedError;

    //12:   for tau iterations do
    for (int iter = 0; iter < nIterPerEpoch; iter++) {
      //13:     { Gk } = StocGrad(X, { Ak } , s)
      stocGrad(nStocGradNonZeroSamples, nStocGradZeroSamples, lossFn, grad,
               sampler);

      scalar_t divb = (1. - std::pow<scalar_t>(beta1, nAdamIterations);
      scalar_t divc = (1. - std::pow<scalar_t>(beta2, nAdamIterations);
      
      //14:     for k = 1,...,d do
      for (mode_t m = 0; m < nModes; m++) {
        factormatrix_t *a = A->getFactorMatrix(m);
        factormatrix_t *b = B->getFactorMatrix(m);
        factormatrix_t *c = C->getFactorMatrix(m);
        factormatrix_t *g = grad->getFactorMatrix(m);

        //15:       Bk = beta1*Bk +(1-beta1)Gk
        gcp_adam_line_15(g, b, beta1);

        //16:       Ck = beta2*Ck + (1 - beta2)(Gk^2)
        gcp_adam_line_16(g, c, beta2);

        //17:       Bhat_k = Bk/(1 - beta1^t)
        //18:       Chat_k = Ck/(1-beta2^t)
        //19:       Ak = Ak - alpha * ( Bhat_k / sqrt( Chat_k + eps )
        //20:       Ak = max{Ak, l}
        gcp_adam_lines_17_thru_20(a, b, c, alpha, eps, lowerBound, divb, divc);
      } //21:     end for

      //22:     t = t+1
      nAdamIterations++;
    } //23:   end for

    //24:   F = EstObj(X, { Ak })
    fixedError = fixedSystem.computeLossFn(lossFn);

    //25:   if F > Fold then
    if (fixedError > fixedErrorOld) {

      // No progress; Restore state and try smaller step
      //26:     Restore saved copied of { Ak }, { Bk }, { Ck }
      A.copyData(Acopy);
      B.copyData(Bcopy);
      C.copyData(Ccopy);

      //27:     F =  Fold
      //28:     t = t - tau
      //29:     alpha =  alpha * nu
      //30:     c = c+1
      fixedError = fixedErrorOld;
      nAdamIteration -= nIterPerEpoch;
      alpha *= nu;
      nBadEpochs++;

    }  //31:   end if
  }  //32: end while

  //33: return { Ak }
  //34: end function
  delete Acopy;
  delete B;
  delete Bcopy;
  delete C;
  delete Ccopy;
  delete grad;
}*/
}
#endif
