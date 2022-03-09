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
 
#ifndef PT_LOSSFNS_
#define PT_LOSSFNS_

namespace pt {

// Base class for loss fns
// All loss functions can derive from this base class.
// They can take parameters (e.g., beta) through their custom constructors
// They must implement the overloaded operator ()
template <typename scalar_t>
class lossFunction 
{
public:
  lossFunction() {}
  virtual scalar_t operator()(const scalar_t x, const scalar_t m) const = 0;
  virtual scalar_t dfdm(const scalar_t x, const scalar_t m) const  = 0;
  virtual scalar_t getLowerBound() const = 0;
  virtual std::string name() const = 0;
};

// L2 loss function
template <typename scalar_t>
class L2_lossFunction : public lossFunction<scalar_t>
{
// A.k.a. "normal" or "gaussian" in tensor toolbox
public:
  L2_lossFunction() {}
  scalar_t operator()(const scalar_t x, const scalar_t m) const {
    return ((x-m) * (x-m));
  }
  scalar_t dfdm(const scalar_t x, const scalar_t m) const {
    return 2. * (m-x);
  }
  scalar_t getLowerBound() const {
    return std::numeric_limits<scalar_t>::lowest();
  }
  std::string name() const { 
    return "L2";
  }
};

// Poisson loss function
template <typename scalar_t>
class Poisson_lossFunction : public lossFunction<scalar_t>
{
// A.k.a. "count" in tensor toolbox
// GB: 1e-10 scooch parameter is copied from tensor toolbox
//   should it be parametrized some way? 
public:
  Poisson_lossFunction() {}
  scalar_t operator()(const scalar_t x, const scalar_t m) const {
    return m - x * log(m+1e-10);
  }
  scalar_t dfdm(const scalar_t x, const scalar_t m) const {
    return 1. - x / (m+1e-10);
  }
  scalar_t getLowerBound() const {
    return 0;
  }
  std::string name() const {
    return "Poisson";
  }
};

// Bernoulli-odds loss function
template <typename scalar_t>
class Bernoulli_odds_lossFunction : public lossFunction<scalar_t>
{
// A.k.a. "binary" in tensor toolbox
// GB: 1e-10 scooch parameter is copied from tensor toolbox
//   should it be parametrized some way? 
public:
  Bernoulli_odds_lossFunction() {}
  // from TT: fh = @(x,m) log(m+1) - x.*log(m + 1e-10);
  scalar_t operator()(const scalar_t x, const scalar_t m) const {
    return log(m+1) - x * log(m+1e-10);
  }
  // from TT: gh = @(x,m) 1./(m+1) - x./(m + 1e-10);
  scalar_t dfdm(const scalar_t x, const scalar_t m) const {
    return 1./(m+1) - x / (m+1e-10);
  }
  scalar_t getLowerBound() const {
    return 0;
  }
  std::string name() const {
    return "Bernoulli-odds";
  }
};

// Bernoulli-logit loss function
template <typename scalar_t>
class Bernoulli_logit_lossFunction : public lossFunction<scalar_t>
{
public:
  Bernoulli_logit_lossFunction() {}
  // from TT: fh = @(x,m) log(exp(m) + 1) - x .* m;
  scalar_t operator()(const scalar_t x, const scalar_t m) const {
    return log(exp(m)+1) - x * m;
  }
  // from TT: gh = @(x,m) exp(m)./(exp(m) + 1) - x;
  scalar_t dfdm(const scalar_t x, const scalar_t m) const {
    return exp(m) / (exp(m)+1.) - x;
  }
  scalar_t getLowerBound() const {
    return std::numeric_limits<scalar_t>::lowest();
  }
  std::string name() const {
    return "Bernoulli-logit";
  }
};

// Poisson-log loss function
template <typename scalar_t>
class Poisson_log_lossFunction : public lossFunction<scalar_t>
{
public:
  Poisson_log_lossFunction() {}
  // from TT: fh = @(x,m) exp(m) - x.*m;
  scalar_t operator()(const scalar_t x, const scalar_t m) const {
    return exp(m) - x * m;
  }
  // from TT: gh = @(x,m) exp(m) - x;
  scalar_t dfdm(const scalar_t x, const scalar_t m) const {
    return exp(m) - x;
  }
  scalar_t getLowerBound() const {
    return std::numeric_limits<scalar_t>::lowest();
  }
  std::string name() const {
    return "Poisson-log";
  }
};

// Rayleigh loss function
// M_PI_4 and M_PI_2 are pi/4 and pi/2 from cmath library
template <typename scalar_t>
class Rayleigh_lossFunction : public lossFunction<scalar_t>
{
public:
  Rayleigh_lossFunction() {}
  // from TT: fh = @(x,m) 2*log(m+1e-10) + (pi/4)*(x./(m+1e-10)).^2;
  scalar_t operator()(const scalar_t x, const scalar_t m) const {
    scalar_t y = x/(m+1e-10);
    return 2.*log(m+1e-10) + M_PI_4*y*y;
  }
  // from TT: gh = @(x,m) 2./(m+1e-10) - (pi/2)*x.^2./(m+1e-10).^3;
  scalar_t dfdm(const scalar_t x, const scalar_t m) const {
    scalar_t y = m + 1e-10;
    return 2./y - M_PI_2*x*x / (y*y*y);
  }
  scalar_t getLowerBound() const {
    return 0;
  }
  std::string name() const {
    return "Rayleigh";
  }
}; 

// Gamma loss function
template <typename scalar_t>
class Gamma_lossFunction : public lossFunction<scalar_t>
{
public:
  Gamma_lossFunction() {}
  // from TT: fh = @(x,m) x./(m+1e-10) + log(m+1e-10);
  scalar_t operator()(const scalar_t x, const scalar_t m) const {
    return x/(m+1e-10) + log(m+1e-10);
  }
  // from TT: gh = @(x,m) -x./((m+1e-10).^2) + 1./(m+1e-10);
  scalar_t dfdm(const scalar_t x, const scalar_t m) const {
    scalar_t y = m + 1e-10;
    return -x/(y*y) + 1./y;
  }
  scalar_t getLowerBound() const {
    return 0;
  }
  std::string name() const {
    return "Gamma";
  }
};  
        
}

#endif
