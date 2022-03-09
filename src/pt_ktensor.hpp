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
 
#ifndef PT_KTENSOR_
#define PT_KTENSOR_

#include "pt_sptensor.hpp"
#include "pt_factormatrix.hpp"
#include "pt_squarelocalmatrix.hpp"
#include "pt_gram.hpp"
#include "pt_shared.h"
#include "pt_tiebreak.hpp"
#include <vector>

namespace pt {

template <typename FM>
class distKtensor
{
public:

  typedef FM factormatrix_t;
  typedef typename factormatrix_t::map_t map_t;
  typedef typename factormatrix_t::scalar_t scalar_t;
  typedef typename factormatrix_t::lno_t lno_t;
  typedef typename factormatrix_t::gno_t gno_t;
  typedef typename factormatrix_t::valueview_t valueview_t;

  // Construct a distKtensor given array of mode sizes modeSizeView.
  // Uses default Tpetra::Map for each factor matrix.
  distKtensor(rank_t rank_, 
              const std::vector<size_t> &modeSizes_,
              const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
              rank(rank_),  
              nModes(modeSizes_.size()),
              modeSizes(nModes),
              lambda("distKtensor::lambda", rank),
              factors(modeSizes_.size()),
              comm(comm_)
  {
    for (mode_t m = 0; m < nModes; m++) modeSizes[m] = modeSizes_[m];
    for (mode_t m = 0; m < nModes; m++) 
      factors[m] = new factormatrix_t(rank, modeSizes[m], comm);
    for (rank_t r = 0; r < rank; r++) lambda(r) = 1.;
  }

  // Construct a distKtensor given an array of Tpetra::Map describing
  // factor matrix layout
  distKtensor(rank_t rank_,
              std::vector<const map_t *> &maps_,
              const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
              rank(rank_),  
              nModes(maps_.size()),
              modeSizes(nModes),
              lambda("distKtensor::lambda", rank),
              factors(nModes, NULL),
              comm(comm_)
  {
    for (mode_t m = 0; m < nModes; m++) {
      factors[m] = new factormatrix_t(rank, maps_[m]);
      modeSizes[m] = maps_[m]->getMaxAllGlobalIndex()+1; // indices are 0-based
    }
    for (rank_t r = 0; r < rank; r++) lambda(r) = 1.;
  }
 
  // Construct a distKtensor given a sptensor's layout
  // Use createOneToOne to make ktensor maps better align with sptensor maps
  // ktensor maps may not be load balanced
  distKtensor(rank_t rank_,
              distSptensor<scalar_t> *sptensor,
              const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
              rank(rank_),  
              nModes(sptensor->getNumModes()),
              modeSizes(nModes),
              lambda("distKtensor::lambda", rank),
              factors(nModes, NULL),
              comm(comm_)
  {
    if (comm->getRank() == 0) std::cout << "Using oneToOne maps" << std::endl;
//    HashTieBreak<lno_t, gno_t> tieBreak;

    for (mode_t m = 0; m < nModes; m++) {
      Teuchos::RCP<const map_t> spMap = Teuchos::rcp(sptensor->getMap(m),false);
      LoadedTieBreak<lno_t, gno_t> tieBreak(spMap->getNodeNumElements(), *comm);

      Teuchos::RCP<const map_t> oneToOneMap = 
                   Tpetra::createOneToOne<lno_t, gno_t>(spMap, tieBreak);
      factors[m] = new factormatrix_t(rank, oneToOneMap);
      modeSizes[m] = sptensor->getModeSize(m);
    }
    for (rank_t r = 0; r < rank; r++) lambda(r) = 1.;
  }

  // Copy constructor -- creates a deep copy
  distKtensor(distKtensor<factormatrix_t> *src) :
              rank(src->getFactorRank()),  
              nModes(src->getNumModes()),
              modeSizes(nModes),
              lambda("distKtensor::lambda", rank),
              factors(nModes, NULL),
              comm(src->getComm())
  {
    for (mode_t m = 0; m < nModes; m++) {
      factors[m] = new factormatrix_t(*(src->getFactorMatrix(m)));
      modeSizes[m] = src->getModeSize(m);
    }
    Kokkos::deep_copy(lambda, src->getLambdaView());
  }

  // Destructor
  ~distKtensor() { for (mode_t m = 0; m < nModes; m++) delete factors[m]; }

  // Return ktensor's communicator
  inline Teuchos::RCP<const Teuchos::Comm<int> > getComm() const {return comm;}

  // Return rank of all factor matrices
  inline rank_t getFactorRank() const { return rank; }

  // Return number of factor matrices (number of modes)
  inline mode_t getNumModes() const { return nModes; }

  // Return the size (length) of a particular mode 
  inline size_t getModeSize(mode_t m) const { return modeSizes[m]; }

  // Return the array of mode sizes
  inline const std::vector<size_t> &getModeSizes() const { return modeSizes; }

  // Return pointer to a single factor matrix
  inline factormatrix_t *getFactorMatrix(mode_t m) const { return factors[m]; }

  // Return pointer to maps of a single factor matrix
  inline const map_t *getFactorMap(mode_t m) const { 
    return factors[m]->getMap();
  }

  // Return Ktensor Frobenius norm for use in computeResNorm of system
  scalar_t frobeniusNorm() {

    pt::squareLocalMatrix<scalar_t> upsilon(rank);
    upsilon.setValues(1.);

    pt::gramianMatrix<factormatrix_t> gamma(rank);
    // TODO:  This approach uses nModes allreduces; can we combine into one?
    // TODO:  Maybe ...
    // TODO:  -  add squareLocalMatrix constructors that allow user to provide
    // TODO:      Kokkos::View as input
    // TODO:  -  add computeLocal method to gramianMatrix which skips the 
    // TODO:      allreduce in compute
    // TODO:  -  create one view with memory for nModes gamma matrices
    // TODO:      (nModes * (r * r)
    // TODO:  -  subview the memory block for each gamma per mode
    // TODO:  -  computeLocal for each gamma per mode
    // TODO:  -  do one big allreduce of the memory block
    // TODO:  -  do the hadamard products
    for (pt::mode_t m = 0; m < nModes; m++) {
      gamma.compute(factors[m]);
      upsilon.hadamard(gamma);
    }

    upsilon.hadamard(lambda);
    return sqrt(std::abs(upsilon.sum()));
  }
  
  // set each factor matrix to value val
  inline void setValues(scalar_t val) { 
    for (pt::mode_t m = 0; m < nModes; m++) {
      factors[m]->setValues(val);
    }
  }
  
  // Debugging output
  void print(const std::string &msg, std::ostream &ostr = std::cout) const 
  {
    if (comm->getRank() == 0) {

      ostr << "Distributed KTensor " << msg << std::endl
           << "    Number of Modes: " << nModes << std::endl
           << "    Rank: " << rank << std::endl;

      ostr << "    ModeSizes:  ";
      for (mode_t m = 0; m < nModes; m++) ostr << modeSizes[m] << " ";
      ostr << std::endl;

      ostr << "    Lambda:  ";
      for (rank_t r = 0; r < rank; r++) ostr << lambda(r) << " ";
      ostr << std::endl;
    }

    for (mode_t m = 0; m < nModes; m++) {
      char fmsg[256];
      sprintf(fmsg, "Factor Matrix %d", m);
      factors[m]->print(fmsg);
    }
  }

  void printStats(const std::string &msg, std::ostream &ostr = std::cout) const
  {
    // General stats
    if (comm->getRank() == 0) {
      ostr << std::endl;
      ostr << "KSTATS Distributed KTensor: " << msg << std::endl;
      ostr << "KSTATS   Number of processors: " << comm->getSize() << std::endl;

      ostr << std::endl;
      ostr << "KSTATS   Number of modes: " << nModes << std::endl;
      ostr << "KSTATS   Mode sizes:      ";
      for (mode_t m = 0; m < nModes; m++) ostr << modeSizes[m] << " ";
      ostr << std::endl << std::endl;

    }

    // Map stats
    std::vector<size_t> mapSize(nModes);
    for (mode_t m = 0; m < nModes; m++)
      mapSize[m] = getFactorMap(m)->getNodeNumElements();

    std::vector<size_t> gvmin(nModes), gvmax(nModes), gvsum(nModes);
    Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, nModes,
                                 &(mapSize[0]), &(gvsum[0]));
    Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MAX, nModes,
                                 &(mapSize[0]), &(gvmax[0]));
    Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_MIN, nModes,
                                 &(mapSize[0]), &(gvmin[0]));

    if (comm->getRank() == 0) {
      ostr << std::endl;
      for (mode_t m = 0; m < nModes; m++) {
        double avg = gvsum[m] / comm->getSize();
        ostr << "KSTATS   Mode " << m << " map local elements min/max/avg: "
             << gvmin[m] << " / " << gvmax[m] << " / " << avg
             << " Imbal:  " << (avg > 0 ? gvmax[m] / avg : 0.)
             << std::endl;
      }
    }
  }

  // Normalize all factor matrices of Ktensor; update lambda
  // Defaults to normTwo.
  void normalize(enum factormatrixNormType normType = NORM_TWO) 
  {
    Kokkos::View<scalar_t *> norms("distKtensor::norms", rank);
    for (mode_t m = 0; m < nModes; m++) {
      factors[m]->normalize(norms, normType);
      for (rank_t r = 0; r < rank; r++)
        lambda(r) *= norms(r);
    }
  }

  // Normalize one factor matrix of Ktensor; update lambda
  // Defaults to normTwo.
  void normalize(mode_t mode, enum factormatrixNormType normType = NORM_TWO)
  {
    Kokkos::View<scalar_t *> norms("distKtensor::norms", rank);
    factors[mode]->normalize(norms, normType);
    for (rank_t r = 0; r < rank; r++)
      lambda(r) *= norms(r);
  }

  // Return view of lambda array
  inline Kokkos::View<scalar_t *> getLambdaView() const { return lambda; }

  // Reset the values of lambda to a single value
  inline void setLambda(scalar_t newlambda) 
  { 
    for (rank_t r = 0; r < rank; r++) lambda(r) = newlambda;
  }

  // Reset (via copy) the values of lambda to values in an array
  inline void setLambda(Kokkos::View<scalar_t *> &newlambda) 
  { 
    for (rank_t r = 0; r < rank; r++) lambda(r) = newlambda(r); 
  }

  // Reset (via view) the values of lambda
  inline void setLambdaView(Kokkos::View<scalar_t *> &newlambda) 
  { 
    lambda = newlambda;
  }

  // Distribute the lambda values into a factor matrix, allowing reset of 
  // lambda to one.
  // KDD:  I am not sure why this function is needed, but TTB uses it at
  // KDD:  the beginning of CP-ALS.
  void distributeLambda(mode_t mode) {
    factors[mode]->scale(lambda);
    setLambda(1.);
  }

  // Fill the Ktensor with uniform random values, normalized to be stochastic.
  // Fill each factor matrix with random variables, uniform from [0,1),
  // scale each vector so it sums to 1 (adjusting the weights), apply
  // random factors to the weights, and finally normalize weights.
  // The result has stochastic columns in all factors and weights that
  // sum to 1.  Follows TTB::Ktensor::setRandomUniform
  void setRandomUniform() {

    // Fill factor matrices with random values; normalize to sum to one
    for (mode_t m = 0; m < nModes; m++)
      factors[m]->randomize();

    Kokkos::deep_copy(lambda, 1.);
    normalize(NORM_ONE);

    // Adjust lambda by a random factor and scale to sum to one
    // Do it on one processor so we don't have to make random-number
    // generators agree across processors.
    if (comm->getRank() == 0) {
      scalar_t sum = 0.;
      for (rank_t r = 0; r < rank; r++) {
        lambda(r) *= (scalar_t(rand()) / scalar_t(RAND_MAX));
        sum += lambda(r);
      }
      if (sum != 0.) 
        for (rank_t r = 0; r < rank; r++) lambda(r) /= sum;
    }

    Teuchos::broadcast(*comm, 0, rank, lambda.data());
  }   

  // Fill the Ktensor with random values that are consistent for different 
  // numbers of processors.  This enables testing to ensure we get the same
  // answers on different numbers of processors.
  // This is useful for testing the parallel implementation with file-based
  // input; we should get the same result on any number of processors.
  // Don't use this initialization for real problems, though.
  void setFunky() {

    // Fill factor matrices with funky but consistent values; 
    // normalize to sum to one
    for (mode_t m = 0; m < nModes; m++) {
      std::srand(12345+m);
      valueview_t data = factors[m]->getLocalView();

      // To make parallel and serial initialization match, need to "advance"
      // the random number generator to equivalent point in parallel as serial
      // Assumes default Tpetra map for factor matrices
      typename factormatrix_t::gno_t firstGID = 
                                     factors[m]->getMap()->getGlobalElement(0);
      for (typename factormatrix_t::gno_t i = 0; i < firstGID; i++)
        for (rank_t r= 0; r < rank; r++)
          std::rand();

      // Now initialize the local data
      for (size_t i = 0; i < factors[m]->getLocalLength(); i++) {
        for (rank_t r = 0; r < rank; r++) {
          data(i,r) = std::rand(); 
        }
      }
    }

    Kokkos::deep_copy(lambda, 1.);
    normalize(NORM_TWO); 
  }   

  // Copy data from ktensor src to this ktensor
  inline void copyData(
    distKtensor<factormatrix_t> *src,
    bool copyLambda = true  // Some methods (e.g., GCP_ADAM) do not change/use
                            // lambda, so no need to copy it.
  ) 
  {
    for (mode_t m = 0; m < nModes; m++)
      factors[m]->copyData(src->getFactorMatrix(m));
    if (copyLambda)
      Kokkos::deep_copy(lambda, src->getLambdaView());
  }

private:
  rank_t rank;   // rank of each factor matrix in the ktensor
  mode_t nModes; // number of modes (and, thus, factor matrices) in the ktensor
  std::vector<size_t> modeSizes;          // range of each mode; indices for
                                          // mode m can be in [0,modeSizes[m]).
                                          // this is a global value

  Kokkos::View<scalar_t *> lambda;        // Normalization weight
  std::vector<factormatrix_t *> factors;  // Array of the factor matrices

  Teuchos::RCP<const Teuchos::Comm<int> > comm;
};

}
#endif
