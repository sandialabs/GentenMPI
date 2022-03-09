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
 
#ifndef PT_SYSTEM_
#define PT_SYSTEM_

#define TIME_DETAIL_STOCGRAD
//#define TIME_ADAM_IMBALANCE
//#define TIME_SG_IMBALANCE

// Describes operations on system consisting of a sptensor and a ktensor.
// The ktensor serves as both input and output.  (We would need two sets of 
// internal factor matrices if had separate input and output ktensors.)

#include "pt_shared.h"
#include "pt_ktensor.hpp"
#include "pt_sptensor.hpp"
#include "pt_squarelocalmatrix.hpp"
#include "pt_mixed.hpp"
#include "pt_lossfns.hpp"
#include "pt_adam.hpp"

#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_ParameterList.hpp"

namespace pt {

template <typename sptensor_t, typename ktensor_t>
class distSystem
{
public:

  typedef typename sptensor_t::scalar_t scalar_t;
  typedef typename sptensor_t::lno_t lno_t;
  typedef typename sptensor_t::gno_t gno_t;

  typedef typename ktensor_t::factormatrix_t factormatrix_t;
  typedef typename factormatrix_t::valueview_t factormatrixview_t;
  typedef squareLocalMatrix<scalar_t> slm_t;

  typedef typename factormatrix_t::map_t map_t;
  typedef typename factormatrix_t::import_t import_t;

  static const mode_t UPDATE_NONE = -2;  // Indicates to update no internal
                                         // factor matrices
  static const mode_t UPDATE_ALL  = -3;  // Indicates to update all internal
                                         // factor matrices

  // Constructor 
  distSystem(sptensor_t *sptensor_, ktensor_t *ktensor_,
             mode_t updateFM = UPDATE_ALL, bool optimizeMaps = false);

  // Destructor
  ~distSystem() 
  {
    for (mode_t m = 0; m < nModes; m++) {
      if (importers[m] != NULL) {
        delete importers[m];
        delete internalFactorMatrices[m]; // delete only if allocated for 
                                          // imports
      }
    }
  }

  // Accessors

  inline sptensor_t *getSptensor() const { return sptensor; }

  inline ktensor_t *getKtensor() const { return ktensor; }

  inline const map_t *getDomainMap(mode_t m) const {
    return ktensor->getFactorMatrix(m)->getMap();
  }

  inline const map_t *getRangeMap(mode_t m) const {
    return ktensor->getFactorMatrix(m)->getMap();
  }

  inline const import_t *getImporter(mode_t m) const {
    return importers[m];
  }

  // print stats about the communication
  void printStats(const std::string &msg, std::ostream &ostr=std::cout) const;

  // compute model value of ktensor at a given sptensor index
  // This version might be expensive with repeated use, as it mallocs
  // memory in each call.  See alternative version that takes temporary 
  // memory as arguments.
  inline scalar_t computeModelAtIndex(size_t idx)
  {
    Kokkos::View<scalar_t*> prod("prod", rank);
    return computeModelAtIndex(idx, prod);
  }

  // compute model value of ktensor at a given sptensor index.
  // accepts memory of size rank to avoid repeated allocations
  // TODO:  accept array of views if available to avoid indirection
  scalar_t computeModelAtIndex(size_t tidx, Kokkos::View<scalar_t*> &prod)
  {
#ifdef DEBUG
    // Don't want to do this check every time
    // Be careful:  since we are sharing factor matrices (via pointers) 
    // among systems, a system might think its internal factor matrices 
    // are in sync when, in reality, some other system has changed its 
    // factor matrices.  Ouch!  Maybe sharing is a bad design, but copying
    // is expensive.  Try to manage it manually for now.
    for (mode_t m = 0; m < nModes; m++) {
      if (!internalInSyncWithKtensor[m]) {
        throw std::runtime_error("Internal factor matrices must be "
                                 "in sync with Ktensor before calling "
                                 "computeModelAtIndex; call "
                                 "updateAllInternalFactorMatricesFromKtensor");
      }
    }
#endif

    Kokkos::View<scalar_t *> lambda = ktensor->getLambdaView();

    // Compute the model's value at index tidx
    scalar_t valModel = 0.;

    for (rank_t r = 0; r < rank; r++) prod(r) = lambda(r);

    for (mode_t m = 0; m < nModes; m++) {
      lno_t lidx = sptensor->getLocalIndices()(tidx,m);
      factormatrixview_t data = internalFactorMatrices[m]->getLocalView();
      for (rank_t r = 0; r < rank; r++) prod(r) *= data(lidx,r);
    }

    for (rank_t r = 0; r < rank; r++) valModel += prod(r);

    return valModel;
  }

  // inner product of sptensor and ktensor, weighted by external vector
  // element-wise dot product of all elements
  scalar_t innerprod(Kokkos::View<scalar_t *> &scaleValues);

  // mttkrp with sptensor and ktensor -- 
  // in place:  replace values in mode-th factor matrix
  inline void mttkrp(mode_t mode)
  { 
    mttkrpGuts(mode, ktensor->getFactorMatrix(mode)); 
  }

  // mttkrp with sptensor and ktensor -- 
  // external:  replace values in the provided factor matrix.
  // This version of mttkrp is more expensive than mttkrp(mode), as it
  // error checks the output factor matrix's maps to ensure they match 
  // those of the system.
  void mttkrp(mode_t mode, factormatrix_t * const outFactorMatrix) {

    // Check that map of output factor matrix is compatible with ktensor.
    const map_t *outMap = outFactorMatrix->getMap();
    if (!outMap->isSameAs(*(ktensor->getFactorMatrix(mode)->getMap()))) {
      // Maps are not compatible; doExport will not work correctly.
      // Original code (tag 2018_LDRD_Review) saved exporters explictly
      // and, in this case, created a temporary Export object for use here.
      // But since the range and domain maps are identical in most cases,
      // using only Import objects (in forward and reverse mode) save memory
      // and complexity.  This use case, then, is no longer supported.
      throw std::runtime_error("Tpetra::Map for output FactorMatrix is not "
                               "compatible with ktensor's Map");
    }

    // Perform the mttkrp
    mttkrpGuts(mode, outFactorMatrix);
  }

  // CP_ALS
  void cp_als(scalar_t tolerance, int minIters, int maxIters, 
              int &numIter, scalar_t &resNorm);
 
  // SGD/Adam algorithm            
  void GCP_Adam(lossFunction<scalar_t> &lossFn, Teuchos::ParameterList &params);

  // approximates CP gradient via sampling
  void stocGrad(size_t numNonzeroSamples, size_t numZeroSamples, 
                lossFunction<scalar_t> &f, ktensor_t* gradient,
                sptensor_t *&Y,
                SamplingStrategy<sptensor_t> *sampler,
                unsigned int seed,
                Teuchos::RCP<Teuchos::Time> &timeSample,
                Teuchos::RCP<Teuchos::Time> &timeSystem,
                Teuchos::RCP<Teuchos::Time> &timeDfDm,
                Teuchos::RCP<Teuchos::Time> &timeMTTKRP
  );

  // Return the residual Frobenius norm between this system's sparse tensor and 
  // its ktensor.
  scalar_t getResidualNorm()
  {
    scalar_t knorm = ktensor->frobeniusNorm();
    scalar_t xnorm = sptensor->frobeniusNorm();
    auto lambda = ktensor->getLambdaView();
    scalar_t xdotk = innerprod(lambda);
    return computeResNorm(xnorm, knorm, xdotk);
  }

  // Compute the loss function f(x,m) for each tensor entry, 
  // with appropriate scaling for sampled nonzeros and zeros 
  // (e.g., see Kolda & Hong eqn 5.2)
  // The lossFunction argument is a class defining the particular operations
  // to be done for f(x,m)
  scalar_t computeLossFn(lossFunction<scalar_t> &lossFn);

  // Compute the residual Frobenius norm between data tensor X
  // and Ktensor model M as
  //     sqrt{ |X|^2_F - 2(X . M) + |M|^2_F }
  // Adapted from TTB:
  // The residual can be slightly negative due to roundoff errors
  // if the model is a nearly exact fit to the data.  The threshold
  // for fatal error was copied from TTB where it was determined 
  // from experimental observations.
  scalar_t computeResNorm(const scalar_t xNorm,
                          const scalar_t mNorm,
                          const scalar_t xDotm)
  {
    scalar_t d = (xNorm * xNorm) + (mNorm * mNorm) - (2 * xDotm);
    scalar_t dSmallNegThresh = -(xDotm * 1e3 *
                                sqrt(std::numeric_limits<scalar_t>::epsilon())); 
    scalar_t result;

    if (d > std::numeric_limits<scalar_t>::min()) result = sqrt(d);
    else if (d > dSmallNegThresh) result = 0.0;
    else {
      std::ostringstream sMsg;
      sMsg << "computeResNorm:  residual norm is negative: " << d;
      throw std::runtime_error(sMsg.str());
    }

    return( result );
  }

private:

  mode_t nModes;
  rank_t rank;
  sptensor_t *sptensor;
  ktensor_t *ktensor;

  // if importers[m] != NULL, internalFactorMatrices[m] is a temporary 
  // factor matrices wrt tensor map in mode m -- needed for communication of
  // factor matrix values for mttkrp, model evaluation, etc.;
  // if importers[m] == NULL, internalFactorMatrices[m] will point to the 
  // ktensor's mode m factor matrix directly; no communication is needed.
  std::vector<factormatrix_t *> internalFactorMatrices;

  // Flags indicating whether ktensor's factor matrices and 
  // internalFactorMatrices are in-sync.  false means communication is needed
  // to update internalFactorMatrices from ktensor factor matrices' values.
  std::vector<bool> internalInSyncWithKtensor;

  // Communication patterns between ktensor and sptensor.
  // domainmap[m] == rangemap[m] == ktensor->getFactorMatrix(m)->getMap().
  // When domainmap[m] is same as sptensor's map[m], importers[m] == NULL

  std::vector<import_t*> importers;  // Importers relative to domainmaps

  // communicate ktensor's mode-th factor matrix 
  // to tensor's factor matrices tensorFMs, if needed
  inline void updateInternalFactorMatrixFromKtensor(
    mode_t mode, 
    bool force = false
  )
  {
    if (force || !internalInSyncWithKtensor[mode]) {
      if (importers[mode] != NULL) {
        internalFactorMatrices[mode]->doImport(ktensor->getFactorMatrix(mode), 
                                               importers[mode], 
                                               Tpetra::INSERT);
      }
      internalInSyncWithKtensor[mode] = true;
    }
  }

  // communicate all of ktensor's factor matrices to internal factor matrices
  inline void updateAllInternalFactorMatricesFromKtensor(bool force = false)
  {
    for (mode_t m = 0; m < nModes; m++)
      updateInternalFactorMatrixFromKtensor(m, force);
  }

  // communicate almost all of ktensor's factor matrices to tensor's factor 
  // matrices tensorFMs, if needed, skipping mode skipmode
  // (can use this function at beginning of CP-ALS iterations for first mode)
  inline void updateAlmostAllInternalFactorMatricesFromKtensor(mode_t skipmode)
  {
    for (mode_t m = 0; m < nModes; m++) {
      if (m == skipmode) continue;
      updateInternalFactorMatrixFromKtensor(m);
    }
  }

  // distribute Ktensor's lambda into the ktensor's mode-th factor matrix; 
  // update internalInSyncWithKtensor to show that the mode-th factor matrix 
  // changed.
  inline void distributeLambdaToKtensor(mode_t mode)
  {
    ktensor->distributeLambda(mode);
    internalInSyncWithKtensor[mode] = false;
  }

  // Call pt::solveTransposeRHS with ktensor's mode-th factor matrix;
  // update internalInSyncWithKtensor to show that the mode-th factor matrix 
  // changed.
  inline void solveTransposeRHS(slm_t &upsilon, mode_t mode)
  {
    Teuchos::TimeMonitor
           tm(*Teuchos::TimeMonitor::getNewTimer("SolveTransposeRHS"));
    pt::solveTransposeRHS(upsilon, *(ktensor->getFactorMatrix(mode)));
    // pt::solveTransposeRHS overwrites the factor matrix
    internalInSyncWithKtensor[mode] = false;
  }

  // Scale ktensor's mode-th factor matrix by inverse of scaleValues;
  // update internalInSyncWithKtensor to show that the mode-th factor matrix 
  // changed.
  inline void scaleInverseFactorMatrix(Kokkos::View <scalar_t *> scaleValues,
                                       mode_t mode)
  {
    Teuchos::TimeMonitor
           tm(*Teuchos::TimeMonitor::getNewTimer("ScaleInverseFactorMatrix"));
    try {
      ktensor->getFactorMatrix(mode)->scaleInverse(scaleValues);
    }
    catch (std::exception &e) {
      std::cout << "Error in scaleInverseFactorMatrix:  "
                << "lambda contains zero value for inverse scaling" 
                << std::endl;
      throw(e);
    }
    internalInSyncWithKtensor[mode] = false; 
  }

  // The actual guts of mttkrp
  void mttkrpGuts(mode_t mode, factormatrix_t * const outFactorMatrix);

  // print stats about an importer
  void printImportStats(const import_t *importer, mode_t m, 
                        std::ostream &ostr) const
  {
    Teuchos::RCP<const Teuchos::Comm<int> > comm= sptensor->getComm();
    int me = comm->getRank();
    int np = comm->getSize();

    if (importer == NULL) {
      if (me == 0) 
        ostr << "SYSSTATS  Mode " << m << " " << "importer is NULL" 
             << std::endl;
    }
    else {
      // Communication volume
      const int nstats=5;
      size_t mystats[nstats];
      mystats[0] = importer->getNumExportIDs();
      mystats[1] = importer->getNumRemoteIDs();
      mystats[2] = importer->getNumPermuteIDs();
  
      // Number of messages
      std::map<int,int> exportpidmap;
      Teuchos::ArrayView<const int> exportpids = importer->getExportPIDs();
      size_t nexportpids = importer->getNumExportIDs();
      for (size_t i = 0; i < nexportpids; i++) {
        int k = exportpids[i];
        if (exportpidmap.find(k) != exportpidmap.end())
          exportpidmap[k] = exportpidmap[k] + 1;
        else
          exportpidmap[k] = 1;
      }
      mystats[3] = exportpidmap.size();
  
      // Size of largest message 
      int maxmsg = 0;
      for (std::map<int,int>::iterator it = exportpidmap.begin();
           it != exportpidmap.end(); it++)
        if (it->second > maxmsg) maxmsg = it->second;
      mystats[4] = maxmsg;
  
      // Some detailed output for low processor counts
#if 0
      if (np < 26) {
        ostr << "        ";
        ostr << me << " " << "Importer " << m << ":"
                   << " nSend " << mystats[0]
                   << " nRecv " << mystats[1]
                   << " nPermute " << mystats[2]
                   << " nPids " << mystats[3]
                   << " maxmsg " << mystats[4]
                   << std::endl;
        if (exportpidmap.size() > 0) {
          ostr << me << "    " << "IMPORT " << m << ": ";
          for (std::map<int,int>::iterator it = exportpidmap.begin();
               it != exportpidmap.end(); it++)
            ostr << "(" << it->first << " " << it->second << ") ";
          ostr << std::endl;
        }
      }
#endif
  
      size_t gmin[nstats], gmax[nstats], gsum[nstats];
      Teuchos::reduceAll<int, size_t>(*comm, Teuchos::REDUCE_SUM, nstats, 
                                      mystats, gsum);
      Teuchos::reduceAll<int, size_t>(*comm, Teuchos::REDUCE_MIN, nstats, 
                                      mystats, gmin);
      Teuchos::reduceAll<int, size_t>(*comm, Teuchos::REDUCE_MAX, nstats, 
                                      mystats, gmax);
  
      if (me == 0) {
        ostr << "SYSSTATS  Mode " << m << " " << "importer:  "
             << "nSend min/max/avg " 
             << gmin[0] << " / " << gmax[0] << " / " << gsum[0] / np 
             << std::endl;
        ostr << "SYSSTATS  Mode " << m << " " << "importer:  "
             << "nRecv min/max/avg " 
             << gmin[1] << " / " << gmax[1] << " / " << gsum[1] / np 
             << std::endl;
        ostr << "SYSSTATS  Mode " << m << " " << "importer:  "
             << "nPermute min/max/avg " 
             << gmin[2] << " / " << gmax[2] << " / " << gsum[2] / np 
             << std::endl;
        ostr << "SYSSTATS  Mode " << m << " " << "importer:  "
             << "nPids min/max/avg " 
             << gmin[3] << " / " << gmax[3] << " / " << gsum[3] / np 
             << std::endl;
        ostr << "SYSSTATS  Mode " << m << " " << "importer:  "
             << "maxmsg min/max/avg " 
             << gmin[4] << " / " << gmax[4] << " / " << gsum[4] / np 
             << std::endl;
      }
    }
  }

};

//////////////////////////////////////////////////////////////////////////////
// Constructor 
template <typename sptensor_t, typename ktensor_t>
distSystem<sptensor_t, ktensor_t>::distSystem(
  sptensor_t *sptensor_, // sparse tensor in the system
  ktensor_t *ktensor_,   // ktensor in the system
  mode_t updateFM,       // how to update the internal factor matrices
                         // updateFM == UPDATE_ALL  ==> update all FMs (default)
                         // updateFM == UPDATE_NONE ==> update no FMs
                         // otherwise, update all FMs EXCEPT 
                         //            updateFM-th FM
  bool optimizeMaps      // optional flag indicating whether to optimize the
                         // system's maps ("column" maps)
) :
  nModes(sptensor_->getNumModes()),
  rank(ktensor_->getFactorRank()),
  sptensor(sptensor_),
  ktensor(ktensor_), 
  internalFactorMatrices(sptensor->getNumModes(),NULL),
  internalInSyncWithKtensor(nModes, false),
  importers(nModes, NULL)
{
  // Sanity checking of input

  if (ktensor->getNumModes() != nModes)
    throw std::runtime_error("distSystem:  Incompatible number of modes");

  for (mode_t m = 0; m < nModes; m++) {
    if (sptensor->getModeSize(m) != ktensor->getModeSize(m))
      throw std::runtime_error("distSystem:  Incompatible mode sizes");
  }

  if (optimizeMaps && (sptensor->getComm()->getRank() == 0))
    std::cout << "Using optimized maps" << std::endl;

  // Build communication patterns and internal FM storage as needed.

  for (mode_t m = 0; m < nModes; m++) {

    const map_t *fmMap = ktensor->getFactorMatrix(m)->getMap();
    const map_t *spMap = sptensor->getMap(m);

    // Build importers as needed for the given ktensors;
    // if maps match in/out ktensors' maps, importers are not needed.

    if (!(spMap->isSameAs(*(fmMap)))) {

#ifdef PRINT_MAPS
      {
        std::cout << sptensor->getComm()->getRank()
                  << " BEFORE BEFORE BEFORE BEFORE BEFORE ::  MODE " << m 
                  << std::endl;
        Teuchos::FancyOStream fancy(Teuchos::rcpFromRef(std::cout));
        sptensor->getMap(m)->describe(fancy, Teuchos::VERB_EXTREME);
      }
#endif // PRINT_MAPS

      if (optimizeMaps) {
        // Optimize layout of the sptensor's map wrt the factormatrix's map
        // (changes map and local indexing)
        // Return an importer between the optimized sptensor map and the 
        // factormatrix map

        importers[m] = sptensor->optimizeMapAndBuildImporter(m, fmMap);
      }
      else {
        // build the importer for the map wrt fmMap as the domain map
        importers[m] = new import_t(Teuchos::rcp(fmMap, false),
                                    Teuchos::rcp(spMap,false));
      }

#ifdef PRINT_MAPS
      {
        std::cout << sptensor->getComm()->getRank()
                  << " AFTER AFTER AFTER AFTER AFTER :: MODE " << m
                  << std::endl;
        Teuchos::FancyOStream fancy(Teuchos::rcpFromRef(std::cout));
        sptensor->getMap(m)->describe(fancy, Teuchos::VERB_EXTREME);
      }
#endif // PRINT_MAPS

      // Build internal factor matrix
      internalFactorMatrices[m] = new factormatrix_t(rank,
                                                     sptensor->getMap(m));
    }
    else {
      // tensor in mode m and m-th factor matrix have same maps;
      // internalFactorMatrices[m] can point directly to factor matrix m
      internalFactorMatrices[m] = ktensor->getFactorMatrix(m);
      internalInSyncWithKtensor[m] = true;
    }
  }

  // Update internal factor matrices' values
  if (updateFM == UPDATE_ALL) {
    // Communicate all needed factor-matrix data
    updateAllInternalFactorMatricesFromKtensor();
  }
  else if (updateFM != UPDATE_NONE) {
    // Communicate all factor-matrix data except the updateFM-th 
    // factor matrix.
    updateAlmostAllInternalFactorMatricesFromKtensor(updateFM);
  }
}

//////////////////////////////////////////////////////////////////////////////

#undef TESTCOPY

template <typename sptensor_t, typename ktensor_t>
void distSystem<sptensor_t, ktensor_t>::mttkrpGuts(
  mode_t mode,                            // input:  mode along which to mttkrp
  factormatrix_t * const outFactorMatrix  // output: factor matrix 
                                          //         computed by mttkrp
)
{
//  Teuchos::TimeMonitor tm(*Teuchos::TimeMonitor::getNewTimer("MTTKRP"));

#ifdef TIME_DETAIL_MTTKRP
  Teuchos::RCP<Teuchos::Time> 
    timeImport(Teuchos::TimeMonitor::getNewTimer("MTTKRP 00 Import")),
    timeSetupViews(Teuchos::TimeMonitor::getNewTimer("MTTKRP 01 SetupViews")), 
    timeLRCopyIn(Teuchos::TimeMonitor::getNewTimer("MTTKRP 02 LR COPY IN")), 
    timeLoop(Teuchos::TimeMonitor::getNewTimer("MTTKRP 03 Loop")), 
    timeExport(Teuchos::TimeMonitor::getNewTimer("MTTKRP 04 Export")),
    timeLRCopyOut(Teuchos::TimeMonitor::getNewTimer("MTTKRP 05 LR COPY OUT"));
#endif // TIME_DETAIL_MTTKRP

  // TODO:  See Eric Phipp's Kokkos-ized version of mttkrp.
  // TODO:  See also Jiajia's slides 2/20/17, slide 5 for reordering of 
  // TODO:  mttkrp operations, for reduced FLOPS (a la SPLATT).

  
  // Gather needed input (using importers, if needed).
  // Want to do operations using the local views, so extract them, too.

#ifdef TIME_DETAIL_MTTKRP
  timeImport->start();
#endif

  std::vector<factormatrixview_t> midFMView(nModes);

  for (mode_t m = 0; m < nModes; m++) {

    if (m == mode) continue;       // m == mode is not an input factor

    midFMView[m] = internalFactorMatrices[m]->getLocalView();
    updateInternalFactorMatrixFromKtensor(m);
  }

#ifdef TIME_DETAIL_MTTKRP
  timeImport->stop();

  timeSetupViews->start();
#endif

  // Get view of result factor matrix to accumulate local product values
  // If outFactorMatrix is the mode-th factor matrix of ktensor (as in
  // CP_ALS), we can use internalFactorMatrix[mode] for the MTTKRP result.
  // If outFactorMatrix is not the mode-th factor matrix of ktensor (as in
  // GCP_Adam, where outFactorMatrix is the gradient), allocate a new
  // factor matrix with the same maps as internalFactorMatrix[mode] to 
  // receive the result.

  factormatrix_t *result;
  bool useInternalFMForResult = 
       (outFactorMatrix == ktensor->getFactorMatrix(mode));
  bool allocatedResultInternal = false;

  // initialize the factor matrix associated with the output to zero
  outFactorMatrix->setValues(0.);

  if (importers[mode] != NULL) {
    // Local products and output factor matrix (mode) have different maps
    // Use output factor matrix with same map in mode as tensor
    if (useInternalFMForResult) {
      // Store mttkrp results in internal factor matrix
      // OK for CP_ALS loop; bad for GCP-Adam loop
      result = internalFactorMatrices[mode];
    }
    else {
      // Allocate result with same map as internalFactorMatrices[mode]
      // Good for GCP-Adam loop; unnecessary for CP_ALS
      result = new factormatrix_t(rank, 
                                  internalFactorMatrices[mode]->getMap());
      allocatedResultInternal = true;
    }
    result->setValues(0.);
  }
  else {
    // Tensor in mode mode and output factor matrix have same maps;
    // no export needed; can use outFactorMatrix directly
    result = outFactorMatrix;
  }
  midFMView[mode] = result->getLocalView();
  factormatrixview_t resultView = midFMView[mode];

#ifdef TIME_DETAIL_MTTKRP
  timeSetupViews->stop();
#endif

#ifdef TESTCOPY
#ifdef TIME_DETAIL_MTTKRP
  timeLRCopyIn->start();
#endif

  typedef Kokkos::View<scalar_t **, Kokkos::LayoutRight> LRfactormatrixview_t;
  std::vector<LRfactormatrixview_t> LRmidFMView;
  LRmidFMView.reserve(nModes);
    
  for (mode_t m = 0; m < nModes; m++) {
    LRmidFMView.push_back(LRfactormatrixview_t("copym", 
                                               midFMView[m].extent(0),
                                               midFMView[m].extent(1)));
    Kokkos::deep_copy(LRmidFMView[m], midFMView[m]);
  }

  LRfactormatrixview_t LRresultView = LRmidFMView[mode];

#ifdef TIME_DETAIL_MTTKRP
  timeLRCopyIn->stop();
#endif
#endif

  // For now, we'll do the product like Tensor ToolBox does.
  // Later, we'll do it like Eric Phipps does in Kokkos-ized TTB. TODO

#ifdef TIME_DETAIL_MTTKRP
  timeLoop->start();
#endif

  size_t nnz = sptensor->getLocalNumIndices();  // use indices for sampled tensors
  typename sptensor_t::lnoview_t localIndices = sptensor->getLocalIndices();
  typename sptensor_t::valueview_t values = sptensor->getValues();

  Kokkos::View<scalar_t *> prod("prod", rank);

  for (size_t nz = 0; nz < nnz; nz++) {

    // Initialize product to tensor nonzero value
    scalar_t nzVal = values(nz);
    for (rank_t r = 0; r < rank; r++) prod(r) = nzVal;
    // KDD on my mac, deep_copy was slower than above
    // Kokkos::deep_copy(prod, values(nz)); 


    // Get the local index in the result factor matrix
    lno_t resultIdx = localIndices(nz, mode);

    // For each mode m except mode...
    for (mode_t m = 0; m < nModes; m++) {

      if (m == mode) continue;

      lno_t localIdx = localIndices(nz, m);

      // For each rank...
      for (rank_t r = 0; r < rank; r++) {
     
        // Multiply in contribution from factor matrix
#ifdef TESTCOPY
        prod(r) *= LRmidFMView[m](localIdx, r);
#else
        prod(r) *= midFMView[m](localIdx, r);
#endif
      }
    }

    // Accumulate the partial sum value into result
    for (rank_t r = 0; r < rank; r++) {
#ifdef TESTCOPY
      LRresultView(resultIdx, r) += prod(r);
#else
      resultView(resultIdx, r) += prod(r);
#endif
    }
  }
#ifdef TIME_DETAIL_MTTKRP
  timeLoop->stop();
#endif

#ifdef TESTCOPY
#ifdef TIME_DETAIL_MTTKRP
  timeLRCopyOut->start();
#endif
    
  for (mode_t m = 0; m < nModes; m++)
    Kokkos::deep_copy(midFMView[m], LRmidFMView[m]);
   
#ifdef TIME_DETAIL_MTTKRP
  timeLRCopyOut->stop();
#endif
#endif

#ifdef TIME_DETAIL_MTTKRP
  timeExport->start();
#endif
  // Export the local values back to the ktensor's mode-th factor matrix,
  // if needed
  if (importers[mode] != NULL) {
    outFactorMatrix->doExport(result, importers[mode], Tpetra::ADD);
  }

  // Internal factor matrix mode may no longer be in-sync with ktensor.
  // Two possible reasons (depending on how mttkrp was called):
  // - mttkrp overwrote ktensor's mode-th factor matrix, or
  // - mttkrp used internalFactorMatrix to store temporary values of external
  //   output factor matrix, making internalFactorMatrix[mode] inconsistent
  //   with ktensor's mode-th factor matrix.
  if (useInternalFMForResult)
    internalInSyncWithKtensor[mode] = false;

  if (allocatedResultInternal) delete result;

#ifdef TIME_DETAIL_MTTKRP
  timeExport->stop();
#endif
}
  
////////////////////////////////////////////////////////////////////////////
template <typename sptensor_t, typename ktensor_t>
void distSystem<sptensor_t, ktensor_t>::cp_als(
  scalar_t tolerance,  // Input:  solver tolerance
  int minIters,        // Input:  min number of iterations
  int maxIters,        // Input:  max number of iterations
  int &numIter,       // Output: number of iterations needed
  scalar_t &resNorm    // Output: resulting residual Norm
)
{
  int me = sptensor->getComm()->getRank();

  // Distribute the initial guess to have weights of one.
  // KDD:  Why?  Genten does not do this
  // KDD:  distributeLambdaToKtensor(0);

  // Local lambda; differs from ktensor's lambda
  Kokkos::View<scalar_t *> lambda("cpals_lambda", rank);
  Kokkos::deep_copy(lambda, 1.);

  // Define gamma, an array of Gramian Matrices corresponding to the
  // factor matrices in ktensor.
  // Note that we skip computing the zero-th entry of gamma since 
  // it is not used in the zero-th inner iteration.
  typedef gramianMatrix<factormatrix_t> gram_t;
  std::vector<gram_t *> gamma(nModes);
  for (mode_t m = 0; m < nModes; m++) 
    gamma[m] = new gram_t(ktensor->getFactorMatrix(m), (m > 0));

  // Define upsilon to store Hadamard products of the gamma matrices.
  // The matrix is called Z in the Matlab Computations paper.
  slm_t upsilon(rank);

  // Pre-calculate the Frobenius norm of the tensor x.
  scalar_t sptensorNorm = sptensor->frobeniusNorm();

  // Allocate one extra factor matrix for use in convergence test
  // It holds the last mttkrp result, so that computation of innerproduct
  // can be made less expensive (Smith, Karypis).
  // Copy constructor here gives proper maps, etc.  Later will copy just data.
  factormatrix_t keepLastMttkrp(*(ktensor->getFactorMatrix(nModes-1)));

  // KDD Taking the easy unknown fit branch here
  scalar_t fitold = 0., fit = 0.;

  //--------------------------------------------------
  // Main loop.
  //--------------------------------------------------
  for (numIter = 0; numIter < maxIters; numIter++) {

    Teuchos::TimeMonitor 
           tm(*Teuchos::TimeMonitor::getNewTimer("CP-ALS 00 OuterIteration"));

    fitold = fit;

    // Iterate over all N modes of the tensor
    for (mode_t mode = 0; mode < nModes; mode++) {

      Teuchos::TimeMonitor 
          tmi(*Teuchos::TimeMonitor::getNewTimer("CP-ALS 01 Iteration ModeLoop"));

      factormatrix_t *factorMatrix = ktensor->getFactorMatrix(mode);

      // Update ktensor's mode-th factor via MTTKRP with sptensor
      // (Khattri-Rao product).
      {
        Teuchos::TimeMonitor 
          tmm(*Teuchos::TimeMonitor::getNewTimer("CP-ALS 02 mttkrp"));

        mttkrp(mode);
      }

      {
        Teuchos::TimeMonitor 
          tmm(*Teuchos::TimeMonitor::getNewTimer("CP-ALS 02a saveMttkrp"));
        if (mode == (nModes-1)) {
          // Keep copy of last MTTKRP for use in convergence check
          Kokkos::deep_copy(keepLastMttkrp.getLocalView(),
                            factorMatrix->getLocalView());
        }
      }

      {
        Teuchos::TimeMonitor 
          tmm(*Teuchos::TimeMonitor::getNewTimer("CP-ALS 03 UpsilonHadamard"));
        // Compute the matrix of coefficients in the solve step.
        upsilon.setValues(1.);
  
        for (mode_t m = 0; m < nModes; m++) {
          if (m != mode) {
            upsilon.hadamard(*(gamma[m]));
          }
        }
      }

      {
        Teuchos::TimeMonitor 
          tmm(*Teuchos::TimeMonitor::getNewTimer("CP-ALS 04 solvetranspose"));

        // Solve upsilon * X = factorMatrix' for X, 
        // and overwrite factorMatrix with the result.  
        // Equivalent to the Matlab operation 
        // factorMatrix = (upsilon \ factorMatrix')'.
        solveTransposeRHS(upsilon, mode);
      }

      {
        Teuchos::TimeMonitor 
          tmm(*Teuchos::TimeMonitor::getNewTimer("CP-ALS 05 norms"));

        // Compute lambda; copying TTB here with choice of norm
        if (numIter == 0) {
          // L2 norm on first iteration.
          factorMatrix->norm2(lambda);
        }
        else {
          // L0 norm (max) on other iterations.
          factorMatrix->normInf(lambda);
        }
      }

      {
        Teuchos::TimeMonitor 
          tmm(*Teuchos::TimeMonitor::getNewTimer("CP-ALS 06 scaleInverse"));

        // Scale factormatrix by the inverse of lambda.
        // This can throw an exception, divide-by-zero, if lambda has a zero
        scaleInverseFactorMatrix(lambda, mode);
      }

      {
        Teuchos::TimeMonitor 
          tmm(*Teuchos::TimeMonitor::getNewTimer("CP-ALS 07 gammaCompute"));

        // Update factormatrix's corresponding Gramian matrix.
        gamma[mode]->compute(factorMatrix);
      }
    }    

    {
      Teuchos::TimeMonitor 
        tmm(*Teuchos::TimeMonitor::getNewTimer("CP-ALS 08 checkConvergence"));

      // Compute Frobenius norm of "p", the current factorization consisting of 
      // lambda and ktensor.
      upsilon.hadamard(*(gamma[nModes-1]));
      upsilon.hadamard(lambda);
      scalar_t pNorm = sqrt(std::abs(upsilon.sum()));

      // Compute inner product of input data sptensor with "p".
      // OLD scalar_t xpipold = innerprod(lambda);
      scalar_t xpip = 
               keepLastMttkrp.innerprod(*(ktensor->getFactorMatrix(nModes-1)),
                                        lambda);

      // Compute Frobenius norm of residual using quantities formed earlier.
      resNorm = computeResNorm(sptensorNorm, pNorm, xpip);

      // Compute the relative fit and change since the last iteration.
      fit = 1. - (resNorm / sptensorNorm);
    }
    scalar_t fitchange = std::abs(fitold - fit);

    // Print progress of the current iteration.
    if (me == 0) 
      printf ("Iter %2d: fit = %13.6e  fitdelta = %8.1e\n",
              numIter+1, fit, fitchange);

    // Check for convergence. 
    if (((numIter >= (minIters-1)) && (fitchange < tolerance))) 
      break;
  }

  // Increment so the count starts from one.
  numIter++;

  // Normalize the final result, incorporating the final lambda values.
  {
    Teuchos::TimeMonitor 
           tm(*Teuchos::TimeMonitor::getNewTimer("CP-ALS 09 Cleanup"));

    ktensor->normalize();
    Kokkos::View<scalar_t *> ktensorLambda = ktensor->getLambdaView();
    for (rank_t r = 0; r < rank; r++) lambda(r) *= ktensorLambda(r);
    ktensor->setLambda(lambda);

    for (mode_t m = 0; m < nModes; m++) delete gamma[m];
  }

  if (me == 0)
    std::cout << "CP-ALS completed: " << numIter << " iterations" << std::endl;
}

//////////////////////////////////////////////////////////////////////////////
template <typename sptensor_t, typename ktensor_t>
void distSystem<sptensor_t, ktensor_t>::GCP_Adam(
  lossFunction<scalar_t> &lossFn,
  Teuchos::ParameterList &params
)
{

  Teuchos::RCP<Teuchos::Time> 
    timeSample(Teuchos::TimeMonitor::getNewTimer(
                                     "CP-ADAM 05 Stoc Grad   Sample")),
    timeSystem(Teuchos::TimeMonitor::getNewTimer(
                                     "CP-ADAM 05 Stoc Grad   System")), 
    timeDfDm(Teuchos::TimeMonitor::getNewTimer(
                                     "CP-ADAM 05 Stoc Grad   dF/dM")), 
    timeMTTKRP(Teuchos::TimeMonitor::getNewTimer(
                                     "CP-ADAM 05 Stoc Grad   MTTKRP"));

  Teuchos::RCP<Teuchos::Time>
    timeGCPAdam(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 00 Total")),
    timeOnetimeSetup(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 01 OneTimeSetup")),
    timeInitAndCopy(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 02 Init and Copy Ktensors")),
    timeFixedSys(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 03 Fixed System Constr")),
    timeSamplerSetup(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 04 Sampler Setup")),
    timeStocGrad(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 05 Stoc Grad")),
    timeLocal(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 06 LocalMatrixComp")),
    timeLossFn(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 07 Comp Loss Func")),
    timeRollBack(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 08 Roll Back"));

#ifdef TIME_ADAM_IMBALANCE
  Teuchos::RCP<Teuchos::Time>
    timeWaitOne(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 01 WaitBeforeOneTime")),
    timeWaitTwo(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 02 WaitBeforeInit")),
    timeWaitThree(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 03 WaitBeforeFixed")),
    timeWaitFour(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 04 WaitBeforeSampler")),
    timeWaitFive(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 05 WaitBeforeStocGrad")),
    timeWaitSix(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 06 WaitBeforeLocal")),
    timeWaitSeven(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 07 WaitBeforeLossFn")),
    timeWaitEight(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 08 WaitBeforeRollBack")),
    timeWaitNine(Teuchos::TimeMonitor::getNewTimer("CP-ADAM 09 WaitBeforeDone"));
#endif

  sptensor->getComm()->barrier();  // Needed only for timers
  timeGCPAdam->start();

  timeOnetimeSetup->start();
  typedef typename pt::distSystem<sptensor_t, ktensor_t> distsystem_t;

  const int np = sptensor->getComm()->getSize();
  const int me = sptensor->getComm()->getRank();

  // General Parameters
  bool debug = params.get("debug", false);  // print verbose info
  bool stats = params.get("stats", false);  // print stats at end (not timed)
  bool randomizeKtensor = params.get("randomizeKtensor", true);

  // Algorithm running parameters
  const int minEpochs = params.get("minEpochs", 1);
  const int defaultMaxEpochs = 1000;
  const int maxEpochs = params.get("maxEpochs", defaultMaxEpochs);
  const int maxBadEpochs = params.get("maxBadEpochs", 2);
  const int nIterPerEpoch = params.get("nIterPerEpoch", 1000);
  scalar_t tolerance = params.get("tol", 0.0001);  // minimum progress needed

  // Sampling parameters
  const std::string samplingType = params.get("sampling", "semi-stratified");
  int seed = params.get("seed", me+1);  // different default value on each proc

  // Adam parameters:  default values match those in tensor tool box (TTB)
  scalar_t alpha = params.get("alpha", .001);       // In TTB:  step 
                                // (decay ^ nfails * rate = 0.1 ^ 0 * 0.001)
  const scalar_t beta1 = params.get("beta1", .9);    // In TTB:  beta1
  const scalar_t beta2 = params.get("beta2", .999);  // In TTB:  beta2
  const scalar_t eps = params.get("epsilon", 1e-8);  // In TTB:  epsilon
  const scalar_t nu = params.get("decay", .1);       // In TTB:  decay
  const scalar_t lowerBound = lossFn.getLowerBound();

  //2:  for k = 1,2,...,d do
  //3:    Ak = random matrix of size nk * r
  //4:    Bk,Ck = all-zero matrices of size nk * r 
  //5:  end for

  sptensor_t *X = getSptensor();
  const size_t X_nnz = X->getLocalNumNonZeros();      // Local num nonzeros
  const double X_nz = X->getLocalTensorSize() - X_nnz;// Local num avail zeros
  size_t X_gnnz;  // Global num nonzeros
  Teuchos::reduceAll<int, size_t>(*(sptensor->getComm()),
                                    Teuchos::REDUCE_SUM, 1, &X_nnz, &X_gnnz);
  double X_gnz = 1.;  // double to prevent overflow
  for (mode_t m = 0; m < nModes; m++) X_gnz *= X->getModeSize(m);
  X_gnz -= X_gnnz;

  timeOnetimeSetup->stop();
#ifdef TIME_ADAM_IMBALANCE
  timeWaitTwo->start();
  sptensor->getComm()->barrier();
  timeWaitTwo->stop();
#endif
  timeInitAndCopy->start();

  if (randomizeKtensor) ktensor->setRandomUniform();
  distributeLambdaToKtensor(0); internalInSyncWithKtensor[0] = false;

  ktensor_t *A = getKtensor();
  ktensor_t *Acopy = new ktensor_t(A);

  ktensor_t *B = new ktensor_t(A);     // Get A's structure
  B->setValues(0.);                    // Store zero values
  ktensor_t *Bcopy = new ktensor_t(B);

  ktensor_t *C = new ktensor_t(B);
  ktensor_t *Ccopy = new ktensor_t(C);

  ktensor_t *grad = new ktensor_t(B);

  timeInitAndCopy->stop();

  //6:  F = EstObj(X, { Ak })  // estimate loss with fixed set of samples
  //    Sample size for fixed sample:  default from tensor toolbox (fsamp)
  // TensorToolBox's strategy:
  // const size_t nFixedTmp = std::max<size_t>(0.01 * X_nnz, 10^5);
  // size_t nFixedNonZeroSamples = std::min<size_t>(nFixedTmp, X_nnz);
  // size_t nFixedZeroSamples = std::min<size_t>(X_nz, nFixedNonZeroSamples);
  //
  // Distributed strategy:  aiming for same number of samples per proc
  // Sample nonzeros proportional to number of nonzeros on processor

#ifdef TIME_ADAM_IMBALANCE
  timeWaitThree->start();
  sptensor->getComm()->barrier();
  timeWaitThree->stop();
#endif
  timeFixedSys->start();
  // Proposal global number of samples
  //const size_t nFixedTmp = std::max<size_t>(size_t(0.01 * X_gnnz), 100000);
  //const size_t nFixedSamplesGlobal = std::min<size_t>(nFixedTmp, X_gnnz);
  size_t nFixedSamplesGlobal = params.get("fns", size_t(0));
  size_t nFixedNonZeroSamples = params.get("fnnz", size_t(0));
  size_t nFixedZeroSamples = params.get("fnz", size_t(0));
  
  if (nFixedSamplesGlobal > 0) { 
    // parameter fns has priority; puts same number of samples on each proc
    // Half the global samples are nonzeros; 
    // scale per processor numbers by number of nonzeros on processor
    nFixedNonZeroSamples = 0.5 * double(nFixedSamplesGlobal) 
                         * double(X_nnz) / double(X_gnnz);
    nFixedZeroSamples = nFixedSamplesGlobal / np - nFixedNonZeroSamples;
  }
  else {
    nFixedSamplesGlobal = 0;
    for (mode_t m = 0; m < nModes; m++) 
      nFixedSamplesGlobal += sptensor->getModeSize(m);
    nFixedSamplesGlobal *= 10;

    if (nFixedNonZeroSamples > 0) {
      // User specified number of samples; 
      // scale global request to each processor
      nFixedNonZeroSamples *= (double(X_nnz) / double (X_gnnz));
    }
    else {
      // Use a fraction of the proposed global number of samples
      nFixedNonZeroSamples = 
           size_t(double(nFixedSamplesGlobal) * 
                 (double(X_nnz) / double (X_gnnz))) / 2;
    }
  
    if (nFixedZeroSamples > 0) {
      // User specified number of samples; 
      // scale global request to each processor
      nFixedZeroSamples *= (X_nz / X_gnz);
    }
    else {
      // Use a fraction of the proposed global number of samples
      nFixedZeroSamples =
           std::min<size_t>(X_nz,
                            size_t(double(nFixedSamplesGlobal)/double(np)) 
                            - nFixedNonZeroSamples);
    }
  }

  timeFixedSys->stop();

  // Prepare for stochastic gradient computation
  // Sample size for stoc grad sample:  default from tensor toolbox (gsamp)
  // TensorToolBox's strategy
  // const size_t nStocGradTmp = std::max<size_t>(1000, 3 * X_nnz / maxEpochs);
  // size_t nStocGradNonZeroSamples = std::min<size_t>(nStocGradTmp, X_nnz);
  // size_t nStocGradZeroSamples = std::min<size_t>(X_nz, nStocGradNonZeroSamples);
  // 
  // Distributed strategy:  aiming for same number of samples per proc
  //const size_t nStocGradTmp = std::max<size_t>(1000, 3*X_gnnz/maxEpochs);
  //const size_t nStocGradTmp = std::max<size_t>(1000, 3*X_gnnz/defaultMaxEpochs);
  //const size_t nStocGradSamplesGlobal = std::min<size_t>(nStocGradTmp, X_gnnz);
#ifdef TIME_ADAM_IMBALANCE
  timeWaitFour->start();
  sptensor->getComm()->barrier();
  timeWaitFour->stop();
#endif
  timeSamplerSetup->start();
  // Proposed global number of samples
  size_t nStocGradSamplesGlobal = params.get("gns", size_t(0));
  size_t nStocGradNonZeroSamples = params.get("gnnz", size_t(0));
  size_t nStocGradZeroSamples = params.get("gnz", size_t(0));

  if (nStocGradSamplesGlobal > 0) { 
    // parameter gns has priority; puts same number of samples on each proc
    // Half the global samples are nonzeros; 
    // scale per processor numbers by number of nonzeros on processor
    nStocGradNonZeroSamples = 0.5 * double(nStocGradSamplesGlobal) 
                            * double(X_nnz) / double(X_gnnz);
    nStocGradZeroSamples = nStocGradSamplesGlobal/np - nStocGradNonZeroSamples;
  }
  else {
    nStocGradSamplesGlobal = nFixedSamplesGlobal / 10;

    if (nStocGradNonZeroSamples > 0) {
      // User specified number of samples; 
      // scale global request to each processor
      nStocGradNonZeroSamples *= (double(X_nnz) / double(X_gnnz));
    }
    else {
      // Use a fraction of the proposed global number of samples
      nStocGradNonZeroSamples = 
           size_t(double(nStocGradSamplesGlobal) * 
                 (double(X_nnz) / double (X_gnnz))) / 2;
    }
  
    if (nStocGradZeroSamples > 0) {
      // User specified number of samples; 
      // scale global request to each processor
      nStocGradZeroSamples *= (double(X_nz) / X_gnz);
    }
    else {
      // Use a fraction of the proposed global number of samples
      nStocGradZeroSamples = 
           std::min<size_t>(X_nz,
                            size_t(double(nStocGradSamplesGlobal)/double(np)) 
                            - nStocGradNonZeroSamples);
    }
  }

  SamplingStrategy<sptensor_t> *sampler = NULL; 
  if (samplingType == "semi-stratified") 
    sampler = new SemiStratifiedSamplingStrategy<sptensor_t>(X);
  else if (samplingType == "stratified")
    sampler = new StratifiedSamplingStrategy<sptensor_t>(X);
  else {
    std::cout << "Invalid sampling strategy " << samplingType << std::endl;
    throw std::runtime_error("Invalid sampling strategy ");
  }

  sptensor_t *stocGradSampleTensor = NULL; 
  timeSamplerSetup->stop();

  if (stats) {
    // Stop times and gather / print info
    timeGCPAdam->stop();
    size_t lsize[7] = {nFixedZeroSamples, nFixedNonZeroSamples,
                       nFixedZeroSamples+nFixedNonZeroSamples,
                       nStocGradZeroSamples, nStocGradNonZeroSamples,
                       nStocGradZeroSamples+nStocGradNonZeroSamples,
                       X_nnz};
    size_t gsum[7], gmax[7], gmin[7];
    Teuchos::reduceAll<int, size_t>(*(sptensor->getComm()),
                                    Teuchos::REDUCE_SUM, 7, lsize, gsum);
    Teuchos::reduceAll<int, size_t>(*(sptensor->getComm()),
                                    Teuchos::REDUCE_MAX, 7, lsize, gmax);
    Teuchos::reduceAll<int, size_t>(*(sptensor->getComm()),
                                    Teuchos::REDUCE_MIN, 7, lsize, gmin);

    if (me == 0)  {
      std::cout << std::endl;
      std::cout << "Fixed sample:  \tstratified with \t"
                << gsum[1] << " (out of " << gsum[6] << ") nonzeros and \t"
                << gsum[0] << "  zeros "
                << std::endl;
      std::cout << "Fixed sample nonzeros:  per-proc min/max/avg " 
                << gmin[1] << " " << gmax[1] << " " << gsum[1] / np
                << std::endl;
      std::cout << "Fixed sample zeros:     per-proc min/max/avg " 
                << gmin[0] << " " << gmax[0] << " " << gsum[0] / np
                << std::endl;
      std::cout << "Fixed sample indices:   per-proc min/max/avg " 
                << gmin[2] << " " << gmax[2] << " " << gsum[2] / np
                << std::endl;
  
      std::cout << "StocGrad sample: \t" << samplingType << " with \t"
                << gsum[4] << " (out of " << gsum[6] << ") nonzeros and \t"
                << gsum[3] << "  zeros "
                << std::endl;
      std::cout << "StocGrad sample nonzeros:  per-proc min/max/avg " 
                << gmin[4] << " " << gmax[4] << " " << gsum[4] / np
                << std::endl;
      std::cout << "StocGrad sample zeros:     per-proc min/max/avg " 
                << gmin[3] << " " << gmax[3] << " " << gsum[3] / np
                << std::endl;
      std::cout << "StocGrad sample indices:   per-proc min/max/avg " 
                << gmin[5] << " " << gmax[5] << " " << gsum[5] / np
                << std::endl;
    }
    timeGCPAdam->start();
  }

#ifdef TIME_ADAM_IMBALANCE
  timeWaitThree->start();
  sptensor->getComm()->barrier();
  timeWaitThree->stop();
#endif
  // Create fixed sampled tensor for error computation
  timeFixedSys->start();
  sptensor_t *fixedSampleTensor = X->stratSampledTensor(nFixedNonZeroSamples,
                                                        nFixedZeroSamples);

  distsystem_t fixedSystem(fixedSampleTensor, A);

  timeFixedSys->stop();

#ifdef TIME_ADAM_IMBALANCE
  timeWaitFive->start();
  sptensor->getComm()->barrier();
  timeWaitFive->stop();
#endif
  timeLossFn->start();

  scalar_t fixedError = fixedSystem.computeLossFn(lossFn);

  timeLossFn->stop();


#ifdef TIME_ADAM_IMBALANCE
  timeWaitOne->start();
  sptensor->getComm()->barrier();
  timeWaitOne->stop();
#endif
  timeOnetimeSetup->start();
  //7:  c = 0
  //8:  t = 0 //t=#ofAdamiterations
  int nBadEpochs = 0;
  int nEpochs = 0;
  int nAdamIterations = 0;
  scalar_t makingProgress = std::numeric_limits<scalar_t>::max();

  if (me == 0) 
    std::cout << "Epoch 0: fixed error = " << fixedError << std::endl;
  timeOnetimeSetup->stop();

  //9:  while c <= kappa do //#=max#ofbadepochs
  while ((nEpochs < minEpochs) ||  // minimum number of epochs required
         ((nEpochs < maxEpochs) && (nBadEpochs < maxBadEpochs) && 
          (makingProgress > tolerance))) {

#ifdef TIME_ADAM_IMBALANCE
    timeWaitTwo->start();
    sptensor->getComm()->barrier();
    timeWaitTwo->stop();
#endif
    timeInitAndCopy->start();

    nEpochs++;
    seed += nEpochs;  // Don't use same sampling in each epoch

    //10:   Save copies of {Ak }, {Bk }, {Ck }; don't care about lambda

    Acopy->copyData(A, false);
    Bcopy->copyData(B, false);
    Ccopy->copyData(C, false);

    //11:   Fold = F
    scalar_t fixedErrorOld = fixedError;

    timeInitAndCopy->stop();

    //12:   for tau iterations do
    for (int iter = 0; iter < nIterPerEpoch; iter++) {
    
#ifdef TIME_ADAM_IMBALANCE
      timeWaitFive->start();
      sptensor->getComm()->barrier();
      timeWaitFive->stop();
#endif
      timeStocGrad->start();

      stocGrad(nStocGradNonZeroSamples, nStocGradZeroSamples, lossFn, grad,
               stocGradSampleTensor, sampler, seed * iter,
               timeSample, timeSystem, timeDfDm, timeMTTKRP);

      timeStocGrad->stop();

      //22:     t = t+1 -- KDD Must be done before computing divb, divc;
      //                   KDD if not, will have div by zero in first iteration
#ifdef TIME_ADAM_IMBALANCE
      timeWaitSix->start();
      sptensor->getComm()->barrier();
      timeWaitSix->stop();
#endif
      timeLocal->start();

      nAdamIterations++;

      scalar_t divb = (1. - std::pow<scalar_t>(beta1, nAdamIterations));
      scalar_t divc = (1. - std::pow<scalar_t>(beta2, nAdamIterations));
      
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
        gcp_adam_lines_17_thru_20(a, b, c, alpha, eps, lowerBound, 
                                  divb, divc);
      } //21:     end for

      timeLocal->stop();
    } //23:   end for

#ifdef TIME_ADAM_IMBALANCE
    timeWaitSeven->start();
    sptensor->getComm()->barrier();
    timeWaitSeven->stop();
#endif
    timeLossFn->start();

    //24:   F = EstObj(X, { Ak })
    // Factor matrices have changed; need to recommunicate them to temporary
    // factor matrices for model evaluation
    fixedSystem.updateAllInternalFactorMatricesFromKtensor(true);  
    fixedError = fixedSystem.computeLossFn(lossFn);
    makingProgress = std::abs(fixedErrorOld - fixedError);

    timeLossFn->stop();

    if (me == 0) 
      std::cout << "Epoch " << nEpochs << ": fixed error = " << fixedError
                << "\t Delta = " << fixedErrorOld - fixedError 
                << "\t Step = " << alpha
                << (fixedError > fixedErrorOld ? "\tBAD " : " ")
                << std::endl;


#ifdef TIME_ADAM_IMBALANCE
    timeWaitEight->start();
    sptensor->getComm()->barrier();
    timeWaitEight->stop();
#endif
    timeRollBack->start();
    //25:   if F > Fold then
    if (fixedError > fixedErrorOld) {

      // No progress; Restore state and try smaller step
      //26:     Restore saved copied of { Ak }, { Bk }, { Ck }
      //        Don't care about lambda
      A->copyData(Acopy, false);
      B->copyData(Bcopy, false);
      C->copyData(Ccopy, false);

      //27:     F =  Fold
      //28:     t = t - tau
      //29:     alpha =  alpha * nu
      //30:     c = c+1
      fixedError = fixedErrorOld;
      nAdamIterations -= nIterPerEpoch;
      alpha *= nu;
      nBadEpochs++;

    }  //31:   end if
    timeRollBack->stop();
  }  //32: end while

  //33: return { Ak }
  //34: end function

#ifdef TIME_ADAM_IMBALANCE
  timeWaitOne->start();
  sptensor->getComm()->barrier();
  timeWaitOne->stop();
#endif
  timeOnetimeSetup->start();
  // Clean up
  delete sampler;
  delete Acopy;
  delete B;
  delete Bcopy;
  delete C;
  delete Ccopy;
  delete grad;

  // Changed ktensor, so need to invalidate internal factor matrices
  for (mode_t m = 0; m < nModes; m++) internalInSyncWithKtensor[m] = false;

  timeOnetimeSetup->stop();

  timeGCPAdam->stop();

  // Statistics
  if (debug) {
    std::cout << me << " Per-processor Fixed samples:  " 
              << nFixedNonZeroSamples << " (out of " << X_nnz 
              << ") nonzeros and "
              << nFixedZeroSamples << " (out of " << X_nz << ") zeros"
              << std::endl;
    std::cout << me << " Per-processor StocGrad samples:  " 
              << nStocGradNonZeroSamples << " (out of " << X_nnz 
              << ") nonzeros and "
              << nStocGradZeroSamples << " (out of " << X_nz << ") zeros"
              << std::endl;
  }


  
  if (stats) {

    updateAllInternalFactorMatricesFromKtensor(true);  
 //   scalar_t actualLossFn = computeLossFn(lossFn);

    if (me == 0) {
      std::cout << "\nDONE  StocGrad Sampling: " << samplingType
                << "; nEpochs = " << nEpochs 
                << "; nIterations = " << nEpochs * nIterPerEpoch
                << "; LossFn = " << lossFn.name()
//                << "; Loss = " << actualLossFn
//                << "; sqrt(Loss) = " << std::sqrt(actualLossFn)
                << std::endl;
    }

    if (lossFn.name() == "L2") {
      scalar_t resNorm = getResidualNorm();
      if (me == 0) {
        std::cout << "Residual norm = " << resNorm 
                  << "Residual norm squared = " << resNorm * resNorm
                  << std::endl;
      }
    }
    if (me == 0) std::cout << std::endl;
  }

  delete fixedSampleTensor;     // Don't used fixedSystem after this
  delete stocGradSampleTensor;
}

//////////////////////////////////////////////////////////////////////////////
template <typename sptensor_t, typename ktensor_t>
void distSystem<sptensor_t, ktensor_t>::stocGrad (
  size_t nRequestedNonZeroSamples, 
  size_t nRequestedZeroSamples, 
  lossFunction<scalar_t> &f,
  ktensor_t* gradient, 
  sptensor_t *&Y,
  SamplingStrategy<sptensor_t> *sampler,
  unsigned int seed,
  Teuchos::RCP<Teuchos::Time> &timeSample,
  Teuchos::RCP<Teuchos::Time> &timeSystem,
  Teuchos::RCP<Teuchos::Time> &timeDfDm,
  Teuchos::RCP<Teuchos::Time> &timeMTTKRP
)
{
#ifdef TIME_SG_IMBALANCE
  Teuchos::RCP<Teuchos::Time> 
    timeModelEval(Teuchos::TimeMonitor::getNewTimer( "CP-ADAM 05 SG ModelEval")),
    timeFirstWait(Teuchos::TimeMonitor::getNewTimer( "CP-ADAM 05a1 SG WaitBeforeSample")),
    timeSecondWait(Teuchos::TimeMonitor::getNewTimer( "CP-ADAM 05a2 SG WaitBeforeSystem")), 
    timeThirdWait(Teuchos::TimeMonitor::getNewTimer( "CP-ADAM 05a3 SG WaitBeforeDfDM")), 
    timeFourthWait(Teuchos::TimeMonitor::getNewTimer( "CP-ADAM 05a4 SG WaitBeforeMTTKRP")),
    timeFifthWait(Teuchos::TimeMonitor::getNewTimer( "CP-ADAM 05a5 SG WaitBeforeReturn"))
  ;
#endif


  typedef Kokkos::View<scalar_t *> valueview_t;
  typedef Kokkos::View<gno_t **> gnoview_t;
  
  scalar_t modelValue;
  Kokkos::View<scalar_t*> prod("prod", rank);
    
#ifdef TIME_SG_IMBALANCE
  timeFirstWait->start();
  sptensor->getComm()->barrier();
  timeFirstWait->stop();
#endif

  // create new sampled sptensor with same entry values as original sptensor
#ifdef TIME_DETAIL_STOCGRAD
  timeSample->start();
#endif

  if (Y != NULL) {
    Y->resample(seed);
  }
  else {
    Teuchos::RCP<SamplingStrategy<sptensor_t> > samplerRCP = 
                                                Teuchos::rcp(sampler, false);
    Y = sptensor->getSampledTensor(samplerRCP, nRequestedNonZeroSamples,
                                               nRequestedZeroSamples, seed);
  }

  bool semiStrat = !(sampler->isStratified());

#ifdef TIME_DETAIL_STOCGRAD
  timeSample->stop();
#endif
    
  // create new system with sampled sptensor and existing ktensor
  // to obtain rows of ktensor corresponding to local sampled entries

#ifdef TIME_SG_IMBALANCE
  timeSecondWait->start();
  sptensor->getComm()->barrier();
  timeSecondWait->stop();
#endif

#ifdef TIME_DETAIL_STOCGRAD
  timeSystem->start();
#endif

  distSystem<sptensor_t, ktensor_t> stocGradSystem(Y, ktensor);
    
#ifdef TIME_DETAIL_STOCGRAD
  timeSystem->stop();
#endif

  // loop through samples and update values following Kolda&Hong, Alg. 4.3
#ifdef TIME_SG_IMBALANCE
  timeThirdWait->start();
  sptensor->getComm()->barrier();
  timeThirdWait->stop();
#endif

#ifdef TIME_DETAIL_STOCGRAD
  timeDfDm->start();
#endif

  gnoview_t indices = Y->getGlobalIndices();
  valueview_t values = Y->getValues();

  // set scaling parameters as scalar_t's to avoid integer division
  scalar_t eta = sptensor->getLocalNumNonZeros();   // # nonzeros in orig tensor
  scalar_t omega = sptensor->getLocalTensorSize();  // # indices in orig tensor
  scalar_t zeta = omega - eta;                      // # zeros in orig tensor

  size_t nSampledNonZeros = Y->getLocalNumNonZeros(); // # sampled nonzeros
  size_t nSampledZeros = Y->getLocalNumZeros();       // # sampled zeros
  size_t nIndices = Y->getLocalNumIndices();          // # sampled indices

  if (semiStrat) {
    for (size_t i = 0; i < nSampledNonZeros; i++) {  // for each nonzero
      // compute corresponding model entry value (pass in scratch memory)
      modelValue = stocGradSystem.computeModelAtIndex(i, prod);

      // use formulae from Kolda&Hong: f.dfdm corresponds to g function
      // includes correction term for semi-stratified sampling
      values(i) = eta / nSampledNonZeros * 
                  (f.dfdm(values(i),modelValue) - f.dfdm(0,modelValue));
    }

    for (size_t i = nSampledNonZeros; i < nIndices; i++) { // for each zero

      modelValue = stocGradSystem.computeModelAtIndex(i, prod);
      values(i) = omega / nSampledZeros * f.dfdm(0,modelValue);
    }
  }
  else {  // stratified
    for (size_t i = 0; i < nSampledNonZeros; i++) {  // for each nonzero
      modelValue = stocGradSystem.computeModelAtIndex(i, prod);
      values(i) = eta / nSampledNonZeros * f.dfdm(values(i),modelValue);
    }

    for (size_t i = nSampledNonZeros; i < nIndices; i++) { // for each zero
      modelValue = stocGradSystem.computeModelAtIndex(i, prod);
      values(i) = zeta / nSampledZeros * f.dfdm(0,modelValue);
    }
  }
        
#ifdef TIME_DETAIL_STOCGRAD
  timeDfDm->stop();
#endif
      
  // store MTTKRP results in gradient ktensor
#ifdef TIME_SG_IMBALANCE
  timeFourthWait->start();
  sptensor->getComm()->barrier();
  timeFourthWait->stop();
#endif

#ifdef TIME_DETAIL_STOCGRAD
  timeMTTKRP->start();
#endif

  for (mode_t m = 0; m < nModes; m++) {
    stocGradSystem.mttkrp(m,gradient->getFactorMatrix(m));
  }

#ifdef TIME_DETAIL_STOCGRAD
  timeMTTKRP->stop();
#endif

#ifdef TIME_SG_IMBALANCE
  timeFifthWait->start();
  sptensor->getComm()->barrier();
  timeFifthWait->stop();
#endif
}


//////////////////////////////////////////////////////////////////////////////
template <typename sptensor_t, typename ktensor_t>
typename distSystem<sptensor_t, ktensor_t>::scalar_t 
  distSystem<sptensor_t, ktensor_t>::innerprod (
    Kokkos::View<typename distSystem<sptensor_t, ktensor_t>::scalar_t *> 
                                                             &scaleValues
)
{
  Teuchos::TimeMonitor
         tm(*Teuchos::TimeMonitor::getNewTimer("Innerprod"));

  // Get views from internalFactorMatrices; 
  // need internalFactorMatrices to match up with sptensor's entries.
  std::vector<factormatrixview_t> midFMView(nModes);
  for (mode_t m = 0; m < nModes; m++) {
    midFMView[m] = internalFactorMatrices[m]->getLocalView();

    // This request for communication looks worse than it probably is.
    // In CP-ALS, most internal factor matrices will already be up-to-date;
    // perhaps one factor matrix will need to be communicated.
    updateInternalFactorMatrixFromKtensor(m);
  }

  // Compute local terms of inner product (i.e., for local nonzeros)
  scalar_t localresult = 0.;
  size_t nnz = sptensor->getLocalNumNonZeros();
  typename sptensor_t::lnoview_t localIndices = sptensor->getLocalIndices();
  typename sptensor_t::valueview_t spvalues = sptensor->getValues();

  for (size_t nz = 0; nz < nnz; nz++) {

    Kokkos::View<scalar_t *> ktensorEntry("ktensorEntry", rank);
    Kokkos::deep_copy(ktensorEntry, scaleValues);

    for (mode_t m = 0; m < nModes; m++) {

      // Get the local index in the factor matrix
      lno_t idx = localIndices(nz, m);

      // For each rank...
      for (rank_t r = 0; r < rank; r++) {
        ktensorEntry(r) *= midFMView[m](idx, r);
      }
    }
    
    scalar_t sumKtensorEntry = 0.;
    for (rank_t r = 0; r < rank; r++)
      sumKtensorEntry += ktensorEntry(r);

    localresult += (spvalues(nz) * sumKtensorEntry);
  }

  // Accumulate global result
  scalar_t globalresult;
  Teuchos::reduceAll<int, scalar_t>(*(sptensor->getComm()),
                                    Teuchos::REDUCE_SUM, 1, 
                                    &localresult, &globalresult);

  return globalresult;
}


//////////////////////////////////////////////////////////////////////////////
template <typename sptensor_t, typename ktensor_t>
void distSystem<sptensor_t, ktensor_t>::printStats(
  const std::string &msg, std::ostream &ostr) const
{
  int me = sptensor->getComm()->getRank();

  if (me == 0) 
    ostr << "SYSSTATS  DistributedSystem " << msg << std::endl;

  // Print stats for the sparse tensor first
  sptensor->printStats(msg);
  if (me == 0) ostr << std::endl << "----------------" << std::endl;

  // Print stats for the ktensor next
  ktensor->printStats(msg);
  if (me == 0) ostr << std::endl << "----------------" << std::endl;

  // Importer stats
  for (mode_t m = 0; m < nModes; m++) 
    printImportStats(importers[m], m, ostr);

  if (me == 0)
    ostr << std::endl 
         << "****************************************************" << std::endl;
}

//////////////////////////////////////////////////////////////////////////////
template <typename sptensor_t, typename ktensor_t>
typename sptensor_t::scalar_t distSystem<sptensor_t, ktensor_t>::computeLossFn(
  lossFunction<scalar_t> &lossFn
)
{
  // Need up-to-date internal factor matrices; import if they are not up-to-date
  updateAllInternalFactorMatricesFromKtensor();

  // Get the sptensor values
  typename sptensor_t::valueview_t sptensorValues = sptensor->getValues();

  // Variables to accrue the entrywise loss; separate zeros' and nonzeros' loss
  // lLoss[0] is loss for zero entries; lLoss[1] is loss for nonzero entries
  scalar_t lLoss[2] = {0., 0.};  // on-processor loss
  scalar_t gLoss[2];             // global loss

  size_t nNonZeros = sptensor->getLocalNumNonZeros();
  size_t nZeros = sptensor->getLocalNumZeros();
  size_t nIndices = sptensor->getLocalNumIndices();

  // Temporary storage to allow iteration over all ranks of a factor matrix 
  // index
  Kokkos::View<scalar_t*> prod("prod", rank);

  // Loop over all indices in tensor
  for (size_t n = 0; n < nIndices; n++) {

    // Compute loss function; accrue the appropriate loss variable for the index
    scalar_t valTensor = sptensorValues(n);
    scalar_t valModel = computeModelAtIndex(n, prod);
    lLoss[n < nNonZeros] += lossFn(valTensor, valModel);
  }

  // Scaling zero and nonzero loss from Kolda&Hong, eqn. 5.2
  // eta = number of local nonzeros from which sptensor is sampled
  scalar_t eta = sptensor->getSourceTensor()->getLocalNumNonZeros();
  // zeta = number of local zeros from which sptensor is sampled
  scalar_t zeta = sptensor->getLocalTensorSize() - eta;

  if (nZeros)    lLoss[0] = lLoss[0] * zeta / scalar_t(nZeros);
  if (nNonZeros) lLoss[1] = lLoss[1] * eta / scalar_t(nNonZeros);
  
  // Gather loss across processors
  Teuchos::reduceAll<int, scalar_t>(*(sptensor->getComm()), Teuchos::REDUCE_SUM,
                                    2, lLoss, gLoss);

  return (gLoss[0] + gLoss[1]);
}

}

#endif
