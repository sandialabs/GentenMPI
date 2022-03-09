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
 
#include "Teuchos_TimeMonitor.hpp"
#include "Tpetra_Core.hpp"
#include "pt_system.hpp"
#include "pt_random.hpp"
#include "pt_read.hpp"
#include "pt_read_by_box.hpp"

//////////////////////////////////////////////////////////////////////////////
template <typename T>
void stringToSizes(std::string &str, std::vector<T> &sizes)
{
  // parse string of format IxJxKx...; store I, J, K, ..., in vector
  std::string x("x");
  size_t start = 0U;
  size_t end = str.find(x);
  while (end != std::string::npos) {
    sizes.push_back(T(std::stol(str.substr(start, end - start))));
    start = end + x.length();
    end = str.find(x, start);
  }
  sizes.push_back(T(std::stol(str.substr(start, end))));
}

//////////////////////////////////////////////////////////////////////////////

int main(int narg, char *arg[])
{
  typedef double scalar_t;

  // Usual Teuchos::Comm initialization
  Tpetra::ScopeGuard scopeguard(&narg, &arg);
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();

  int me = comm->getRank();
  int ierr = 0;

  // Initialize Teuchos timer
  Teuchos::RCP<Teuchos::Time>
         timeFileRead(Teuchos::TimeMonitor::getNewTimer("00 FILE READ")),
         timeRandom(Teuchos::TimeMonitor::getNewTimer("00 RANDOM GENERATE")),
         timeAdam(Teuchos::TimeMonitor::getNewTimer("01 DISTRIBUTED CP-ALS"));

  // Accept command-line parameters
  Teuchos::CommandLineProcessor cmdp (false, false);

  // options for all problems
  int verbosity = 1;
  cmdp.setOption("verbosity", &verbosity,
                 "0 = quiet; 1 = print status; 2 = print factor matrices; "
                 "3 = print tensor");

  // factor matrix parameters
  pt::rank_t rank = 10;
  cmdp.setOption("rank", &rank, "number of components; int >= 1");

  int randomizeKtensor = 0;
  cmdp.setOption("randomizeKtensor", &randomizeKtensor,
                 "(0/1):  use random initial ktensor in Adam");

  int funkyInitKtensor = 0;
  cmdp.setOption("funkyInitKtensor", &funkyInitKtensor, 
                 "(0/1):  consistent (across different number of "
                 "processors) random initialization of factor matrices; "
                 "for testing only");

  // GCP-Adam parameters
  size_t fns = 0;
  cmdp.setOption("fns", &fns, 
                 "global number of samples for fixed error tensor");
  size_t fnnz = 0;
  cmdp.setOption("fnnz", &fnnz, 
                 "global number of NONZERO samples for fixed error tensor");
  size_t fnz = 0;
  cmdp.setOption("fnz", &fnz, 
                 "global number of ZERO samples for fixed error tensor");

  size_t gns = 0;
  cmdp.setOption("gns", &gns, 
                 "global number of samples for stocGrad tensor");
  size_t gnnz = 0;
  cmdp.setOption("gnnz", &gnnz, 
                 "global number of NONZERO samples for stocGrad tensor");
  size_t gnz = 0;
  cmdp.setOption("gnz", &gnz, 
                 "global number of ZERO samples for stocGrad tensor");

  int minEpochs = 1;
  cmdp.setOption("minEpochs", &minEpochs, "minimum number of epochs; int >= 1");

  int maxEpochs = 1000;
  cmdp.setOption("maxEpochs", &maxEpochs, "maximum number of epochs; int >= 1");

  int maxBadEpochs = 2;
  cmdp.setOption("maxBadEpochs", &maxBadEpochs,
                 "maximum number of bad epochs; int >= 1");

  int nIterPerEpoch = 1000;
  cmdp.setOption("nIterPerEpoch", &nIterPerEpoch,
                 "number of iterations per epoch; int >= 1");

  scalar_t tol = 1.0e-3;
  cmdp.setOption("tol", &tol, "tolerance for adam progress");

  std::string samplingType("semi-stratified");
  cmdp.setOption("sampling", &samplingType, 
                 "type of sampling to use in stochastic gradient: "
                 "stratified or semi-stratified");

  std::string lossFunction("gaussian");
  cmdp.setOption("type", &lossFunction,
                 "type of loss function to minimize: gaussian or poisson");

  int seed = 1;
  cmdp.setOption("seed", &seed, "random seed to use in sampling");

  scalar_t alpha = 0.001;
  cmdp.setOption("alpha", &alpha, "initial step size");
 
  scalar_t beta1 = 0.9;
  cmdp.setOption("beta1", &beta1, "ADAM beta1 parameter");

  scalar_t beta2 = 0.999;
  cmdp.setOption("beta2", &beta2, "ADAM beta2 parameter");

  scalar_t epsilon = 1e-8; 
  cmdp.setOption("epsilon", &epsilon, "ADAM epsilon parameter");
  
  scalar_t nu = 0.1;
  cmdp.setOption("decay", &nu, "ADAM decay rate for failed steps");

  // Parallel distribution options
  std::string diststring;
  cmdp.setOption("dist", &diststring,
                 "processor layout for distribution: IxJxKx...");

  int optimizeMaps = 0;
  cmdp.setOption("optimizeMaps", &optimizeMaps, 
                 "let Tpetra optimize the tensor's maps for import operations");

  int oneToOneMaps = 0;
  cmdp.setOption("oneToOneMaps", &oneToOneMaps, 
                 "use KTensor maps that are derived from the SpTensor maps; "
                 "used only with file-based input");

  // options for randomly generated problem
  size_t randomsize = 0;
  cmdp.setOption("random", &randomsize, 
                 "number of nonzeros to randomly generate");

  std::string modestring;
  cmdp.setOption("modes", &modestring, "mode sizes for the tensor: IxJxKx...");

  // options for reading file a la SPLATT
  std::string filename("NoFiLeNaMe");
  cmdp.setOption("file", &filename, "COO-formatted file containing tensor");

  std::string initDecomp("medGrain");
  cmdp.setOption("initDecomp", &initDecomp, 
                 "For COO-formatted files, distribute into "
                 "uniform bounding boxes (uniformBox) or "
                 "medium grain decomp from SPLATT (medGrain)");

  cmdp.parse(narg, arg);

  if (randomizeKtensor && funkyInitKtensor) {
    if (me == 0) {
      std::cout << "Cannot use both randomizeKtensor and funkyInitKtensor"
                << std::endl;
    }
    return -1;
  }

  if (optimizeMaps && oneToOneMaps) {
    if (me == 0) {
      std::cout << "Cannot use both optimizeMaps and oneToOneMaps"
                << std::endl;
    }
    return -1;
  }

  Teuchos::ParameterList params;
  params.set("alpha", alpha);
  params.set("beta1", beta1);
  params.set("beta2", beta2);
  params.set("decay", nu);
  params.set("epsilon", epsilon);
  params.set("randomizeKtensor", bool(randomizeKtensor));
  params.set("minEpochs", minEpochs);
  params.set("maxEpochs", maxEpochs);
  params.set("maxBadEpochs", maxBadEpochs);
  params.set("nIterPerEpoch", nIterPerEpoch);
  params.set("tolerance", tol);
  params.set("sampling", samplingType);
  params.set("type",lossFunction);
  params.set("seed", seed);
  params.set("stats", true);
  params.set("fns", fns);
  params.set("fnnz", fnnz);
  params.set("fnz", fnz);
  params.set("gns", gns);
  params.set("gnnz", gnnz);
  params.set("gnz", gnz);

  typedef typename pt::distSptensor<scalar_t> ptsptensor_t;
  typedef typename pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename pt::distKtensor<factormatrix_t> ktensor_t;
  typedef typename pt::distSystem<ptsptensor_t, ktensor_t> distsystem_t;

  // Create a distributed system
  distsystem_t *distSys = NULL;
  ptsptensor_t *sptensor = NULL;
  ktensor_t *ktensor = NULL;

  if (randomsize) {
    // Generate random sptensor and ktensor
    if ((diststring == "") || (modestring == "")) {
      std::cout << "Must specify dist and modes for random tensor generation"
                << std::endl;
      return -1;
    }
    try {
      Teuchos::TimeMonitor lcltimer(*timeRandom);
      std::vector<size_t> modeSizes;
      stringToSizes<size_t>(modestring, modeSizes);
      std::vector<int> procsPerMode;
      stringToSizes<int>(diststring, procsPerMode);

      pt::randomData<ptsptensor_t, ktensor_t> rd(comm);
      // Generate random ktensor with appropriate dimensions
      ktensor = rd.createRandomKtensor(rank, modeSizes);

      // Generate nonzero elements in sptensor relative to ktensor
      sptensor = rd.createRandomSptensor(ktensor, randomsize, procsPerMode);
      //sptensor->printStats("RandomSptensor");
    }
    catch (std::exception &e) {
      std::cout << "FAIL: Error caught in random generation " 
                << e.what() << std::endl;
      return -1;
    }
  }

  else if (filename != "NoFiLeNaMe") {

    // Read using SPLATT formatted coordinate file
    try {
      Teuchos::TimeMonitor lcltimer(*timeFileRead);

      if (initDecomp == "uniformBox")
        sptensor = readWithUniformBlocks<ptsptensor_t>(filename, comm);

      else if (initDecomp == "medGrain")
        sptensor = readUsingSplattIO<ptsptensor_t>(filename, diststring, 
                                                   false, comm);
      else {
        std::cout << "FAIL:  invalid initDecomp option " << initDecomp
                  << std::endl;
        return -1;
      }
    }
    catch (std::exception &e) {
      std::cout << "FAIL:  Error caught from file reading "
                << e.what() << std::endl;
      return -1;
    }
    //sptensor->printStats("FileSptensor");

    if (funkyInitKtensor) {
      if (oneToOneMaps)
        ktensor = new ktensor_t(rank, sptensor, comm);
      else
        ktensor = new ktensor_t(rank, sptensor->getModeSizes(), comm);
      ktensor->setFunky();
    }
    else {
      // Build random ktensor with default maps
      pt::randomData<ptsptensor_t, ktensor_t> rd(comm);
      if (oneToOneMaps)
        ktensor = rd.createRandomKtensor(rank, sptensor);
      else
        ktensor = rd.createRandomKtensor(rank, sptensor->getModeSizes());
    }
  }
  else {
    std::cout << "FAIL:  No input specifications provided" << std::endl;
    return -1;
  }

  // Some sanity-checking output
  {
  double fb = sptensor->frobeniusNorm();
  if (me == 0) 
    std::cout << "SPTensor complete; Frobenius norm = " << fb << std::endl;

  pt::mode_t nModes = ktensor->getNumModes();

  auto lv = ktensor->getLambdaView();
  if (me == 0) {
    std::cout << "KTensor complete; Lambda = ";
    for (pt::mode_t m = 0; m < nModes; m++) std::cout << lv(m) << " ";
    std::cout << std::endl;
  }
  }

  // Create system and print some useful stats
  distSys = new distsystem_t(sptensor, ktensor, 
                             distsystem_t::UPDATE_ALL, optimizeMaps);
  distSys->printStats(randomsize ? "random " : filename.c_str());
  
  // Do Adam
  pt::L2_lossFunction<scalar_t> l2;
  pt::Poisson_lossFunction<scalar_t> poisson;

  if (me == 0) std::cout << "Calling DIST ADAM" << std::endl;

  {
    Teuchos::TimeMonitor lcltimer(*timeAdam);
    if(lossFunction == "gaussian")
      distSys->GCP_Adam(l2,params);
    if(lossFunction == "poisson")
      distSys->GCP_Adam(poisson,params);
  }

  // Clean up
  delete distSys;
  delete sptensor;
  delete ktensor;

  // Report the timers
  Teuchos::TimeMonitor::summarize();
  Teuchos::TimeMonitor::zeroOutTimers();

  int gierr = 0;
  Teuchos::reduceAll<int,int>(*comm, Teuchos::REDUCE_SUM, 1, &ierr, &gierr);

  if (me == 0) {
    if (gierr == 0) std::cout << " PASS" << std::endl;
    else            std::cout << " FAIL" << std::endl;
  }

  return gierr;
}
