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
  // Usual Teuchos::Comm initialization
  Tpetra::ScopeGuard scopeguard(&narg, &arg);
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();

  int me = comm->getRank();
  int ierr = 0;

  // Initialize Teuchos timer
  Teuchos::RCP<Teuchos::Time>
         timeFileRead(Teuchos::TimeMonitor::getNewTimer("00 FILE READ")),
         timeRandom(Teuchos::TimeMonitor::getNewTimer("00 RANDOM GENERATE")),
         timeCpAls(Teuchos::TimeMonitor::getNewTimer("01 DISTRIBUTED CP-ALS"));

  // Accept command-line parameters
  Teuchos::CommandLineProcessor cmdp (false, false);

  // options for all problems
  int verbosity = 1;
  cmdp.setOption("verbosity", &verbosity,
                 "0 = quiet; 1 = print status; 2 = print factor matrices; "
                 "3 = print tensor");

  pt::rank_t rank = 10;
  cmdp.setOption("rank", &rank, "number of components; int >= 1");

  int maxIter = 100;
  cmdp.setOption("maxiter", &maxIter, "maximum number of iterations; int >= 1");

  double tol = 1.0e-6;
  cmdp.setOption("tol", &tol, "tolerance for solver");

  int funkyInitKtensor = 0;
  cmdp.setOption("funkyInitKtensor", &funkyInitKtensor, 
       "(0/1):  consistent (across different number of processors) random "
       "initialization of factor matrices; for testing only");

  int minIter = 1;
  cmdp.setOption("miniter", &minIter, "minimum number of iterations; int >= 1");

  std::string diststring;
  cmdp.setOption("dist", &diststring,
                 "processor layout for distribution: IxJxKx...");

  int optimizeMaps = 0;
  cmdp.setOption("optimizeMaps", &optimizeMaps, 
                 "let Tpetra optimize the tensor's maps for import operations");

  // options for randomly generated problem
  size_t randomsize = 0;
  cmdp.setOption("random", &randomsize, 
                 "number of nonzeros to randomly generate");

  std::string modestring;
  cmdp.setOption("modes", &modestring, "mode sizes for the tensor: IxJxKx...");

  // options for reading file a la SPLATT
  std::string filename("NoFiLeNaMe");
  cmdp.setOption("file", &filename, "COO-formatted file containing tensor");

  int rebalance = 0;
  cmdp.setOption("rebalance", &rebalance, "call Zoltan to rebalance nonzeros");

  int oneToOneMaps = 0;
  cmdp.setOption("oneToOneMaps", &oneToOneMaps, 
                 "use KTensor maps that are derived from the SpTensor maps; "
                 "used only with file-based input");


  cmdp.parse(narg, arg);

  typedef double scalar_t;
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
      sptensor = readUsingSplattIO<ptsptensor_t>(filename, diststring, 
                                                 rebalance, comm);
    }
    catch (std::exception &e) {
      std::cout << "FAIL:  Error caught from readUsingSplattIO "
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
  
  // Do CP-ALS
  int numIter = 0;
  double resNorm = 0.;

  if (me == 0) std::cout << "Calling DIST CP_ALS" << std::endl;

  {
    Teuchos::TimeMonitor lcltimer(*timeCpAls);
    distSys->cp_als(tol, minIter, maxIter, numIter, resNorm);
  }

  if (me == 0)
    std::cout << "DIST CP_ALS numIter " << numIter
              << " resNorm " << resNorm << std::endl;

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
