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
 
// Test timing of copy from LayoutLeft to LayoutRight and back again.
// Time appeared to have increased from March to May.

#include <Kokkos_Core.hpp>
#include <Tpetra_Core.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Tpetra_MultiVector.hpp>

int main(int narg, char **arg)
{
  Kokkos::initialize(narg, arg); {
  Teuchos::GlobalMPISession dudette(&narg, &arg);
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int me = comm->getRank();
  int np = comm->getSize();

  Teuchos::RCP<Teuchos::Time> 
    timeUseless(Teuchos::TimeMonitor::getNewTimer("USELESS")), 
    timeLeftToRight(Teuchos::TimeMonitor::getNewTimer("COPY LEFT TO RIGHT")), 
    timeRightToLeft(Teuchos::TimeMonitor::getNewTimer("COPY RIGHT TO LEFT"));

  const int nIters = 100;
  const int nViews = 3;
  const int nJ = 10;

  // Initialize MultiVectors
  if (me == 0) std::cout << "creating multivectors" << std::endl;

  typedef Tpetra::Map<> map_t;
  typedef Tpetra::MultiVector<double> mv_t;
  typedef typename mv_t::node_type::memory_space memspace_t;

  std::vector<mv_t*> mvecs;
  mvecs.reserve(nViews);

  for (int m = 0; m < nViews; m++) {
    int nids = pow(100, (m+1)) * np;
    Teuchos::RCP<const map_t> map = Teuchos::rcp(new map_t(nids, 0, comm));
    mvecs.push_back(new mv_t(map, nJ));
    mvecs[m]->putScalar(1.);
  }

  // Initialize LayoutLeft vectors
  if (me == 0) std::cout << "initializing layoutLeft views" << std::endl;

  typedef Kokkos::View<double **, Kokkos::LayoutLeft> layoutLeft_t;
  std::vector<layoutLeft_t> left;
  left.reserve(nViews);

  for (int m = 0; m < nViews; m++) {
    left.push_back(mvecs[m]->template getLocalView<memspace_t>());
  }

  // Initialize LayoutRight vectors
  if (me == 0) std::cout << "initializing layoutRight views" << std::endl;

  typedef Kokkos::View<double **, Kokkos::LayoutRight> layoutRight_t;
  std::vector<layoutRight_t> right;
  right.reserve(nViews);

  for (int m = 0; m < nViews; m++) {
    right.push_back(layoutRight_t("copym", left[m].extent(0),
                                           left[m].extent(1)));
  }

  // Main loop:  do copy from left to right and back again
  if (me == 0) std::cout << "beginning loop" << std::endl;
  for (int i = 0; i < nIters; i++) {

    // Copy from LayoutLeft to LayoutRight
    timeLeftToRight->start();

    for (int m = 0; m < nViews; m++) {
      Kokkos::deep_copy(right[m], left[m]);
    }

    timeLeftToRight->stop();


    // Useless memory access
    timeUseless->start();

    for (int m = 0; m < nViews; m++) 
      for (size_t jj = 0; jj < right[m].extent(0); jj++) 
        for (size_t kk = 0; kk < right[m].extent(1); kk++)
          right[m](jj,kk) += m;

    timeUseless->stop();


    // Copy from LayoutRight to LayoutLeft
    timeRightToLeft->start();
    
    for (int m = 0; m < nViews; m++)
      Kokkos::deep_copy(left[m], right[m]);
   
    timeRightToLeft->stop();
  }

  // The end
  Teuchos::TimeMonitor::summarize();
  for (int m = 0; m < nViews; m++) delete mvecs[m];

  if (me == 0) std::cout << "PASS" << std::endl;
  } Kokkos::finalize();
  return 0;
}

