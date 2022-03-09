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
// Copies 2D arrays ("multivectors") of varying lengths.
// Runtime increased from March to May.

#include "Kokkos_Core.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_TimeMonitor.hpp"

int main(int narg, char **arg)
{
  Kokkos::initialize(narg, arg); {
  Teuchos::GlobalMPISession session(&narg, &arg);
  Teuchos::RCP<const Teuchos::Comm<int> > comm = 
           Teuchos::DefaultComm<int>::getComm();
  int me = comm->getRank();

  Teuchos::RCP<Teuchos::Time> 
    timeLeftToRight(Teuchos::TimeMonitor::getNewTimer("COPY LEFT TO RIGHT")), 
    timeRightToLeft(Teuchos::TimeMonitor::getNewTimer("COPY RIGHT TO LEFT"));

  const int nIters = 100;
  const int nViews = 3;
  const int nJ = 10;
  
  typedef Kokkos::View<double **, Kokkos::LayoutLeft> layoutLeft_t;
  typedef Kokkos::View<double **, Kokkos::LayoutRight> layoutRight_t;

  // Initialize LayoutLeft vectors
  if (me == 0) std::cout << "Initializing LayoutLeft multivectors" << std::endl;

  std::vector<layoutLeft_t> left;
  left.reserve(nViews);

  for (int m = 0; m < nViews; m++) {
    char name[10];
    int nids = pow(100, (m+1));
    sprintf(name, "left%d", m);
    left.push_back(layoutLeft_t(name, nids, nJ, 1.));
    if (me == 0) std::cout << name << " is " << nids << "x" << nJ << std::endl;
  }

  // Initialize LayoutRight vectors
  if (me == 0) std::cout << "Allocating LayoutRight multivectors" << std::endl;

  std::vector<layoutRight_t> right;
  right.reserve(nViews);

  for (int m = 0; m < nViews; m++) {
    right.push_back(layoutRight_t("copym", left[m].extent(0),
                                           left[m].extent(1)));
  }

  if (me == 0) std::cout << "Copying multivectors" << std::endl;

  for (int i = 0; i < nIters; i++) {

    // Copy from LayoutLeft to LayoutRight
    timeLeftToRight->start();

    for (int m = 0; m < nViews; m++) {
      Kokkos::deep_copy(right[m], left[m]);
    }

    timeLeftToRight->stop();


    // Copy from LayoutRight to LayoutLeft
    timeRightToLeft->start();
    
    for (int m = 0; m < nViews; m++)
      Kokkos::deep_copy(left[m], right[m]);
   
    timeRightToLeft->stop();
  }

  Teuchos::TimeMonitor::summarize();

  if (me == 0) std::cout << "PASS" << std::endl;
  } Kokkos::finalize();
  return 0;
}

