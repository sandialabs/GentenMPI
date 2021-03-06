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
 
#include <iostream>
#include "Tpetra_Core.hpp"
#include "Tpetra_MultiVector.hpp"

int main(int narg, char** arg)
{
  Tpetra::ScopeGuard scopeguard(&narg, &arg);
  auto comm = Tpetra::getDefaultComm();

  typedef Tpetra::Map<> map_t;
  typedef Tpetra::MultiVector<double> mvector_t;

  const int len = 2000000000;

  if (comm->getRank() == 0) 
    std::cout << "Creating map with global len " << len << std::endl;

  Teuchos::RCP<map_t> map = Teuchos::RCP<map_t>(new map_t(len, 0, comm));

  for (int i = 2; i <= 10; i+=2) {

    if (comm->getRank() == 0) 
      std::cout << "Creating multivec with " << i << " vecs and global len "
                << len << std::endl;
    mvector_t *mv = new mvector_t(map, i);

    if (comm->getRank() == 0) 
      std::cout << "   Success " << mv->getGlobalLength() << " "
                                 << mv->getNumVectors() << std::endl;

    //if (comm->getRank() == 0) 
    //  std::cout << "Initializing multivec " << std::endl;
  
    //mv.putScalar(i);

    if (comm->getRank() == 0) 
      std::cout << "All good" << std::endl;
  }

  return 0;
}
