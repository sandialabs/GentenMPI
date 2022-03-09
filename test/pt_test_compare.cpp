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
 

#include "pt_test_compare.hpp"

int main()
{
  int ierr = 0;

  double a = 0.;
  double b = 1e-10;
  
  if (pt::nearlyEqual<double>(a, b)) {
    std::cout << "Error:  " << a << " does not equal " << b << std::endl;
    ierr++;
  }

  if (!pt::nearlyEqual<double>(a, b, 1e-6)) {
    std::cout << "Error:  " << a << " is close enough to " << b 
              << "(1e-6)" << std::endl;
    ierr++;
  }

  if (!pt::nearlyEqual<double>(a, b, 1e-8)) {
    std::cout << "Error:  " << a << " is close enough to " << b 
              << "(1e-8)" << std::endl;
    ierr++;
  }

  if (!pt::nearlyEqual<double>(a, b, 1e-10)) {
    std::cout << "Error:  " << a << " is close enough to " << b 
              << "(1e-10)" << std::endl;
    ierr++;
  }

  if (pt::nearlyEqual<double>(a, b, 1e-11)) {
    std::cout << "Error:  " << a << " is not close enough to " << b 
              << "(1e-11)" << std::endl;
    ierr++;
  }


  double c = 123.123456789;
  double d = 123.123456788;
  
  if (pt::nearlyEqual<double>(c, d)) {
    std::cout << "Error:  " << c << " does not equal " << d << std::endl;
    ierr++;
  }

  if (!pt::nearlyEqual<double>(c, d, 1e-8)) {
    std::cout << "Error:  " << c << " is close enough to " << d << std::endl;
    ierr++;
  }

  float e = 0.;
  float f = 0.;

  if (!pt::nearlyEqual<float>(e, f)) {
    std::cout << "Error:  " << e << " equal " << f << std::endl;
    ierr++;
  }

  double g = 12345.6789;
  double h = -12345.6789;

  if (pt::nearlyEqual<double>(g, h)) {
    std::cout << "Error:  " << g << " does not equal " << h << std::endl;
    ierr++;
  }

  double i = 0.0000001;
  double j = -0.0000001;

  if (pt::nearlyEqual<double>(i, j)) {
    std::cout << "Error:  " << i << " does not equal " << j << std::endl;
    ierr++;
  }

  if (!pt::nearlyEqual<double>(i, j, 1e-6)) {
    std::cout << "Error:  " << i << " is close enough to " << j 
              << "(1e-6)" << std::endl;
    ierr++;
  }

  std::cout << (ierr == 0 ? "PASS" : "FAIL") << std::endl;

  return ierr;
}
