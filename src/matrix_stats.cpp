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
 

#ifndef __MATRIX_STATS_HPP
#define __MATRIX_STATS_HPP

#include <iostream>
#include <map>
#include "Epetra_Comm.h"
#ifdef EPETRA_MPI
#include "mpi.h"
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Import.h"
#include "Epetra_Export.h"

using namespace std;

void matrix_stats(Epetra_CrsMatrix *A)
{
  int np = A->Comm().NumProc();
  int me = A->Comm().MyPID();

  const int statArraySize = 17;
  long long maxStat[statArraySize], minStat[statArraySize];
  long long sumStat[statArraySize];
  long long stat[statArraySize];
  for (int i = 0; i < statArraySize; i++) stat[i] = 0;

  // Matrix size stats
  stat[0] = A->NumMyNonzeros();
  stat[1] = A->RowMap().NumMyElements();
  stat[2] = A->ColMap().NumMyElements();

  // Vector size stats
  stat[3] = A->RangeMap().NumMyElements();

  // Importer stats
  if (A->Importer()) {
    // Communication volume
    stat[4] = A->Importer()->NumSend();
    stat[5] = A->Importer()->NumRecv();
    stat[15] = A->Importer()->NumRemoteIDs();

    // Number of messages
    std::map<int,int> exportpidmap;
    int *exportpids = A->Importer()->ExportPIDs();
    int nexportpids = A->Importer()->NumExportIDs();
    for (int i = 0; i < nexportpids; i++) {
      int k = exportpids[i];
      if (exportpidmap.find(k) != exportpidmap.end())
        exportpidmap[k] = exportpidmap[k] + 1;
      else
        exportpidmap[k] = 1;
    }
    stat[6] = exportpidmap.size();

    // Size of largest message in importer
    int maxmsg = 0;
    for (std::map<int,int>::iterator it = exportpidmap.begin();
         it != exportpidmap.end(); it++)
      if (it->second > maxmsg) maxmsg = it->second;
    stat[7] = maxmsg;

    // Some detailed output for low processor counts
    if (np < 26) {
      cout << me << " IMPORTER nSend " << A->Importer()->NumSend()
                 << " nRecv " << A->Importer()->NumRecv()
                 << " nPids " << exportpidmap.size()
                 << endl;
      cout << me << "    IMPORT ";
      for (std::map<int,int>::iterator it = exportpidmap.begin();
           it != exportpidmap.end(); it++)
        cout << "(" << it->first << " " << it->second << ") ";
      cout << endl;
    }
  }

  if (A->Exporter()) {
    // Communication volume
    stat[8] = A->Exporter()->NumSend();
    stat[9] = A->Exporter()->NumRecv();
    stat[16] = A->Exporter()->NumRemoteIDs();

    // Number of messages
    std::map<int,int> exportpidmap;
    int *exportpids = A->Exporter()->ExportPIDs();
    int nexportpids = A->Exporter()->NumExportIDs();
    for (int i = 0; i < nexportpids; i++) {
      int k = exportpids[i];
      if (exportpidmap.find(k) != exportpidmap.end())
        exportpidmap[k] = exportpidmap[k] + 1;
      else
        exportpidmap[k] = 1;
    }
    stat[10] = exportpidmap.size();

    // Size of largest message in exporter
    int maxmsg = 0;
    for (std::map<int,int>::iterator it = exportpidmap.begin();
         it != exportpidmap.end(); it++)
      if (it->second > maxmsg) maxmsg = it->second;
    stat[11] = maxmsg;

    // Some detailed output for low processor counts
    if (np < 26) {
      cout << me << " EXPORTER nSend " << A->Exporter()->NumSend()
                 << " nRecv " << A->Exporter()->NumRecv()
                 << " nPids " << exportpidmap.size()
                 << endl;
      cout << me << "    EXPORT ";
      for (std::map<int,int>::iterator it = exportpidmap.begin();
           it != exportpidmap.end(); it++)
        cout << "(" << it->first << " " << it->second << ") ";
      cout << endl;
    }
  }

  // Total communication (Import + Export)
  stat[12] = stat[4] + stat[8];
  stat[13] = stat[5] + stat[9];
  stat[14] = stat[6] + stat[10];

  A->Comm().MaxAll(stat, maxStat, statArraySize);
  A->Comm().MinAll(stat, minStat, statArraySize);
  A->Comm().SumAll(stat, sumStat, statArraySize);

  if (me == 0) {
    cout << endl << "************************************************" << endl;
    cout << "Matrix Stats:  "                 << endl
         << "  Global number of rows:       " << A->NumGlobalRows64() << endl
         << "  Global number of columns:    " << A->NumGlobalCols64() << endl
         << "  Global number of nonzeros:   " << A->NumGlobalNonzeros64()<< endl
         << "  Rows per proc (min|max|avg|tot): "
               << minStat[1] << " | "
               << maxStat[1] << " | "
               << (float)sumStat[1]/(float)np << " | "
               << sumStat[1]
               << "   imbal=" << maxStat[1] / ((float)sumStat[1]/(float)np)
               << endl
         << "  Cols per proc (min|max|avg|tot): "
               << minStat[2] << " | "
               << maxStat[2] << " | "
               << (float)sumStat[2]/(float)np << " | "
               << sumStat[2]
               << "   imbal=" << maxStat[2] / ((float)sumStat[2]/(float)np)
               << endl
         << "  Nonzeros/proc (min|max|avg|tot): "
               << minStat[0] << " | "
               << maxStat[0] << " | "
               << (float)sumStat[0]/(float)np << " | "
               << sumStat[0]
               << "   imbal=" << maxStat[0] / ((float)sumStat[0]/(float)np)
               << endl
               << endl;

    cout << "Range/Domain Vector Stats: "     << endl
         << "  Global number of entries:    "
               << A->RangeMap().NumGlobalElements64()
               << endl
         << "  Entries/proc (min|max|avg|tot):  "
               << minStat[3] << " | "
               << maxStat[3] << " | "
               << (float)sumStat[3]/(float)np << " | "
               << sumStat[3]
               << "   imbal=" << maxStat[3] / ((float)sumStat[3]/(float)np)
               << endl
               << endl;

    cout << "Importer Stats:  " 
         << (A->Importer() == NULL ? "Importer is NULL" : " ")
         << endl
         << "  Imp NumSend (min|max|avg|tot):  "
               << minStat[4] << " | "
               << maxStat[4] << " | "
               << (float)sumStat[4]/(float)np << " | "
               << sumStat[4]
               << endl
         << "  Imp NumRecv (min|max|avg|tot):  "
               << minStat[5] << " | "
               << maxStat[5] << " | "
               << (float)sumStat[5]/(float)np << " | "
               << sumStat[5]
               << endl
         << "  Imp NumMsgs (min|max|avg|tot):  "
               << minStat[6] << " | "
               << maxStat[6] << " | "
               << (float)sumStat[6]/(float)np << " | "
               << sumStat[6]
               << endl
         << "  Imp NumRemoteIDs (min|max|avg|tot):  "
               << minStat[15] << " | "
               << maxStat[15] << " | "
               << (float)sumStat[15]/(float)np << " | "
               << sumStat[15]
               << endl
         << "  Imp LargestMsgPerRank (min|max):  "
               << minStat[7] << " | "
               << maxStat[7]
               << endl
               << endl;

    cout << "Exporter Stats:  " 
         << (A->Exporter() == NULL ? "Exporter is NULL" : " ")
         << endl
         << "  Exp NumSend (min|max|avg|tot):  "
               << minStat[8] << " | "
               << maxStat[8] << " | "
               << (float)sumStat[8]/(float)np << " | "
               << sumStat[8]
               << endl
         << "  Exp NumRecv (min|max|avg|tot):  "
               << minStat[9] << " | "
               << maxStat[9] << " | "
               << (float)sumStat[9]/(float)np << " | "
               << sumStat[9]
               << endl
         << "  Exp NumMsgs (min|max|avg|tot):  "
               << minStat[10] << " | "
               << maxStat[10] << " | "
               << (float)sumStat[10]/(float)np << " | "
               << sumStat[10]
               << endl
         << "  Exp NumRemoteIDs (min|max|avg|tot):  "
               << minStat[16] << " | "
               << maxStat[16] << " | "
               << (float)sumStat[16]/(float)np << " | "
               << sumStat[16]
               << endl
         << "  Exp LargestMsgPerRank (min|max):  "
               << minStat[11] << " | "
               << maxStat[11]
               << endl
               << endl;

    cout << "Communication Volume (Import + Export):  " << endl
         << "  CV NumSend (min|max|avg|tot):  "
               << minStat[12] << " | "
               << maxStat[12] << " | "
               << (float)sumStat[12]/(float)np << " | "
               << sumStat[12]
               << endl
         << "  CV NumRecv (min|max|avg|tot):  "
               << minStat[13] << " | "
               << maxStat[13] << " | "
               << (float)sumStat[13]/(float)np << " | "
               << sumStat[13]
               << endl
         << "  CV NumMsgs (min|max|avg|tot):  "
               << minStat[14] << " | "
               << maxStat[14] << " | "
               << (float)sumStat[14]/(float)np << " | "
               << sumStat[14]
               << endl
               << endl;

    cout << endl << "************************************************" << endl;
  }
}

#endif
