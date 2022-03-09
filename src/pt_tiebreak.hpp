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
 
#ifndef PT_TIEBREAK_
#define PT_TIEBREAK_

#include <iostream>
#include <functional>
#include <vector>
#include <cstdlib>
#include <numeric>
#include <utility>
#include <vector>
#include "Tpetra_Map.hpp"

namespace pt {

// Class to enable breaking ties in creation of oneToOneMaps that 
// correspond to the sptensor's maps.
// Use a hash function to select assigned PID; should pick equally
// among PIDs.
template <typename LocalOrdinal, typename GlobalOrdinal>
class HashTieBreak : 
      public Tpetra::Details::TieBreak<LocalOrdinal,GlobalOrdinal> {
private:
  typedef std::pair<int, LocalOrdinal> PidLidPair_t;

  std::hash<GlobalOrdinal> computeHash;

  std::vector<size_t> sortPids(const std::vector<PidLidPair_t> &v) const {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // Assuming not many entries in v, so sort shouldn't be too expensive
    std::sort(idx.begin(), idx.end(),
             [&v](size_t i1, size_t i2) {return v[i1].first < v[i2].first;});

    return idx;
  }

public:
  bool mayHaveSideEffects () const { return false; }

  ~HashTieBreak () {}

  std::size_t selectedIndex (
    GlobalOrdinal GID,
    const std::vector<PidLidPair_t> &pid_and_lid) const 
  {
    // check if no tie to break
    if (pid_and_lid.size() == 1) return 0;

    // Sort index of pid_and_lid by pid; 
    // Can't assume pid_and_lid is in same order on all processors, so
    // need sort to allow all processors to choose the same pid_and_lid entry.
    // Assuming pid_and_lid has few entries so sorting is not too expensive
    std::size_t nPairs = pid_and_lid.size();
    std::vector<size_t> sortedPids = sortPids(pid_and_lid);

    // Compute a hash of the GID
    // Use hash to decide which sorted entry to return
    std::size_t h1 = computeHash(GID);
    size_t idx = h1 % nPairs;
    return (sortedPids[idx]);
  }
};

////////////////////////////////////////////////////////////////////////
// Class to enable breaking ties in creation of oneToOneMaps that 
// correspond to the sptensor's maps.
template <typename LocalOrdinal, typename GlobalOrdinal>
class LoadedTieBreak : 
      public Tpetra::Details::TieBreak<LocalOrdinal,GlobalOrdinal> {
private:
  typedef std::pair<int, LocalOrdinal> PidLidPair_t;

  std::hash<GlobalOrdinal> computeHash;

  std::vector<size_t> sortPids(const std::vector<PidLidPair_t> &v) const {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // Assuming not many entries in v, so sort shouldn't be too expensive
    std::sort(idx.begin(), idx.end(),
             [&v](size_t i1, size_t i2) {return v[i1].first < v[i2].first;});

    return idx;
  }

  size_t *gidsPerPid;

public:
  bool mayHaveSideEffects () const { return false; }

  LoadedTieBreak(const size_t n, const Teuchos::Comm<int> &comm) 
  {
    int np = comm.getSize();
    gidsPerPid = new size_t[np];
    Teuchos::gatherAll<int, size_t>(comm, 1, &n, np, gidsPerPid);
  }

  ~LoadedTieBreak () { delete [] gidsPerPid;}

  std::size_t selectedIndex (
    GlobalOrdinal GID,
    const std::vector<PidLidPair_t> &pid_and_lid) const 
  {
    // check if no tie to break
    if (pid_and_lid.size() == 1) return 0;

    // Sort index of pid_and_lid by pid; 
    // Can't assume pid_and_lid is in same order on all processors, so
    // need sort to allow all processors to choose the same pid_and_lid entry.
    // Assuming pid_and_lid has few entries so sorting is not too expensive
    std::size_t nPairs = pid_and_lid.size();
    std::vector<size_t> sortedPids = sortPids(pid_and_lid);

    // Pick the PID with the lowest count at the beginning of createOneToOne
    size_t minidx = 0;
    size_t min = gidsPerPid[pid_and_lid[sortedPids[0]].first];
std::cout << GID << " nPairs " << nPairs << ": ";
    for (size_t i = 1; i < nPairs; i++) {
      if (gidsPerPid[pid_and_lid[sortedPids[i]].first] < min) {
        min = gidsPerPid[pid_and_lid[sortedPids[i]].first];
        minidx = i;
      }
std::cout << pid_and_lid[sortedPids[i]].first << " ";
    }
std::cout << "; return " << sortedPids[minidx] << " " 
          << pid_and_lid[sortedPids[minidx]].first << std::endl;
    return (sortedPids[minidx]);
  }
};
}
#endif
