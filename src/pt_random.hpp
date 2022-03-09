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
 
#ifndef PT_RANDOM_
#define PT_RANDOM_

#include "pt_ktensor.hpp"
#include "pt_shared.h"
#include <array>
#include <vector>
#include <unordered_map>
#include <map>
#include <Kokkos_UnorderedMap.hpp>

namespace pt {

template <typename sptensor_t, typename ktensor_t>
class randomData 
{

public:

  typedef typename sptensor_t::gno_t gno_t;
  typedef typename sptensor_t::scalar_t scalar_t;
  typedef typename sptensor_t::gnoview_t gnoview_t;
  typedef typename sptensor_t::valueview_t spvalueview_t;

  ///////////////////////////////////////////////////////////////////////
  // Constructor doesn't do much.
  randomData(const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
    comm(comm_), me(comm_->getRank()), np(comm_->getSize())
  {}

  ///////////////////////////////////////////////////////////////////////
  // Construct a random distKtensor given array of mode sizes modeSizeView.
  // Uses default Tpetra::Map for each factor matrix.
  ktensor_t *createRandomKtensor(
    rank_t rank, 
    const std::vector<size_t> &modeSizes)
  {
    srand(comm->getRank()+12345);
    ktensor_t *ktensor = new ktensor_t(rank, modeSizes, comm);
    ktensor->setRandomUniform();
    return ktensor;
  }
                      
  ///////////////////////////////////////////////////////////////////////
  // Construct a random distKtensor given distributed sptensor
  // Derive ktensor maps from sptensor maps
  ktensor_t *createRandomKtensor(
    rank_t rank, 
    sptensor_t *sptensor)
  {
    srand(comm->getRank()+12345);
    ktensor_t *ktensor = new ktensor_t(rank, sptensor, comm);
    ktensor->setRandomUniform();
    return ktensor;
  }
                      
  ///////////////////////////////////////////////////////////////////////
  // Construct a random distSptensor given a distKtensor.
  // The distSptensor is distributed block-wise, with gid range in
  // each mode dimension aligned with the gid range of associated 
  // factormatrix .
  sptensor_t *createRandomSptensor(
    const ktensor_t *ktensor,             // input ktensor with maps
    size_t targetNnz,                     // input global max number of nonzeros
    const std::vector<int> &procsPerMode  // input # of procs per mode 
  )
  {
    mode_t nModes = ktensor->getNumModes();

    // Error check proc distribution with number of processors

    int checknp = 1;
    for (mode_t m = 0; m < nModes; m++)
      checknp *= procsPerMode[m];
    if (checknp != np)
      throw std::runtime_error("invalid procsPerMode; product not equal to np");

    // Min/max gids in each mode for this rank, assuming Trilinos default maps
    // used in ktensor.  Want these entries to align with clean subset of 
    // proc entries from ktensor, so that ktensor communicates along row of 
    // procs only.
  
    std::vector<size_t> myNumGids(nModes);
    std::vector<gno_t> myMinGid(nModes);
  
    getMyGidRange(myNumGids, myMinGid, procsPerMode, ktensor);

    // Generate nonzeros on this processor; store them in Kokkos data structures
    
    gnoview_t myGids;
    spvalueview_t myVals;

    size_t maxModeSize = 0;
    for (mode_t m = 0; m < nModes; m++)
      if (ktensor->getModeSize(m) > maxModeSize) 
        maxModeSize = ktensor->getModeSize(m);

    if (maxModeSize < std::numeric_limits<char>::max()) {
      if (nModes <= mode_t(sizeof(uint64_t) / sizeof(char)))
        // Can use the Kokkos::UnorderedMap; allows BIGGER test problems to be
        // run, and is a bit faster, too.
        getMyNonzerosUsingKokkosUMap<char, sizeof(uint64_t)/sizeof(char)> (
                                     targetNnz, nModes, myNumGids, myMinGid,
                                     myGids, myVals);
      else
        getMyNonzeros<char>(targetNnz, nModes, myNumGids, myMinGid,
                            myGids, myVals);
    }
    else if (maxModeSize < std::numeric_limits<short>::max()) {
      if (nModes <= mode_t(sizeof(uint64_t) / sizeof(short)))
        // Can use the Kokkos::UnorderedMap; allows BIGGER test problems to be
        // run, and is a bit faster, too.
        getMyNonzerosUsingKokkosUMap<short, sizeof(uint64_t)/sizeof(short)> (
                                    targetNnz, nModes, myNumGids, myMinGid,
                                    myGids, myVals);
      else
        getMyNonzeros<short>(targetNnz, nModes, myNumGids, myMinGid,
                             myGids, myVals);
    }
    else if (maxModeSize < std::numeric_limits<int>::max())
      getMyNonzeros<int>(targetNnz, nModes, myNumGids, myMinGid, 
                         myGids, myVals);
    else if (maxModeSize < std::numeric_limits<long long>::max())
      getMyNonzeros<long long>(targetNnz, nModes, myNumGids, myMinGid, 
                               myGids, myVals);
    else 
      throw std::runtime_error("you gotta be kidding!");
   
    comm->barrier();
    if (me == 0) std::cout << "Random:  Sptensor constructor" << std::endl;
    return (new sptensor_t(nModes, ktensor->getModeSizes(),
                           myGids, myVals, comm, myMinGid, myNumGids));
  }

private:

  const Teuchos::RCP<const Teuchos::Comm<int> > comm;
  const int me;
  const int np;

  static const mode_t MAX_NMODES_FOR_KOKKOS_UNORDERED_MAP = 4;

  /////////////////////////////////
  void getMyGidRange(
    std::vector<size_t> &myNumGids,
    std::vector<gno_t> &myMinGid,
    const std::vector<int> &procsPerMode,
    const ktensor_t *ktensor
  )
  {
    if (me == 0) std::cout << "Random:  Computing ranges" << std::endl;

    mode_t nModes = ktensor->getNumModes();
    std::vector<gno_t> minFMGidPerProc(np+1);
    
    int prod = 1;
    for (mode_t m = 0; m < nModes; m++) {

      typedef typename ktensor_t::factormatrix_t::map_t map_t;
      const map_t *fmmap = ktensor->getFactorMatrix(m)->getMap();

      gno_t minFMGid = fmmap->getMinGlobalIndex();
      Teuchos::gatherAll<int,gno_t>(*comm, 1, &minFMGid,
                                           np, &(minFMGidPerProc[0]));
      minFMGidPerProc[np] = fmmap->getGlobalNumElements();

      // Make sure enough processors have factor matrix entries
      // If modeSize[m] is too small, not all processors will have
      // factor matrix entries.  Need at least procsPerMode[m]
      // processors to have entries.  With default Trilinos Maps,
      // the entries will be in the low-numbered processors.
      // Then divide the npNotEmptys evenly among the procsPerMode[m].
      int npNotEmpty;
      for (npNotEmpty = 0; npNotEmpty < np; npNotEmpty++)
        if (minFMGidPerProc[npNotEmpty+1]-minFMGidPerProc[npNotEmpty] == 0)
          break;

      if (procsPerMode[m] > npNotEmpty) {
        if (me == 0)
          std::cout << "Error:  Number of procs " << procsPerMode[m]
                    << " assigned to mode " << m
                    << " is greater than number of nonempty procs "
                    << npNotEmpty
                    << " in factor matrix " << m << "."
                    << std::endl
                    << "In distribution, use fewer procs in mode " << m
                    << std::endl;
        throw std::runtime_error("Error in getMyGidRange");
      }

      int myProcM = (me / prod) % procsPerMode[m];
      prod *= procsPerMode[m];

      int npDivPPM = npNotEmpty / procsPerMode[m];
      int npModPPM = npNotEmpty % procsPerMode[m];
      int beginFMProc = myProcM     * npDivPPM 
                      + (myProcM     < npModPPM ? myProcM     : npModPPM);
      int endFMProc   = (myProcM+1) * npDivPPM 
                      + ((myProcM+1) < npModPPM ? (myProcM+1) : npModPPM);
  
      myNumGids[m] = minFMGidPerProc[endFMProc] - minFMGidPerProc[beginFMProc];
      myMinGid[m] = minFMGidPerProc[beginFMProc];

//      std::cout << me 
//                << " PROCDIST mode " << m 
//                << " npNotEmpty " << npNotEmpty
//                << " myProcM " << myProcM << ": " 
//                << " myNumGids " << myNumGids[m] 
//                << " myMinGid " << myMinGid[m]
//                << " beginFMProc " << beginFMProc 
//                << " endFMProc " << endFMProc
//                << std::endl;
    }
  }

  /////////////////////////////////
  //  Function for generating tensor nonzeros using std::unordered_map
  //  or std::map to remove duplicate nonzeros

  template <typename T>  // the smallest int needed to hold the max mode size
  void getMyNonzeros(
    size_t targetNnz,
    int nModes,
    std::vector<size_t> &myNumGids,
    std::vector<gno_t> &myMinGid,
    gnoview_t &myGids,
    spvalueview_t &myVals
  )
  {
    if (me == 0) std::cout << "Random:  Generating nonzeros of size " 
                           << sizeof(T) << "\n"
#define SORTED  // This option uses more memory but is faster than our 
                // crappy hash function for std::unordered_map
#ifdef SORTED
                           << "!! Using std::map !!"
#else
                           << "!! Using std::unordered_map !!"
#endif
                           << std::endl;
  
    size_t myTargetNnz = targetNnz / np;  // num of nonzeros this proc creates
    srand(me);

    std::vector<T> newGid(nModes);    // a single nonzero's coordinates
  
    // Build a map to uniquify the generated nonzeros

#ifdef SORTED  // use std::map
    struct myCompare {
      bool operator()(std::vector<T> const &a, std::vector<T> const &b) const
      {
        std::size_t len = a.size();
        for(std::size_t i = 0; i < len; i++) {
          if (a[i] < b[i]) return true;
          if (a[i] > b[i]) return false;
        }
        return false;
      }
    };

    typedef typename std::map<std::vector<T>, scalar_t, myCompare> gidmap_t;
    gidmap_t generatedGids;

#else // !SORTED:  use std::unordered_map
    struct myHash {
      std::size_t operator()(std::vector<T> const& vec) const {
        // TODO:  This hash function stinks and is dog-slow!  TODO
        std::size_t len = vec.size();
        std::size_t seed = len;
        for(std::size_t i = 0; i < len; i++)
          seed ^= std::hash<T>{}(vec[i]);
        std::cout << "KDDHASH " << seed << std::endl;
        return seed;
      }
    };

    typedef std::unordered_map<std::vector<T>, scalar_t, myHash<T> > gidmap_t;
    gidmap_t generatedGids(myTargetNnz);

#endif

    // Create nonzeros; add them to the map.
    // set nonzeros' value to random data in [1,11)
    int printcnt = 0;
    for (size_t i = 0; i < myTargetNnz; i++) {
      for (mode_t m = 0; m < nModes; m++) 
        newGid[m] = myMinGid[m] + (rand() % myNumGids[m]);
      scalar_t val = scalar_t((rand() % 10) + 1);
      generatedGids[newGid] = val;
      if (printcnt == 10000000) {
        if (me == 0) 
          std::cout << me << "    " << i << std::endl;
        printcnt = 0;
      }
      printcnt++;
    }

    // Create return arguments in format needed for sptensor
    comm->barrier();
    if (me == 0) std::cout << "Random:  Allocating Kokkos::View" << std::endl;

    size_t myNnz = generatedGids.size();
    myGids = gnoview_t("myRandomGids", myNnz, nModes);
    myVals = spvalueview_t("myRandomVals", myNnz);
  
    // Copy map values into return arguments
    comm->barrier();
    if (me == 0) std::cout << "Random:  Copying to Kokkos::View" << std::endl;

    size_t cnt = 0;
    for (auto it = generatedGids.begin(); it != generatedGids.end(); it++) {
      for (mode_t m = 0; m < nModes; m++) {
        myGids(cnt,m) = it->first[m];
      }
      myVals(cnt) = it->second;
      cnt++;
    }
  }

  /////////////////////////////////
  //  Function for generating tensor nonzeros using Kokkos::UnorderedMap 
  //  to remove duplicate nonzeros
  //  Packs the generated IDs into a uint64_t that serves as the key in
  //  Kokkos::UnorderedMap.
  //  Thus, this function cannot be used as-is for all problem ranges.
  template <typename T, int MAX_NMODES>  
  void getMyNonzerosUsingKokkosUMap(
    size_t targetNnz,
    int nModes,
    std::vector<size_t> &myNumGids,
    std::vector<gno_t> &myMinGid,
    gnoview_t &myGids,
    spvalueview_t &myVals
  )
  {
    if (me == 0) std::cout << "Random:  Generating nonzeros of size " 
                           << sizeof(T) << std::endl
                           << "!! Using Kokkos::UnorderedMap !!"
                           << std::endl;

    // Error checking -- are we able to pack an index into uint64_t?
    if (nModes > MAX_NMODES) {
      if (me == 0) 
        std::cout << "Error:  nModes " << nModes << " greater than MAX_NMODES "
                  << MAX_NMODES << " in getMyNonzerosUsingKokkosUMap; "
                  << " use getMyNonzeros instead."
                  << std::endl;
      throw std::runtime_error("Error in getMyNonzerosUsingKokkosUMap");
    }

    if (nModes * sizeof(T) > sizeof(uint64_t)) {
      if (me == 0) 
        std::cout << "Error:  nModes * sizeof(T) " << nModes * sizeof(T)
                  << " greater than sizeof(uint64_t) " << sizeof(uint64_t)
                  << " in getMyNonzerosUsingKokkosUMap; "
                  << " use getMyNonzeros instead."
                  << std::endl;
      throw std::runtime_error("Error in getMyNonzerosUsingKokkosUMap");
    }

    // All good.  Proceed to generate keys.

    typedef std::array<T, MAX_NMODES> key_t;
  
    size_t myTargetNnz = targetNnz / np;  // num of nonzeros this proc creates
    srand(me);

    // a single nonzero's coordinates
    key_t newGid;    
  
    // Build a map to uniquify the generated nonzeros;
    // set nonzeros' value to random data in [1,11)
 
    typedef typename Kokkos::UnorderedMap<uint64_t, scalar_t, 
                                          Kokkos::DefaultExecutionSpace>
                                          gidmap_t;
    gidmap_t generatedGids(myTargetNnz);

    // Create nonzeros; add them to the map 
    int printcnt = 0; 
    size_t failcnt = 0;
    size_t successcnt = 0;
    size_t copysize = nModes * sizeof(T);

    while (successcnt < myTargetNnz) {
      for (mode_t m = 0; m < nModes; m++) 
        newGid[m] = myMinGid[m] + (rand() % myNumGids[m]);
      scalar_t val = scalar_t((rand() % 10) + 1);
   
      uint64_t tmp = 0;
      memcpy(&tmp, &(newGid[0]), copysize);

      auto it = generatedGids.insert(tmp, val);
//    std::cout << me << " "
//              << " KDD " << myHash<key_t>{}(newGid)
//              << " [" << newGid[0] << " " << newGid[1] << " "
//              << newGid[2] << " " << newGid[3] << "] " 
//              << " F" << it.failed() << " S" << it.success() 
//              << " E" << it.existing() << " "
//              << it.list_position() << " " << it.index() << " "
//              << std::endl;
      if (it.failed()) {
        failcnt++;
      }
      else 
        successcnt++;

      printcnt++;
      if (me == 0 && printcnt == 10000000) {
        std::cout << me << "    " << successcnt << " " << failcnt << std::endl;
        printcnt = 0;
      }
    }

//  std::cout << me << " Generation complete:  " 
//            << successcnt << " successes "
//            << failcnt << " failures " 
//            << std::endl;

    // Create return arguments in format needed for sptensor
    comm->barrier();
    if (me == 0) std::cout << "Random:  Allocating Kokkos::View" << std::endl;

    size_t cap = generatedGids.capacity();

#undef SORT_KOKKOS_UMAP  // TODO:  the sort really should be an option for the 
                         // tensor constructor, not the random generator.
                         // I'll keep the syntax here, as it will likely be 
                         // useful later in the constructor.
#ifdef SORT_KOKKOS_UMAP

    // Sorts by last mode, then by next-to-last, etc.
    // Due to endian of the keys -- FUNNY!
    struct IdxCompare
    {
      const gidmap_t &keys;
      IdxCompare(const gidmap_t &keys_): keys(keys_) {}
      bool operator()(int a, int b) const { 
        if (keys.valid_at(a) && keys.valid_at(b))
          return (keys.key_at(a) < keys.key_at(b));
        if (keys.valid_at(a))
          return true;
        return false;
      }
    };

    struct IdxCompare comparator(generatedGids);
    std::vector<size_t> perm(cap);
    for (size_t i = 0; i < cap; i++) perm[i] = i;
    std::sort(perm.begin(), perm.end(), comparator);

#endif

    size_t myNnz = generatedGids.size();
    myGids = gnoview_t("myRandomGids", myNnz, nModes);
    myVals = spvalueview_t("myRandomVals", myNnz);
  
    // Copy map values into return arguments
    comm->barrier();
    if (me == 0) std::cout << "Random:  Copying to Kokkos::View" << std::endl;

    size_t cnt = 0;
    for (size_t it = 0; it < cap; it++) {
#ifdef SORT_KOKKOS_UMAP
      size_t idx = perm[it];
#else
      size_t idx = it;
#endif
      if (!generatedGids.valid_at(idx)) continue;
      uint64_t key = generatedGids.key_at(idx);
      key_t tmp;
      memcpy(&(tmp[0]), &key, copysize);
      for (mode_t m = 0; m < nModes; m++) {
        myGids(cnt,m) = tmp[m];
      }
      myVals(cnt) = generatedGids.value_at(idx);
      cnt++;
    }
  }

//  The syntax for these Kokkos::UnorderedMap functions is correct.
//  However, they aren't really needed when we pack indices into a uint64_t
//  and use that as the Kokkos::UnorderedMap key.  
//  I'll keep the code in the file for now so that if I try to use custom
//  functions at a later date, I don't have to figure out the syntax again.
//
//  template <typename T>
//  struct myEqualKokkos {
//    bool operator()(T const &a, T const &b) const
//    {
//      std::size_t len = a.size();
//      for(std::size_t i = 0; i < len; i++) {
//        if (a[i] < b[i]) return false;
//        if (a[i] > b[i]) return false;
//      }
//      return true;
//    }
//  };
//
//  template <typename T>
//  struct myHashKokkos
//  {
//    KOKKOS_FORCEINLINE_FUNCTION
//    uint32_t operator()(T const& vec) const {
//      // This hash function is a hack for now
//      int sz = vec.size();
//      int szof = sizeof(vec[0]);
//      if ((sz > 4) || (szof != sizeof(short))) {
//        std::cout << "myHash not ready" << std::endl;
//        MPI_Abort(MPI_COMM_WORLD, -1);
//      }
//      uint64_t tmp = 0;
//      memcpy(&tmp, &vec[0], sz*szof);
//      std::size_t ret = std::hash<uint64_t>{}(tmp);
//
//      //std::cout << "MY HASH "<< ret << " [" << vec[0]; 
//      //if (sz > 1) std::cout << " " << vec[1];
//      //if (sz > 2) std::cout << " " << vec[2];
//      //if (sz > 3) std::cout << " " << vec[3];
//      //std::cout << "] " << std::endl;
//
//      return ret;
//    }
//  };

};

}  // namespace pt

#endif // PT_RANDOM
