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
 
// Unit test for distSystem class
// Does not test mttkrp; see pt_test_mttkrp.cpp for mttkrp tests.

#include "pt_system.hpp"
#include "Tpetra_Core.hpp"

template <typename scalar_t>
class testDistSys {

public:

  typedef typename pt::distFactorMatrix<scalar_t> factormatrix_t;
  typedef typename pt::distKtensor<factormatrix_t> ktensor_t;
  typedef typename pt::distSptensor<scalar_t> sptensor_t;
  typedef typename pt::distSystem<sptensor_t, ktensor_t> distsys_t;

  typedef typename factormatrix_t::map_t map_t;
  typedef typename factormatrix_t::valueview_t valueview_t;
  typedef typename factormatrix_t::gno_t gno_t;

  // Constructor:  initializes values
  testDistSys(const Teuchos::RCP<const Teuchos::Comm<int> > &comm_,
              std::vector<size_t> &modeSizes_,
              pt::rank_t rank_,
              typename sptensor_t::gnoview_t &globalIndices_,
              typename sptensor_t::valueview_t &values_
  ):
    comm(comm_),
    me(comm->getRank()),
    np(comm->getSize()),
    modeSizes(modeSizes_),
    nModes(modeSizes_.size()),
    rank(rank_),
    globalIndices(globalIndices_),
    values(values_)
  { }

  // Destructor
  ~testDistSys() { }

  // How to run the tests within testDistSys
  int run();


private:
  const Teuchos::RCP<const Teuchos::Comm<int> > comm;
  const int me;
  const int np;

  std::vector<size_t> modeSizes;
  const pt::mode_t nModes;
  const pt::rank_t rank;

  typename sptensor_t::gnoview_t globalIndices;
  typename sptensor_t::valueview_t values;

  // A method for exercising import / export as would be needed in mttkrp
  int checkImportExport(const std::string &msg, sptensor_t *sptensor,
                        ktensor_t *ktensor, bool optimizeMaps);

  // Default values in factor matrices
  inline scalar_t testVal(gno_t gid, pt::rank_t r) { return scalar_t(gid*r+1); }

  // Build a cyclic map (like dealing cards)
  map_t *buildCyclicMap(size_t nGlobal)
  {
    // Compute number of local entries
    size_t nMyGids = nGlobal / comm->getSize();
    nMyGids += (size_t(me) < (nGlobal % comm->getSize()) ? 1 : 0);

    // Build list of local entries
    Kokkos::View<gno_t *> myGids("local GIDs", nMyGids);
    gno_t myFirstGid = me;
    for (size_t i = 0; i < nMyGids; i++)
      myGids(i) = myFirstGid + (i * np);

    // Return the cyclic map
    const Tpetra::global_size_t inv =
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    return new map_t(inv, myGids, 0, comm);
  }
}; 


////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
int testDistSys<scalar_t>::checkImportExport(
  const std::string &msg,
  sptensor_t *sptensor,
  ktensor_t *ktensor,
  bool optimizeMaps
)
{
  int ierr = 0;

  std::cout << "Starting " <<  msg << std::endl;
  distsys_t distsys(sptensor, ktensor, distsys_t::UPDATE_ALL, optimizeMaps);

  for (pt::mode_t m = 0; m < nModes; m++) {
    const map_t *sptensorMap = sptensor->getMap(m);

    if ((distsys.getImporter(m) == NULL) && 
        !(sptensorMap->isSameAs(*(distsys.getDomainMap(m))))) {
      // Error:  should have a non-null importer
      ierr++;
      std::cout << me << " " << msg 
                << " Error:  null importer for differing maps" << std::endl;
    }

    // Create input factor matrix using domain map; initialize its values
    factormatrix_t *fin = new factormatrix_t(rank, distsys.getDomainMap(m));

    valueview_t findata = fin->getLocalView();
    size_t len = fin->getLocalLength();
    for (size_t i = 0; i < len; i++) {
      for (pt::rank_t r = 0; r < rank; r++) {
        findata(i, r) = 
               testVal(distsys.getDomainMap(m)->getGlobalElement(i), r);
      }
    }

    if (fin->getGlobalLength() < 50) fin->print("fin");

    // Create intermediate factor matrix using sptensor's maps
    factormatrix_t *fmid = NULL;

    if (distsys.getImporter(m) != NULL) {

      // Perform expand-like communication to fill fmid

      fmid = new factormatrix_t(rank, sptensorMap);
      fmid->doImport(fin, distsys.getImporter(m), Tpetra::INSERT);

      if (fmid->getGlobalLength() < 50) fmid->print("fmid");

      // Check values in fmid
      valueview_t fmiddata = fmid->getLocalView();
      for (size_t i = 0; i < fmid->getLocalLength(); i++) {
        for (pt::rank_t r = 0; r < rank; r++) {
          if (fmiddata(i, r) != 
              testVal(sptensor->getMap(m)->getGlobalElement(i), r)) {
            ierr++;
            std::cout << me << " " << msg << " ImportError " << i << " " << r 
                      << " gid " << sptensor->getMap(m)->getGlobalElement(i)
                      << ":  " << fmiddata(i, r) << " != "
                      << testVal(sptensor->getMap(m)->getGlobalElement(i), r)
                      << std::endl;
          }
        }
      }
    }
    else {
      // No communication or testing needed
      fmid = fin;
    }

    // Test export
    if (distsys.getImporter(m) != NULL) {

      // Create output factor matrix using range map; initialize to dummy value
      const scalar_t dummyval = -1111;
      factormatrix_t *fout = new factormatrix_t(rank, distsys.getRangeMap(m));
      fout->setValues(dummyval);

      // Perform fold-like communication
      fout->doExport(fmid, distsys.getImporter(m), Tpetra::INSERT);

      if (fout->getGlobalLength() < 50) fout->print("fout");

      // Check values in fout
      valueview_t foutdata = fout->getLocalView();
      for (size_t i = 0; i < fout->getLocalLength(); i++)
        for (pt::rank_t r = 0; r < rank; r++)
          if ((foutdata(i, r) != dummyval) &&  // entry was updated by export
              (foutdata(i, r) !=               // but has the wrong value
                        testVal(fout->getMap()->getGlobalElement(i), r))) {
            ierr++;
            std::cout << me << " " << msg << " ExportError " << i << " " << r 
                      << " gid " << fout->getMap()->getGlobalElement(i)
                      << ":  " << foutdata(i, r) << " != "
                      << testVal(fout->getMap()->getGlobalElement(i), r)
                      << std::endl;
          }

      // Check that the correct number of values was exported
      // Export does not accrue local values during the copy, so we have
      // to initialize the local values in fout correctly.
      fmid->setValues(1.);
      fout->setValues(0.);
      for (size_t i = 0; i < fout->getLocalLength(); i++) {
        gno_t gid = fout->getMap()->getGlobalElement(i);
        if (sptensor->getMap(m)->getLocalElement(gid) >= 0) {
          // gid is in both fout and fmid on this processor
          foutdata(i, 0) = 1;
        }
      }

      fout->doExport(fmid, distsys.getImporter(m), Tpetra::ADD);
      if (fout->getGlobalLength() < 50) fout->print("fout count");
      
      size_t nOccurrences = 0, gnOccurrences;
      for (size_t i = 0; i < fout->getLocalLength(); i++) 
        nOccurrences += foutdata(i, 0);

      Teuchos::reduceAll<int,size_t>(*comm, Teuchos::REDUCE_SUM, 1, 
                                     &nOccurrences, &gnOccurrences);

      if (gnOccurrences != fmid->getMap()->getGlobalNumElements()) {
        ierr++;
        std::cout << me << " " << msg 
                  << " Error: Incorrect occurrence count after export " << m 
                  << ":  " << nOccurrences << " != "
                  << fmid->getMap()->getGlobalNumElements()
                  << std::endl;
      }
      delete fout;
    }

    if (fmid != fin) delete fmid;
    delete fin;
  }

  return ierr;
}


////////////////////////////////////////////////////////////////////////////////
// How to run the tests within testDistSys
template <typename scalar_t>
int testDistSys<scalar_t>::run()
{
  int ierr = 0;

  ktensor_t *ktensor;
  sptensor_t *sptensor;
  std::vector<const map_t *> ktensorMaps(nModes);

  // Build sptensor
  sptensor = new sptensor_t(nModes, modeSizes, globalIndices, values, comm);

  ///////////////////////////////////////////////////////////
  // Build a ktensor that uses default Tpetra::Maps
  for (pt::mode_t m = 0; m < nModes; m++)
    ktensorMaps[m] = new map_t(modeSizes[m], 0, comm);
   
  ktensor = new ktensor_t(rank, ktensorMaps, comm);

  // Run the tests without and with optimized maps
  ierr += checkImportExport("defaultKtensorMaps", sptensor, ktensor, false);
  ierr += checkImportExport("defaultKtensorMapsOpt", sptensor, ktensor, true);

  // clean up before next test
  for (pt::mode_t m = 0; m < nModes; m++) delete ktensorMaps[m];
  delete ktensor;

  ///////////////////////////////////////////////////////////
  // Build a ktensor created from an array of cyclic maps 
  for (pt::mode_t m = 0; m < nModes; m++) {
    ktensorMaps[m] = buildCyclicMap(modeSizes[m]);
  }

  ktensor = new ktensor_t(rank, ktensorMaps, comm);

  // Run the tests without and with optimized maps
  ierr += checkImportExport("cyclicKtensorMaps", sptensor, ktensor, false);
  ierr += checkImportExport("cyclicKtensorMapsOpt", sptensor, ktensor, true);

  // clean up before next test
  for (pt::mode_t m = 0; m < nModes; m++) delete ktensorMaps[m];
  delete ktensor;

  ///////////////////////////////////////////////////////////
  // Build a ktensor using oneToOne versions of sptensor maps
  ktensor = new ktensor_t(rank, sptensor, comm);

  // Run the tests without and with optimized maps
  ierr += checkImportExport("oneToOneKtensorMaps", sptensor, ktensor, false);
  ierr += checkImportExport("oneToOneKtensorMapsOpt", sptensor, ktensor, true);

  // clean up before next test
  delete ktensor;

  // clean up
  delete sptensor;

  return ierr;
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

int main(int narg, char **arg)
{
  // Usual Teuchos::Comm initialization
  Tpetra::ScopeGuard scopeguard(&narg, &arg);
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int me = comm->getRank();
  int np = comm->getSize();

  int ierr = 0;
  pt::mode_t nModes;
  pt::rank_t rank;
  size_t nnz;

  { // A very small sptensor; all nonzeros initially on proc zero
    // Example from TTB test
    typedef double scalar_t;
    typedef typename testDistSys<scalar_t>::gno_t gno_t;

    nModes = 3; 
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = 2;
    modeSizes[1] = 3;
    modeSizes[2] = 4;

    nnz = (me == 0 ? 1 : 0);
    Kokkos::View<gno_t **> indices("indices", nnz, nModes);
    Kokkos::View<scalar_t *> values("values", nnz);

    if (me == 0) {
      indices(0,0) = 1;
      indices(0,1) = 1;
      indices(0,2) = 1;
      values(0) = 1.;
    }

    rank = 1;

    if (me == 0) {
      std::cout << std::endl
                << "TESTING WITH <DOUBLE> AND RANK=" 
                << rank << ", NMODES=" << nModes << " (";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
      std::cout << ")" << std::endl;
    }

    testDistSys<scalar_t> test(comm, modeSizes, rank, indices, values);
    ierr += test.run();
  }

  { // A generated tensor with 10 nonzeros per processor
    typedef float scalar_t;
    typedef typename testDistSys<scalar_t>::gno_t gno_t;
    nnz = 10;

    nModes = 4;
    std::vector<size_t> modeSizes(nModes);
    modeSizes[0] = np;
    modeSizes[1] = 2 * (np - 1) + (nnz - 1) + 1;
    modeSizes[2] = 2 * (np - 1) + 2 * (nnz - 1) + 1;
    modeSizes[3] = nnz;
   
    Kokkos::View<gno_t **> indices("indices", nnz, nModes);
    Kokkos::View<scalar_t *> values("values", nnz);
    
    for (size_t nz = 0; nz < nnz; nz++) {
      indices(nz, 0) = me;
      indices(nz, 1) = 2 * me + nz;
      indices(nz, 2) = 2 * (me + nz);  // only even indices used
      indices(nz, 3) = nz;
      values(nz) = 1.;
    }

    rank = 4;

    if (me == 0) {
      std::cout << std::endl
                << "TESTING WITH <FLOAT> AND RANK=" 
                << rank << ", NMODES=" << nModes << " (";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
      std::cout << ")" << std::endl;
    }

    testDistSys<scalar_t> test(comm, modeSizes, rank, indices, values);
    ierr += test.run();
  }

  { // A generated tensor with block-based random input
    typedef float scalar_t;
    typedef typename testDistSys<scalar_t>::gno_t gno_t;

    nnz = 1000;
    nModes = 3;
    std::vector<size_t> modeSizes(nModes);
    const size_t mult[3] = {100, 200, 300};
    modeSizes[0] = mult[0] * np;
    modeSizes[1] = mult[1] * np;
    modeSizes[2] = mult[2] * np;

    Kokkos::View<gno_t **> indices("indices", nnz, nModes);
    Kokkos::View<scalar_t *> values("values", nnz);

    srand(me);

    for (size_t nz = 0; nz < nnz; nz++) {
      for (pt::mode_t m = 0; m < nModes; m++)
        indices(nz, m) = mult[m] * me + rand() % mult[m];
      values(nz) = 1.;
    }

    rank = 10;

    if (me == 0) {
      std::cout << std::endl
                << "TESTING WITH <FLOAT> AND RANK=" 
                << rank << ", NMODES=" << nModes << " (";
      for (pt::mode_t m = 0; m < nModes; m++) std::cout << modeSizes[m] << " ";
      std::cout << ")" << std::endl;
    }

    testDistSys<scalar_t> test(comm, modeSizes, rank, indices, values);
    ierr += test.run();
  }

  if (ierr) std::cout << me << ": " << ierr << " errors detected." << std::endl;

  int gierr = 0;
  Teuchos::reduceAll<int,int>(*comm, Teuchos::REDUCE_SUM, 1, &ierr, &gierr);

  if (me == 0) {
    if (gierr == 0)
      std::cout << " PASS" << std::endl;
    else
      std::cout << " FAIL" << std::endl;
  }

  return gierr;
}
