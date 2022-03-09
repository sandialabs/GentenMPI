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
 

#ifndef PT_FACTORMATRIX_
#define PT_FACTORMATRIX_

#include "pt_shared.h"  // defines pt::layout_t
#include "pt_lrmv_decl.hpp"
#include "pt_lrmv_def.hpp"
#include "Tpetra_MultiVector.hpp"

namespace pt{

enum factormatrixNormType {
  NORM_INF,
  NORM_ONE,
  NORM_TWO
};

template <typename SCALAR=double>
class distFactorMatrix {
public:

  typedef SCALAR scalar_t;
  typedef pt::local_ordinal_type lno_t;
  typedef pt::global_ordinal_type gno_t;

  typedef distFactorMatrix<scalar_t> factormatrix_t;

  // Each factor matrix U_m is a Tpetra::MultiVector with 
  // numVectors = rank and 
  // globalLength = modeSizes[m]

#ifdef PT_LAYOUTRIGHT
  // LayoutRightMultiVector is a Tpetra::MultiVector hacked with LayoutRight
  typedef Tpetra::LayoutRightMultiVector<scalar_t, lno_t, gno_t> mvector_t;
#else
  // Tpetra default MV layout is LayoutLeft, but that's not a good layout
  // for factormatrix access; prefer LayoutRight for performance.
  typedef Tpetra::MultiVector<scalar_t, lno_t, gno_t> mvector_t;
#endif

  typedef typename mvector_t::dual_view_type::t_host valueview_t;
  typedef Tpetra::Map<lno_t, gno_t> map_t;
  
  // Need to be able to import and export factor matrices
  typedef Tpetra::Import<lno_t, gno_t> import_t;
  typedef Tpetra::Export<lno_t, gno_t> export_t;

  // Needed for getLocalView().  
  // TODO:  Check whether we are using getLocalView appropriately.  
  // TODO:  Trying to avoid using Teuchos::ArrayRCP
  typedef typename mvector_t::node_type::memory_space memoryspace_t;

  // Constructor that allows user to provide an RCP to a map
  distFactorMatrix(rank_t rank, const Teuchos::RCP<const map_t> &map) :
                   multivector(NULL), 
                   comm(map->getComm())
  {
    // TODO:  error check that map is 1-to-1
    multivector = new mvector_t(map, rank);
//    if (comm->getRank() == 0)
//      printLayout(getLocalView());
  }

  // Constructor that allows user to provide a map
  distFactorMatrix(rank_t rank, const map_t * const map) :
                   multivector(NULL), 
                   comm(map->getComm())
  {
    // TODO:  error check that map is 1-to-1
    multivector = new mvector_t(Teuchos::rcp(map, false), rank);
//    if (comm->getRank() == 0)
//      printLayout(getLocalView());
  }

  // Constructor that allows user to provide a map
  distFactorMatrix(const map_t * const map,
                   const valueview_t &hostView) :
                   multivector(NULL), 
                   comm(map->getComm())
  {
    // TODO:  error check that map is 1-to-1
    using dual_t = typename mvector_t::dual_view_type;
    using dev_t = typename dual_t::t_dev;

    auto devView = create_mirror_view_and_copy(typename dev_t::memory_space(),
                                               hostView);
    dual_t dualView(devView, hostView);

    multivector = new mvector_t(Teuchos::rcp(map, false), dualView);
  }


  // Constructor that uses default Trilinos map.  
  // This constructor may induce lots of communication.
  distFactorMatrix(rank_t rank, size_t globalLength,
                   const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
                   multivector(NULL),
                   comm(comm_)
  {
    const gno_t mingid = 0;
    Teuchos::RCP<const map_t> map = 
                              Teuchos::rcp(new map_t(globalLength,mingid,comm));
    multivector = new mvector_t(map, rank);
//    if (comm->getRank() == 0)
//      printLayout(getLocalView());
  }


  // Constructor that allows user to provide a view of initial factor matrix
  // values for warm-starts in a Kokkos::View
  distFactorMatrix(const Kokkos::View<const gno_t*, Kokkos::HostSpace> &myGids,
                   const valueview_t &hostView,
                   const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
                   multivector(NULL), 
                   comm(comm_)
  {
    using dual_t = typename mvector_t::dual_view_type;
    using dev_t = typename dual_t::t_dev;

    auto devView = create_mirror_view_and_copy(typename dev_t::memory_space(),
                                               hostView);
    dual_t dualView(devView, hostView);

    
    Tpetra::global_size_t dummy =
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();

    Teuchos::RCP<const map_t> map = 
             Teuchos::rcp(new map_t(dummy, myGids, 0, comm));

    multivector = new mvector_t(map, dualView);
  }

  // Constructor that allows user to provide a view of initial factor matrix
  // values for warm-starts in a vectorOfPtrs (one per rank)
  distFactorMatrix(const rank_t rank, 
                   const std::vector<gno_t> &myGids,
                   const std::vector<scalar_t *> &vectorOfPtrs,
                   const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
                   multivector(NULL), 
                   comm(comm_)
  {
    // TODO:  This constructor will not work with GPUs without UVM, and
    // may not work with GPUs with UVM.  (Of course, neither will the rest
    // of gentenmpi.  Very sad.)
    using dual_t = typename mvector_t::dual_view_type;
    using dev_t = typename dual_t::t_dev;
    using host_t = typename dual_t::t_host;

    size_t stride = (rank > 1 ? vectorOfPtrs[1] - vectorOfPtrs[0] 
                              : myGids.size());
    if (rank > 2) {
      // error check -- to cast to Kokkos::View<scalar_t**>,
      // vectorOfPtrs must have equal stride between ptrs
      for (rank_t r = 2; r < rank; r++) {
        auto newstride = vectorOfPtrs[r] - vectorOfPtrs[r-1];
        if (newstride != stride) {
          throw std::runtime_error("distFactorMatrix constructor requires "
                                   "equally strided vectorOfPtrs (i.e., same "
                                   "number of bytes between successive "
                                   "entries)");
        }
      }
    }

    // we know that all pointers are the same distance apart
    // cast the total memory space of the pointers to a Kokkos 2D view
    // with first dimension = number of scalars they are apart
    size_t nLocal = myGids.size();
    host_t hostViewStride(vectorOfPtrs[0], stride, rank);  // unmanaged view

    host_t hostView;

    // If stride == number of GIDs, we have the view that we want.  
    // Otherwise, subview the view down to only the number of GIDs.

    if (stride == nLocal) {
      hostView = hostViewStride;
    }
    else {
      const std::pair<size_t, size_t> gidRng(0, nLocal);
      hostView = Kokkos::subview(hostViewStride, gidRng, Kokkos::ALL());
    }

    // Then the rest follows like the constructor with a KokkosView
    auto devView = create_mirror_view_and_copy(typename dev_t::memory_space(),
                                               hostView);
    dual_t dualView(devView, hostView);
    
    Tpetra::global_size_t dummy =
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();

    auto myGidsView = 
         Kokkos::View<const gno_t*, Kokkos::HostSpace>(myGids.data(), nLocal);

    Teuchos::RCP<const map_t> map = 
             Teuchos::rcp(new map_t(dummy, myGidsView, 0, comm));

    multivector = new mvector_t(map, dualView);
  }


  // Copy constructor
  distFactorMatrix(const distFactorMatrix &fm) :
    multivector(new mvector_t(*(fm.getMultiVector()), Teuchos::Copy)),
    comm(fm.getComm())
  { }

  // Empty constructor; needed by Kokkos::View initializer
  KOKKOS_INLINE_FUNCTION
  distFactorMatrix() = default;


  // Destructor
  ~distFactorMatrix() { delete multivector; }


  // Import src factor matrix values to this factor matrix
  inline void doImport(factormatrix_t * const src, 
                       const import_t * const importer,
                       Tpetra::CombineMode op) 
  {
    multivector->doImport(*(src->getMultiVector()), *importer, op);
  } 


  // Export factor matrix values to this factor matrix from src
  inline void doExport(factormatrix_t * const src,
                       const export_t * const exporter,
                       Tpetra::CombineMode op) 
  {
    multivector->doExport(*(src->getMultiVector()), *exporter, op);
  } 

  // Export factor matrix values to this factor matrix from src; 
  // use reverse-mode communication and import object
  inline void doExport(factormatrix_t * const src,
                       const import_t * const importer,
                       Tpetra::CombineMode op) 
  {
    multivector->doExport(*(src->getMultiVector()), *importer, op);
  } 


  // Initialize entire factor matrix to value val
  inline void setValues(scalar_t val) { multivector->putScalar(val); }

  // Initialize entire factor matrix to random values
  inline void randomize() { multivector->randomize(); }

  // Return innerproduct with a compatible factor matrix
  inline scalar_t innerprod(const factormatrix_t &other, 
                            Kokkos::View<scalar_t *> lambda)
  {
    rank_t rank = multivector->getNumVectors();
    Kokkos::View<scalar_t *> dots("dots", rank);
    const factormatrix_t::mvector_t *otherMV = other.getMultiVector();

    multivector->dot(*otherMV, dots);

    scalar_t sum = 0;
    for (rank_t r = 0; r < rank; r++) sum += dots(r) * lambda(r);
    return sum;
  }

  // Return L1-norm of multivector
  inline void norm1(Kokkos::View<scalar_t *> &result) 
  { 
    return multivector->norm1(result);
  }

  // Return L2-norm of multivector
  inline void norm2(Kokkos::View<scalar_t *> &result) 
  { 
    return multivector->norm2(result);
  }
  
  // Return Inf-norm of multivector
  inline void normInf(Kokkos::View<scalar_t *> &result) 
  { 
    return multivector->normInf(result);
  }
  
  // Return rank of factor matrix 
  inline rank_t getFactorRank() const { return multivector->getNumVectors(); }


  // Return length of factor matrix on local proc
  inline size_t getLocalLength() const { return multivector->getLocalLength(); }

  // Return global length of factor matrix
  inline size_t getGlobalLength() const 
  {
    return multivector->getGlobalLength();
  }

  // Return Tpetra::Map associated with factor matrix
  inline const map_t *getMap() const 
  { 
    return multivector->getMap().getRawPtr(); 
  }

  // Return Tpetra::Comm associated with factor matrix
  inline const Teuchos::RCP<const Teuchos::Comm<int> > &getComm() const 
  { 
    return comm;
  }

  // Return Kokkos::View of storage in factor matrix
  inline valueview_t getLocalView() const 
  {
    return multivector->template getLocalView<memoryspace_t>();
  }


  // Return factor matrix entry corresponding to local index lid and rank
  scalar_t getLocalEntry(const lno_t lid, const rank_t rank) const {

    DBGASSERT((rank >= 0) && (size_t(rank) < multivector->getNumVectors()), 
              "getLocalEntry: invalid rank");
    DBGASSERT((lid >= 0) && (size_t(lid) < multivector->getLocalLength()),
              "getLocalEntry: invalid local index");

    return multivector->template getLocalView<memoryspace_t>()(lid, rank);
  }


  // Overloaded () operators
  // Return reference to factor matrix entry corresponding to 
  // rank and local index lid
  inline scalar_t &operator() (const lno_t lid, const rank_t rank) 
  {
    return multivector->template getLocalView<memoryspace_t>()(lid, rank);
  }

  inline const scalar_t &operator() (const lno_t lid, const rank_t rank) const
  {
    return multivector->template getLocalView<memoryspace_t>()(lid, rank);
  }


  // Replace the factor matrix entry corresponding to GLOBAL indices.
  // Convenience function only for testing; shouldn't use this method in 
  // real operations
  // Attempts to replace a global value on a processor that doesn't own
  // that value will be ignored.
  void replaceGlobalValue(const gno_t gid, const rank_t rank,
                          const scalar_t newValue) 
  { 
    if (multivector->getMap()->isNodeGlobalElement(gid)) {
      lno_t lid = multivector->getMap()->getLocalElement(gid); 
      multivector->template getLocalView<memoryspace_t>()(lid, rank) = newValue;
    }
    else {
      return;  // ignore attempts to set global value if don't own it

      // Rather than ignoring, we could consider a non-local replacement to
      // be an error.

      // std::ostringstream msg;
      // msg << "replaceGlobalValue: gid " << gid << " not on this processor"
      //     << std::endl << e.what() << std::endl;
      // throw std::runtime_error(msg.str());
    }
  }


  // Scale the factor matrix 
  inline void scale(const Kokkos::View<scalar_t *> scalefactors)
  {
    multivector->scale(scalefactors);
  }

  // Scale the factor matrix by the inverse of scalefactors' values
  inline void scaleInverse(const Kokkos::View<scalar_t *> scalefactors)
  {
    size_t len = scalefactors.extent(0);
    Kokkos::View<scalar_t *> invscalefactors("invscale", len);
    for (size_t i = 0; i < len; i++) 
      if (scalefactors(i) != 0.) invscalefactors(i) = 1. / scalefactors(i);
      else {
        throw std::runtime_error("zero-valued scalefactor in scaleInverse");
      }
    multivector->scale(invscalefactors);
  }


  // Normalize the factor matrix; return the norms in argument
  void normalize(Kokkos::View<scalar_t *> &norms, 
                 enum factormatrixNormType normType = NORM_TWO)
  {
    switch (normType) {
      case NORM_ONE:
        multivector->norm1(norms);
        break;
      case NORM_TWO:
        multivector->norm2(norms);
        break;
      case NORM_INF:
        multivector->normInf(norms);
        break;
    };
    rank_t rank = getFactorRank();
    Kokkos::View<scalar_t *> oneOverNorms("distFM::oneOverNorms", rank);
    for (rank_t r = 0; r < rank; r++) 
      if (norms(r) != 0.) oneOverNorms(r) = 1. / norms(r);
      else                oneOverNorms(r) = 1.;
    multivector->scale(oneOverNorms);
    return;
  }

  // Print factor matrix data, using Tpetra's describe function
  void print(const std::string &msg, std::ostream &ostr = std::cout) const { 

    if (comm->getRank() == 0) 
      ostr << "Distributed Factor Matrix " << msg << std::endl;

    Teuchos::FancyOStream fancy(Teuchos::rcpFromRef(ostr));
    multivector->describe(fancy, Teuchos::VERB_EXTREME); 
  }

  // TODO:  Constructor that initializes the factor matrix with input values
  // Tpetra MultiVector needs LayoutLeft to compile
  // TODO:  Do we need DualView here?  Or is View OK?
  //typedef Kokkos::View<scalar_t**, Kokkos::LayoutLeft> kokkosViewForMV_t;
  //kokkosViewForMV_t memory("multivector.MVmemory",
  //                        map->getNodeNumElements(), rank);
  //multivector = new mvector_t(map, memory);

  // Function that deep copies factor matrix from src to this factor matrix
  // Factor matrices are assumed to have identical maps.  
  // To save time, function does not check this condition.
  void copyData(factormatrix_t *src) {
    auto srcView = src->getLocalView();
    auto view = getLocalView();
    Kokkos::deep_copy(view, srcView);
  }

private:

  mvector_t *multivector;

  Teuchos::RCP<const Teuchos::Comm<int> > comm;

  // Function to return actual multivector
  inline mvector_t *getMultiVector() const { return multivector; }

  // Functions to print whether factor matrix is LayoutLeft or LayoutRight
  void printLayout(const Kokkos::View<scalar_t **, Kokkos::LayoutLeft> &data)
  { std::cout << "Factor matrix is LayoutLeft" << std::endl; }

  void printLayout(const Kokkos::View<scalar_t **, Kokkos::LayoutRight> &data)
  { std::cout << "Factor matrix is LayoutRight" << std::endl; }
};

}

#endif
