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
#include <strstream>
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Kokkos_Core.hpp>

#include <Zoltan2_VectorAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_PartitioningSolution.hpp>

namespace Zoltan2 {

template <typename GlobalOrdinal>
struct InputTraits<Kokkos::View<GlobalOrdinal **> >
{
  typedef GlobalOrdinal scalar_t;  // MJ coords are gno indices
  typedef pt::local_ordinal_type lno_t;     // We don't use lno_t, so who cares!
  typedef GlobalOrdinal gno_t;
  typedef GlobalOrdinal offset_t;
  typedef Zoltan2::default_part_t part_t;
  typedef Zoltan2::default_node_t node_t;  // Might need to change this later
  static inline std::string name() {return "gnoview_t";}

  Z2_STATIC_ASSERT_TYPES // validate the types
};

}

template <typename gnoview_t>
  class tensorEntriesAdapter : public Zoltan2::VectorAdapter<gnoview_t> {

public:

  typedef typename Zoltan2::InputTraits<gnoview_t>::gno_t gno_t;
  typedef typename Zoltan2::InputTraits<gnoview_t>::scalar_t scalar_t;

  tensorEntriesAdapter(gnoview_t &gids_,
                       const Teuchos::RCP<const Teuchos::Comm<int> > &comm_) :
    gids(gids_), comm(comm_),
    nnz(gids_.extent(0)), nModes(gids_.extent(1)),
    gnos("gnos", nnz)
  {
    // Zoltan2 requires unique names for IDs; generate them.
    size_t scannnz;
    Teuchos::scan(*comm, Teuchos::REDUCE_SUM, 1, &nnz, &scannnz);
    gno_t myFirstGno = scannnz - nnz;
    for (size_t i = 0; i < nnz; i++) gnos(i) = myFirstGno+i;
  };

  size_t getLocalNumIDs() const { return nnz; }

  void getIDsView(const gno_t *&ids) const { ids = gnos.data(); }

  int getNumWeightsPerID() const { return 0; }

  int getNumEntriesPerID() const { return nModes; }

  void getEntriesView(const scalar_t *&coords, int &stride, int idx = 0) const
  {
    auto subview = Kokkos::subview(gids, Kokkos::ALL(), idx);
    coords = subview.data();
    stride = gids.stride(0);
  }

private:
  
  const gnoview_t gids;
  const Teuchos::RCP<const Teuchos::Comm<int> > comm;

  size_t nnz;
  size_t nModes;
  Kokkos::View<gno_t *> gnos;  
};


template <typename gnoview_t, typename valueview_t>
void zoltanRebalance(gnoview_t &gids, valueview_t &vals,
                     const Teuchos::RCP<const Teuchos::Comm<int> > &comm,
                     int *dims_3d)
{
  size_t nnz = gids.extent(0);
  int nModes = gids.extent(1);

  Teuchos::RCP<Teuchos::Time> timePart(
                        Teuchos::TimeMonitor::getNewTimer("00 REBAL Part"));
  Teuchos::RCP<Teuchos::Time> timeEval(
                        Teuchos::TimeMonitor::getNewTimer("00 REBAL Eval"));
  Teuchos::RCP<Teuchos::Time> timeMig(
                        Teuchos::TimeMonitor::getNewTimer("00 REBAL Migrate"));

  // Partition
  timePart->start();

  typedef tensorEntriesAdapter<gnoview_t> ia_t;
  ia_t ia(gids, comm);

  Teuchos::ParameterList params;
  params.set("algorithm", "multijagged");
  std::stringstream ostr;
  ostr << dims_3d[0];
  for (int m = 1; m < nModes; m++) ostr << "," << dims_3d[m];
  if (comm->getRank() == 0)
    std::cout << "KDDKDD mj_parts " << ostr.str().c_str() << std::endl;
  params.set("mj_parts", ostr.str().c_str());

  Zoltan2::PartitioningProblem<ia_t> problem(&ia, &params, comm);
  try {
    problem.solve();
  }
  catch (std::exception &e) {
    std::cout << "Error in zoltanRebalance: solve " << e.what() << std::endl;
    throw e;
  }

  timePart->stop();

  // Evaluate partitions (old and new)

  timeEval->start();

  Zoltan2::EvaluatePartition<ia_t> before(&ia, &params, comm, NULL);
  Zoltan2::EvaluatePartition<ia_t> after(&ia, &params, comm, 
                                         &problem.getSolution());
  if (comm->getRank() == 0) {
    std::cout << "NONZERO BALANCE BEFORE PARTITIONING" << std::endl;
    before.printMetrics(std::cout);
    std::cout << "NONZERO BALANCE AFTER PARTITIONING " << std::endl;
    after.printMetrics(std::cout);
  }

  timeEval->stop();
  
  // Migrate GIDs and values

  timeMig->start();

  ZOLTAN_COMM_OBJ *plan;

  const Teuchos::MpiComm<int> *tmpicomm =
        dynamic_cast<const Teuchos::MpiComm<int> *>(comm.getRawPtr());
  MPI_Comm mpiComm = *(tmpicomm->getRawMpiComm());

  int newNNZ;
  int *newParts = const_cast<int *>(problem.getSolution().getPartListView());

  Zoltan_Comm_Create(&plan, nnz, newParts, mpiComm, 123, &newNNZ);

  valueview_t newValview("newVals", newNNZ);

  Zoltan_Comm_Do(plan, 234, 
                 reinterpret_cast<char *>(vals.data()),
                 sizeof(typename valueview_t::value_type),
                 reinterpret_cast<char *>(newValview.data()));

  gnoview_t newGIDview("newGIDs", newNNZ, nModes);

  // NOTE:  assumes Kokkos::LayoutRight
  Zoltan_Comm_Do(plan, 345, 
                 reinterpret_cast<char *>(gids.data()), 
                 sizeof(typename gnoview_t::value_type)*nModes,
                 reinterpret_cast<char *>(newGIDview.data()));

  Zoltan_Comm_Destroy(&plan);

  gids = newGIDview;
  vals = newValview;

  timeMig->stop();
}

