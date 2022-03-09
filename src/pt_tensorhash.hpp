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
 

#ifndef PT_TENSORHASH_
#define PT_TENSORHASH_

#include "Kokkos_UnorderedMap.hpp"

namespace pt {

    namespace Impl {

        // Statically sized array for storing tensor indices as keys in UnorderedMap
        template <typename T, unsigned N>
        class Array {
        public:
          typedef T &                                 reference ;
          typedef typename std::add_const<T>::type &  const_reference ;
          typedef size_t                              size_type ;
          typedef ptrdiff_t                           difference_type ;
          typedef T                                   value_type ;
          typedef T *                                 pointer ;
          typedef typename std::add_const<T>::type *  const_pointer ;

          KOKKOS_INLINE_FUNCTION static constexpr size_type size() { return N ; }

          template< typename iType >
          KOKKOS_INLINE_FUNCTION
          reference operator[]( const iType & i ) { return x[i]; }

          template< typename iType >
          KOKKOS_INLINE_FUNCTION
          const_reference operator[]( const iType & i ) const { return x[i]; }

        private:
          T x[N];

        };
    
    }

  // A container for storing hash maps with various size Array<T,N> keys
  template <typename SCALAR, typename INDX>
  class TensorHashMap {
  public:

    TensorHashMap() = default;

    TensorHashMap(INDX ndim_, INDX kddnnz) : ndim(ndim_) {
      INDX nnz = 2 * kddnnz;
      if (ndim == 3) map_3 = map_type_3(nnz);
      else if (ndim == 4) map_4 = map_type_4(nnz);
      else if (ndim == 5) map_5 = map_type_5(nnz);
      else if (ndim == 6) map_6 = map_type_6(nnz);
      else {
        throw std::runtime_error("Invalid tensor dimension for hash map!");
      }
    }

    template <typename ind_t>
    KOKKOS_INLINE_FUNCTION
    void insert(const ind_t& ind, const SCALAR val) const {
      if (ndim == 3) insert_map(ind, val, map_3);
      else if (ndim == 4) insert_map(ind, val, map_4);
      else if (ndim == 5) insert_map(ind, val, map_5);
      else if (ndim == 6) insert_map(ind, val, map_6);
      return;
    }

    template <typename ind_t>
    KOKKOS_INLINE_FUNCTION
    bool exists(const ind_t& ind) const {
      if (ndim == 3) return exists_map(ind, map_3);
      else if (ndim == 4) return exists_map(ind, map_4);
      else if (ndim == 5) return exists_map(ind, map_5);
      else if (ndim == 6) return exists_map(ind, map_6);
      return false;
    }

    void print_histogram(std::ostream& out) {
      if (ndim == 3) print_histogram_map(out, map_3);
      else if (ndim == 4) print_histogram_map(out, map_4);
      else if (ndim == 5) print_histogram_map(out, map_5);
      else if (ndim == 6) print_histogram_map(out, map_6);
      return;
    }

  private:

    typedef Impl::Array<INDX, 3> key_type_3;
    typedef Impl::Array<INDX, 4> key_type_4;
    typedef Impl::Array<INDX, 5> key_type_5;
    typedef Impl::Array<INDX, 6> key_type_6;

    typedef Kokkos::UnorderedMap<key_type_3, SCALAR> map_type_3;
    typedef Kokkos::UnorderedMap<key_type_4, SCALAR> map_type_4;
    typedef Kokkos::UnorderedMap<key_type_5, SCALAR> map_type_5;
    typedef Kokkos::UnorderedMap<key_type_6, SCALAR> map_type_6;

    INDX ndim;
    map_type_3 map_3;
    map_type_4 map_4;
    map_type_5 map_5;
    map_type_6 map_6;

    template <typename ind_t, typename map_type>
    KOKKOS_INLINE_FUNCTION
    void insert_map(const ind_t& ind, const SCALAR val, map_type& map) const {
      typename map_type::key_type key;
      for (INDX i=0; i<ndim; ++i)
        key[i] = ind[i];
      if (map.insert(key,val).failed())
        Kokkos::abort("Hash map insert failed!");
    }

    template <typename ind_t, typename map_type>
    KOKKOS_INLINE_FUNCTION
    bool exists_map(const ind_t& ind, map_type& map) const {
      typename map_type::key_type key;
      for (INDX i=0; i<ndim; ++i)
        key[i] = ind[i];
      return map.exists(key);
    }

    template <typename map_type>
    void print_histogram_map(std::ostream& out, map_type& map) const {
      auto h = map.get_histogram();
      h.calculate();
      out << "length:" << std::endl;
      h.print_length(out);
      out << "distance:" << std::endl;
      h.print_distance(out);
      out << "block distance:" << std::endl;
      h.print_block_distance(out);
    }
  };

}

#endif
