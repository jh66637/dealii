// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


#ifndef dealii_matrix_free_fe_remote_evaluation_h
#define dealii_matrix_free_fe_remote_evaluation_h

#include <deal.II/base/mpi_remote_point_evaluation.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>


DEAL_II_NAMESPACE_OPEN

namespace internal
{
  /**
   * Type traits for supported FEEvaluationTypes. Different FEEvaluationTypes
   * need different communication objects and different access to data at
   * quadrature points. Each specialization defines its CommunicationObjectType
   * and whether a two level CRS structure is needed to access the data by the
   * memeber `cell_face_pairs`. The same type with a different number of
   * components can be obtained with FEEvaluationTypeComponents.
   */
  template <bool is_face, bool use_matrix_free_batches>
  struct FERemoteEvaluationTypeTraits
  {};



  /**
   * Specialization for FEEvaluation.
   */
  template <>
  struct FERemoteEvaluationTypeTraits<false, true>
  {
    static const bool use_two_level_crs = false;

    template <int dim>
    using CommunicationObjectType =
      std::pair<std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>>,
                std::vector<std::pair<unsigned int, unsigned int>>>;

    template <int dim,
              int n_components,
              typename Number,
              typename VectorizedArrayType = VectorizedArray<Number>>
    using value_type =
      typename FEEvaluationAccess<dim,
                                  n_components,
                                  Number,
                                  false,
                                  VectorizedArrayType>::value_type;

    template <int dim,
              int n_components,
              typename Number,
              typename VectorizedArrayType = VectorizedArray<Number>>
    using gradient_type =
      typename FEEvaluationAccess<dim,
                                  n_components,
                                  Number,
                                  false,
                                  VectorizedArrayType>::gradient_type;
  };



  /**
   * Specialization for FEFaceEvaluation.
   */
  template <>
  struct FERemoteEvaluationTypeTraits<true, true>
  {
    static const bool use_two_level_crs = false;

    template <int dim>
    using CommunicationObjectType =
      std::pair<std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>>,
                std::vector<std::pair<unsigned int, unsigned int>>>;

    template <int dim,
              int n_components,
              typename Number,
              typename VectorizedArrayType = VectorizedArray<Number>>
    using value_type =
      typename FEEvaluationAccess<dim,
                                  n_components,
                                  Number,
                                  true,
                                  VectorizedArrayType>::value_type;

    template <int dim,
              int n_components,
              typename Number,
              typename VectorizedArrayType = VectorizedArray<Number>>
    using gradient_type =
      typename FEEvaluationAccess<dim,
                                  n_components,
                                  Number,
                                  true,
                                  VectorizedArrayType>::gradient_type;
  };



  /**
   * Specialization for FEPointEvaluation.
   */
  template <>
  struct FERemoteEvaluationTypeTraits<false, false>
  {
    static const bool use_two_level_crs = false;

    template <int dim>
    using CommunicationObjectType =
      std::pair<std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>>,
                std::vector<typename Triangulation<dim>::cell_iterator>>;

    template <int dim, int n_components, typename Number, typename VectorizedArrayType /*only needed to access value types the same way*/>
    using value_type = typename internal::FEPointEvaluation::
      EvaluatorTypeTraits<dim, n_components, Number>::value_type;

    template <int dim, int n_components, typename Number, typename VectorizedArrayType /*only needed to access value types the same way*/>
    using gradient_type = typename internal::FEPointEvaluation::
      EvaluatorTypeTraits<dim, n_components, Number>::gradient_type;
  };



  /**
   * Specialization for FEPointEvaluation for faces.
   */
  template <>
  struct FERemoteEvaluationTypeTraits<true, false>
  {
    static const bool use_two_level_crs = true;

    template <int dim>
    using CommunicationObjectType = std::pair<
      std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>>,
      std::vector<
        std::pair<typename Triangulation<dim>::cell_iterator, unsigned int>>>;

    template <int dim, int n_components, typename Number, typename VectorizedArrayType /*only needed to access value types the same way*/>
    using value_type = typename internal::FEPointEvaluation::
      EvaluatorTypeTraits<dim, n_components, Number>::value_type;

    template <int dim, int n_components, typename Number, typename VectorizedArrayType /*only needed to access value types the same way*/>
    using gradient_type = typename internal::FEPointEvaluation::
      EvaluatorTypeTraits<dim, n_components, Number>::gradient_type;
  };



  /**
   * A class that stores values and/or gradients at quadrature points
   * corresponding to a FEEvaluationType (FEEvaluation, FEFaceEvaluation,
   * FEPointEvaluation).
   */
  template <int dim,
            int n_components,
            typename Number,
            typename VectorizedArrayType,
            bool is_face,
            bool use_matrix_free_batches>
  struct FERemoteEvaluationData
  {
    using FERETT =
      FERemoteEvaluationTypeTraits<is_face, use_matrix_free_batches>;


    // TODO: get from type traits
    using value_type = typename FERETT::
      template value_type<dim, n_components, Number, VectorizedArrayType>;
    using gradient_type = typename FERETT::
      template gradient_type<dim, n_components, Number, VectorizedArrayType>;

    /**
     * Values at quadrature points.
     */
    std::vector<value_type> values;

    /**
     * Gradients at quadrature points.
     */
    std::vector<gradient_type> gradients;
  };

  /**
   * A class that stores a CRS like structure to access
   * FERemoteEvaluationData. In most cases a one level CRS structure
   * is enough. In this case @c ptrs is constructed and the shift can be
   * obtained with `get_shift(index)`. The field @c ptrs_ptrs stays empty.
   * It is only filled if a two level structure is needed. In this case
   * `get_shift(cell_index, face_number)` return the correct shift.
   */
  struct FERemoteEvaluationDataView
  {
    /**
     * Get a pointer to data at index.
     */
    unsigned int get_shift(const unsigned int index) const
    {
      Assert(ptrs_ptrs.size() == 0, ExcMessage("Two level CRS set up"));

      Assert(index != numbers::invalid_unsigned_int,
             ExcMessage("Index has to be valid!"));

      Assert(start <= index, ExcInternalError());
      AssertIndexRange(index - start, ptrs.size());
      return ptrs[index - start];
    }

    /**
     * Get a pointer to data at (cell_index, face_number).
     */
    unsigned int get_shift(const unsigned int cell_index,
                           const unsigned int face_number) const
    {
      Assert(ptrs_ptrs.size() > 0, ExcMessage("No two level CRS set up"));

      Assert(cell_index != numbers::invalid_unsigned_int,
             ExcMessage("Cell index has to be valid!"));
      Assert(face_number != numbers::invalid_unsigned_int,
             ExcMessage("Face number has to be valid!"));

      Assert(start <= cell_index, ExcInternalError());

      AssertIndexRange(cell_index - start, ptrs_ptrs.size());
      const unsigned int face_index =
        ptrs_ptrs[cell_index - start] + face_number;
      AssertIndexRange(face_index, ptrs.size());
      return ptrs[face_index];
    }

    /**
     * Get the number of stored values.
     */
    unsigned int size() const
    {
      Assert(ptrs.size() > 0, ExcInternalError());
      return ptrs.back();
    }

    /**
     * This parameter can be used if indices do not start with 0.
     */
    unsigned int start = 0;

    /**
     * Pointers to @c ptrs.
     */
    std::vector<unsigned int> ptrs_ptrs;

    /**
     * Pointers to data.
     */
    std::vector<unsigned int> ptrs;
  };

} // namespace internal



/**
 * A class to fill the fields in FERemoteEvaluationData.
 * FERemoteEvaluation is thought to be used with another @p FEEvaluationType
 * (FEEvaluation, FEFaceEvaluation, or FEPointEvaluation). @p is_face specifies
 * if @p FEEvaluationType works on faces.
 */
template <int dim, bool is_face, bool use_matrix_free_batches>
class FERemoteEvaluationCommunicator : public Subscriptor
{
  using FERETT =
    typename internal::FERemoteEvaluationTypeTraits<is_face,
                                                    use_matrix_free_batches>;

  using CommunicationObjectType =
    typename FERETT::template CommunicationObjectType<dim>;

public:
  template <bool F = is_face, bool B = use_matrix_free_batches>
  typename std::enable_if<true == (F && B), void>::type reinit_faces(
    std::vector<std::pair<
      std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>>,
      std::vector<std::pair<unsigned int, unsigned int>>>> comm_objects,
    const std::pair<unsigned int, unsigned int>           &face_batch_range,
    const std::vector<Quadrature<dim>>                    &quadrature_vector)
  {
    communication_objects = comm_objects;
    // fetch points and update communication patterns

    const unsigned int n_cells = quadrature_vector.size();
    AssertDimension(n_cells, face_batch_range.second - face_batch_range.first);

    // construct view:
    view.start = face_batch_range.first;

    view.ptrs.resize(n_cells + 1);

    view.ptrs[0] = 0;
    for (unsigned int face = 0; face < n_cells; ++face)
      {
        view.ptrs[face + 1] = view.ptrs[face] + quadrature_vector[face].size();
      }
  }

  template <typename Iterator,
            bool F = is_face,
            bool B = use_matrix_free_batches>
  typename std::enable_if<true == (F && !B), void>::type reinit_faces(
    std::vector<std::pair<
      std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>>,
      std::vector<std::pair<typename Triangulation<dim>::cell_iterator,
                            unsigned int>>>>             comm_objects,
    const IteratorRange<Iterator>                       &cell_iterator_range,
    const std::vector<std::vector<Quadrature<dim - 1>>> &quadrature_vector)
  {
    communication_objects = comm_objects;

    const unsigned int n_cells = quadrature_vector.size();
    AssertDimension(n_cells,
                    std::distance(cell_iterator_range.begin(),
                                  cell_iterator_range.end()));

    // construct view:
    auto &cell_ptrs = view.ptrs_ptrs;
    auto &face_ptrs = view.ptrs;

    view.start = 0;
    cell_ptrs.resize(n_cells);
    unsigned int n_faces = 0;
    for (const auto &cell : cell_iterator_range)
      {
        cell_ptrs[cell->active_cell_index()] = n_faces;
        n_faces += cell->n_faces();
      }

    face_ptrs.resize(n_faces + 1);
    face_ptrs[0] = 0;
    for (const auto &cell : cell_iterator_range)
      {
        for (const auto &f : cell->face_indices())
          {
            const unsigned int face_index =
              cell_ptrs[cell->active_cell_index()] + f;

            face_ptrs[face_index + 1] =
              face_ptrs[face_index] +
              quadrature_vector[cell->active_cell_index()][f].size();
          }
      }
  }


  /**
   * Fill the fields stored in FERemoteEvaluationData.
   */
  template <int n_components,
            typename FERemoteEvaluationDataType,
            typename MeshType,
            typename VectorType>
  void update_ghost_values(
    FERemoteEvaluationDataType            &dst,
    const MeshType                        &mesh,
    const VectorType                      &src,
    const EvaluationFlags::EvaluationFlags eval_flags,
    const unsigned int                     first_selected_component,
    const VectorTools::EvaluationFlags::EvaluationFlags vec_flags) const
  {
    const bool has_ghost_elements = src.has_ghost_elements();

    if (has_ghost_elements == false)
      src.update_ghost_values();


    for (auto &communication_object : communication_objects)
      {
        if (eval_flags & EvaluationFlags::values)
          {
            copy_data(dst.values,
                      VectorTools::point_values<n_components>(
                        *communication_object.first,
                        mesh,
                        src,
                        vec_flags,
                        first_selected_component),
                      communication_object.second);
          }

        if (eval_flags & EvaluationFlags::gradients)
          {
            copy_data(dst.gradients,
                      VectorTools::point_gradients<n_components>(
                        *communication_object.first,
                        mesh,
                        src,
                        vec_flags,
                        first_selected_component),
                      communication_object.second);
          }

        Assert(!(eval_flags & EvaluationFlags::hessians), ExcNotImplemented());
      }

    if (has_ghost_elements == false)
      src.zero_out_ghost_values();
  }

  /**
   * Get a pointer to data at index.
   */
  template <bool F = is_face, bool B = use_matrix_free_batches>
  typename std::enable_if_t<false == (F && !B), unsigned int>
  get_shift(const unsigned int index) const
  {
    return view.get_shift(index);
  }

  /**
   * Get a pointer to data at (cell_index, face_number).
   */
  template <bool F = is_face, bool B = use_matrix_free_batches>
  typename std::enable_if_t<true == (F && !B), unsigned int>
  get_shift(const unsigned int cell_index, const unsigned int face_number) const
  {
    return view.get_shift(cell_index, face_number);
  }

private:
  /**
   * Copy data obtained with RemotePointEvaluation to corresponding field
   * FERemoteEvaluationData.
   */
  template <typename T1, typename T2>
  void copy_data(std::vector<T1>       &dst,
                 const std::vector<T2> &src,
                 const std::vector<typename Triangulation<dim>::cell_iterator>
                   &data_ptrs) const
  {
    dst.resize(view.size());

    unsigned int c = 0;
    for (const auto &data_ptr : data_ptrs)
      {
        for (unsigned int j = get_shift(data_ptr.first->active_cell_index());
             j < get_shift(data_ptr->active_cell_index() + 1);
             ++j, ++c)
          {
            AssertIndexRange(j, dst.size());
            AssertIndexRange(c, src.size());
            dst[j] = src[c];
          }
      }
  }

  /**
   * Copy data obtained with RemotePointEvaluation to corresponding field
   * FERemoteEvaluationData.
   */
  template <typename T1, typename T2>
  void copy_data(
    std::vector<T1>                            &dst,
    const std::vector<T2>                      &src,
    const std::vector<std::pair<typename Triangulation<dim>::cell_iterator,
                                unsigned int>> &data_ptrs) const
  {
    dst.resize(view.size());

    unsigned int c = 0;
    for (const auto &data_ptr : data_ptrs)
      {
        for (unsigned int j =
               get_shift(data_ptr.first->active_cell_index(), data_ptr.second);
             j < get_shift(data_ptr.first->active_cell_index(),
                           data_ptr.second + 1);
             ++j, ++c)
          {
            AssertIndexRange(j, dst.size());
            AssertIndexRange(c, src.size());

            dst[j] = src[c];
          }
      }
  }

  /**
   * Copy data obtained with RemotePointEvaluation to corresponding field
   * FERemoteEvaluationData.
   */
  template <typename T1, typename T2>
  void copy_data(
    std::vector<T1>                                          &dst,
    const std::vector<T2>                                    &src,
    const std::vector<std::pair<unsigned int, unsigned int>> &data_ptrs) const
  {
    dst.resize(view.size());

    unsigned int c = 0;
    for (const auto &data_ptr : data_ptrs)
      {
        const unsigned int bface     = data_ptr.first;
        const unsigned int n_entries = data_ptr.second;

        for (unsigned int v = 0; v < n_entries; ++v)
          for (unsigned int j = get_shift(bface); j < get_shift(bface + 1);
               ++j, ++c)
            {
              AssertIndexRange(j, dst.size());
              AssertIndexRange(c, src.size());

              copy_data_entries(dst[j], v, src[c]);
            }
      }
  }

  /**
   * Copy data between different data layouts.
   */
  template <typename T1, std::size_t n_lanes>
  void copy_data_entries(VectorizedArray<T1, n_lanes> &dst,
                         const unsigned int            v,
                         const T1                     &src) const
  {
    AssertIndexRange(v, n_lanes);

    dst[v] = src;
  }

  /**
   * Copy data between different data layouts.
   */
  template <typename T1, int rank_, std::size_t n_lanes, int dim_>
  void copy_data_entries(Tensor<rank_, dim_, VectorizedArray<T1, n_lanes>> &dst,
                         const unsigned int                                 v,
                         const Tensor<rank_, dim_, T1> &src) const
  {
    AssertIndexRange(v, n_lanes);

    if constexpr (rank_ == 1)
      {
        for (unsigned int i = 0; i < dim_; ++i)
          dst[i][v] = src[i];
      }
    else
      {
        for (unsigned int i = 0; i < rank_; ++i)
          for (unsigned int j = 0; j < dim_; ++j)
            dst[i][j][v] = src[i][j];
      }
  }

  /**
   * Copy data between different data layouts.
   */
  template <typename T1,
            int         rank_,
            std::size_t n_lanes,
            int         n_components_,
            int         dim_>
  void copy_data_entries(
    Tensor<rank_,
           n_components_,
           Tensor<rank_, dim_, VectorizedArray<T1, n_lanes>>>   &dst,
    const unsigned int                                           v,
    const Tensor<rank_, n_components_, Tensor<rank_, dim_, T1>> &src) const
  {
    if constexpr (rank_ == 1)
      {
        for (unsigned int i = 0; i < n_components_; ++i)
          copy_data(dst[i], v, src[i]);
      }
    else
      {
        for (unsigned int i = 0; i < rank_; ++i)
          for (unsigned int j = 0; j < n_components_; ++j)
            dst[i][j][v] = src[i][j];
      }
  }

  /**
   * CRS like data structure that describes the data positions at given
   * indices.
   */
  internal::FERemoteEvaluationDataView view;
  /**
   * RemotePointEvaluation objects and indices to points used in
   * RemotePointEvaluation.
   */
  std::vector<CommunicationObjectType> communication_objects;
};

/**
 * Class to access data in matrix-free loops for non-matching discretizations.
 * Interfaces are named with FEEvaluation, FEFaceEvaluation or FEPointEvaluation
 * in mind. The main difference is, that `gather_evaluate()` updates and caches
 * all values at once. Therefore, it has to be called only once before a
 * matrix-free loop.
 *
 * FERemoteEvaluation is thought to be used with another @p FEEvaluationType
 * (FEEvaluation, FEFaceEvaluation, or FEPointEvaluation).
 * FERemoteEvaluationCommunicator knows the type. However,
 * FERemoteEvaluationCommunicator is independent of @p n_components.
 */
template <int dim,
          int n_components,
          typename Number,
          typename VectorizedArrayType,
          bool is_face,
          bool use_matrix_free_batches>
class FERemoteEvaluationCache
{
  using FERemoteEvaluationCommunicatorType =
    FERemoteEvaluationCommunicator<dim, is_face, use_matrix_free_batches>;
  using FERETT =
    typename internal::FERemoteEvaluationTypeTraits<is_face,
                                                    use_matrix_free_batches>;


  using FERemoteEvaluationDataType =
    typename internal::FERemoteEvaluationData<dim,
                                              n_components,
                                              Number,
                                              VectorizedArrayType,
                                              is_face,
                                              use_matrix_free_batches>;
  using value_type    = typename FERemoteEvaluationDataType::value_type;
  using gradient_type = typename FERemoteEvaluationDataType::gradient_type;

public:
  /**
   * The constructor needs a corresponding FERemoteEvaluationCommunicator
   * which has to be setup outside of this class. This design choice is
   * motivated since the same FERemoteEvaluationCommunicator can be used
   * for different MeshTypes and number of components.
   *
   * @param[in] comm FERemoteEvaluationCommunicator.
   * @param[in] mesh Triangulation or DoFHandler.
   * @param[in] vt_flags Specify treatment of values at points which are found
   * on multiple cells.
   * @param[in] first_selected_component Select first component of evaluation in
   * DoFHandlers with multiple components.
   */
  template <typename MeshType>
  FERemoteEvaluationCache(const FERemoteEvaluationCommunicatorType &comm,
                          const MeshType                           &mesh,
                          const unsigned int first_selected_component = 0,
                          const VectorTools::EvaluationFlags::EvaluationFlags
                            vt_flags = VectorTools::EvaluationFlags::avg)
    : comm(&comm)
    , first_selected_component(first_selected_component)
    , vt_flags(vt_flags)
  {
    set_mesh(mesh);
  }

  /**
   * Update the data which can be accessed via `get_value()` and
   * `get_gradient()`.
   *
   * @param[in] src Solution vector used to update data.
   * @param[in] flags Evaluation flags. Currently supported are
   * EvaluationFlags::values and EvaluationFlags::gradients.
   */
  template <typename VectorType>
  void gather_evaluate(const VectorType                      &src,
                       const EvaluationFlags::EvaluationFlags flags)
  {
    if (tria)
      {
        AssertThrow(n_components == 1, ExcNotImplemented());
        comm->template update_ghost_values<n_components>(
          this->data, *tria, src, flags, first_selected_component, vt_flags);
      }
    else if (dof_handler)
      {
        comm->template update_ghost_values<n_components>(
          this->data,
          *dof_handler,
          src,
          flags,
          first_selected_component,
          vt_flags);
      }
    else
      AssertThrow(false, ExcNotImplemented());
  }

  /**
   * Get the value at quadrature point @p q. The entity on which the values
   * are defined is set via `reinit()`.
   */
  const value_type get_value(const unsigned int q,
                             const unsigned int data_offset) const
  {
    Assert(data_offset != numbers::invalid_unsigned_int,
           ExcMessage("reinit() not called."));
    AssertIndexRange(data_offset + q, data.values.size());
    return data.values[data_offset + q];
  }

  /**
   * Get the gradients at quadrature point @p q. The entity on which the
   * gradients are defined is set via `reinit()`.
   */
  const gradient_type get_gradient(const unsigned int q,
                                   const unsigned int data_offset) const
  {
    Assert(data_offset != numbers::invalid_unsigned_int,
           ExcMessage("reinit() not called."));
    AssertIndexRange(data_offset + q, data.gradients.size());
    return data.gradients[data_offset + q];
  }

  /**
   * Set entity index at which quadrature points are accessed. This can, e.g.,
   * a cell index, a cell batch index, or a face batch index.
   */
  template <bool T = FERETT::use_two_level_crs>
  typename std::enable_if_t<false == T, unsigned int>
  get_shift(const unsigned int index) const
  {
    return comm->get_shift(index);
  }

  /**
   * Set cell and face_number at which quadrature points are accessed.
   */
  template <bool T = FERETT::use_two_level_crs>
  typename std::enable_if_t<true == T, unsigned int>
  get_shift(const unsigned int cell_index, const unsigned int face_number) const
  {
    return comm->get_shift(cell_index, face_number);
  }

private:
  /**
   * Use Triangulation as MeshType.
   */
  void set_mesh(const Triangulation<dim> &tria)
  {
    this->tria = &tria;
  }

  /**
   * Use DoFHandler as MeshType.
   */
  void set_mesh(const DoFHandler<dim> &dof_handler)
  {
    this->dof_handler = &dof_handler;
  }

  /**
   * Data that is accessed by `get_value()` and `get_gradient()`.
   */
  FERemoteEvaluationDataType data;

  /**
   * Underlying communicator which handles update of the ghost values and
   * gives position of values and gradients stored in
   * FERemoteEvaluationData.
   */
  SmartPointer<const FERemoteEvaluationCommunicatorType> comm;

  /**
   * Pointer to MeshType if used with Triangulation.
   */
  SmartPointer<const Triangulation<dim>> tria;
  /**
   * Pointer to MeshType if used with DoFHandler.
   */
  SmartPointer<const DoFHandler<dim>> dof_handler;

  /**
   * First selected component.
   */
  const unsigned int first_selected_component;

  /**
   * Flags that indicate which ghost values are updated.
   */
  const VectorTools::EvaluationFlags::EvaluationFlags vt_flags;
};


template <int dim,
          int n_components,
          typename Number,
          typename VectorizedArrayType,
          bool is_face,
          bool use_matrix_free_batches>
class FERemoteEvaluation
{
  using FERETT =
    typename internal::FERemoteEvaluationTypeTraits<is_face,
                                                    use_matrix_free_batches>;

public:
  FERemoteEvaluation(
    const FERemoteEvaluationCache<dim,
                                  n_components,
                                  Number,
                                  VectorizedArrayType,
                                  is_face,
                                  use_matrix_free_batches> &cache)
    : cache(cache)
    , data_offset(numbers::invalid_unsigned_int){};

  /**
   * Get the value at quadrature point @p q. The entity on which the values
   * are defined is set via `reinit()`.
   */
  const auto get_value(const unsigned int q) const
  {
    return cache.get_value(q, data_offset);
  }

  /**
   * Get the gradients at quadrature point @p q. The entity on which the
   * gradients are defined is set via `reinit()`.
   */
  const auto get_gradient(const unsigned int q) const
  {
    return cache.get_gradient(q, data_offset);
  }


  /**
   * Set entity index at which quadrature points are accessed. This can, e.g.,
   * a cell index, a cell batch index, or a face batch index.
   */
  template <bool T = FERETT::use_two_level_crs>
  typename std::enable_if_t<false == T, void> reinit(const unsigned int index)
  {
    data_offset = cache.get_shift(index);
  }

  /**
   * Set cell and face_number at which quadrature points are accessed.
   */
  template <bool T = FERETT::use_two_level_crs>
  typename std::enable_if_t<true == T, void>
  reinit(const unsigned int cell_index, const unsigned int face_number)
  {
    data_offset = cache.get_shift(cell_index, face_number);
  }

private:
  const FERemoteEvaluationCache<dim,
                                n_components,
                                Number,
                                VectorizedArrayType,
                                is_face,
                                use_matrix_free_batches> &cache;

  /**
   * Offset to data after last call of `reinit()`.
   */
  unsigned int data_offset;
};

DEAL_II_NAMESPACE_CLOSE

#endif
