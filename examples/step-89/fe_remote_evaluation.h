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

#include <variant>

DEAL_II_NAMESPACE_OPEN

namespace internal
{
  /**
   * A class that stores values and/or gradients at quadrature points
   * corresponding to a FEEvaluationType.
   */
  template <int dim, int n_components, typename value_type_>
  struct PrecomputedFEEvaluationData
  {
    using value_type = typename internal::FEPointEvaluation::
      EvaluatorTypeTraits<dim, n_components, value_type_>::value_type;

    using gradient_type = typename internal::FEPointEvaluation::
      EvaluatorTypeTraits<dim, n_components, value_type_>::gradient_type;

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
   * PrecomputedFEEvaluationData. In most cases a one level CRS structure
   * is enough. In this case @c ptrs has to be constructed and the shift can be
   * obtained with `get_shift(index)`. The field @c ptrs_ptrs stays empty.
   * It is only filled if a two level structure is needed. In this case
   * `get_shift(cell_index, face_number)` return the correct shift.
   */
  struct PrecomputedFEEvaluationDataView
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
     * Get a pointer to data at, e.g. (cell_index, face_number).
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

  /**
   * A class helps to access @c PrecomputedFEEvaluationData in a thread safe
   * manner. Each thread has to create this wrapper class on its own to
   * avoid race-conditions during @c reinit().
   */
  template <int dim, int n_components, typename value_type>
  class PrecomputedFEEvaluationDataAccessor
  {
  public:
    PrecomputedFEEvaluationDataAccessor(
      const PrecomputedFEEvaluationData<dim, n_components, value_type> &data,
      const PrecomputedFEEvaluationDataView                            &view)
      : view(view)
      , data(data)
      , data_offset(numbers::invalid_unsigned_int)
    {}

    /**
     * Get the value at quadrature point @p q. The entity on which the values
     * are defined is set via `reinit()`.
     */
    const auto get_value(const unsigned int q) const
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
    const auto get_gradient(const unsigned int q) const
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
    void reinit(const unsigned int index)
    {
      data_offset = view.get_shift(index);
    }

    /**
     * Set cell and face_number at which quadrature points are accessed.
     */
    void reinit(const unsigned int cell_index, const unsigned int face_number)
    {
      data_offset = view.get_shift(cell_index, face_number);
    }

  private:
    /**
     * PrecomputedFEEvaluationDataView provides information where values are
     * located.
     */
    const PrecomputedFEEvaluationDataView &view;

    /**
     * PrecomputedFEEvaluationData stores the actual values.
     */
    const PrecomputedFEEvaluationData<dim, n_components, value_type> &data;

    /**
     * Offset to data after last call of `reinit()`.
     */
    unsigned int data_offset;
  };


} // namespace internal

/**
 * Communication objects know about the communication pattern.
 * In case of (matrix-free) cells batches or faces batches
 * a @c RemotePointEvaluation object stores the location of the
 * remote points. @c batch_id_n_entities relates these points
 * to the corresponding quadrature points of entity batches.
 * For this the field stores batch IDs and the number of entities
 * in the batch.
 */
template <int dim>
struct FERemoteCommunicationObjectEntityBatches
{
  std::vector<std::pair<unsigned int, unsigned int>> batch_id_n_entities;
  std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>> rpe;

  std::vector<std::pair<unsigned int, unsigned int>> get_pntrs() const
  {
    return batch_id_n_entities;
  };
};

/**
 * Similar as @FERemoteCommunicationObjectEntityBatches.
 * To relate the points from @c RemotePointEvaluation to
 * quadrature points on corresponding cells, cell iterators
 * have to be stored.
 */
template <int dim>
struct FERemoteCommunicationObjectCells
{
  std::vector<typename Triangulation<dim>::cell_iterator>     cells;
  std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>> rpe;

  std::vector<typename Triangulation<dim>::cell_iterator> get_pntrs() const
  {
    return cells;
  };
};

/**
 * Similar as @FERemoteCommunicationObjectCells.
 * To relate the points from @c RemotePointEvaluation to
 * quadrature points on corresponding faces, cell iterators
 * and face numbers have to be stored.
 */
template <int dim>
struct FERemoteCommunicationObjectFaces
{
  std::vector<
    std::pair<typename Triangulation<dim>::cell_iterator, unsigned int>>
                                                              cell_face_nos;
  std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>> rpe;

  std::vector<
    std::pair<typename Triangulation<dim>::cell_iterator, unsigned int>>
  get_pntrs() const
  {
    return cell_face_nos;
  };
};

/**
 * A class to fill the fields in PrecomputedFEEvaluationData.
 */
template <int dim>
class FERemoteEvaluationCommunicator : public Subscriptor
{
public:
  /**
   * This function stores given communication objects
   * and constructs a @c PrecomputedFEEvaluationDataView
   * object if remote points are related to matrix-free face batches.
   */
  void
  reinit_faces(const std::vector<FERemoteCommunicationObjectEntityBatches<dim>>
                                                           &comm_objects,
               const std::pair<unsigned int, unsigned int> &face_batch_range,
               const std::vector<Quadrature<dim>>          &quadrature_vector)
  {
    // erase type by converting to the base object
    communication_objects.clear();
    for (const auto &co : comm_objects)
      communication_objects.push_back(co);


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

  /**
   * This function stores given communication objects
   * and constructs a @c PrecomputedFEEvaluationDataView
   * object if remote points are related to faces of given
   * cells.
   */
  template <typename Iterator>
  void reinit_faces(
    const std::vector<FERemoteCommunicationObjectFaces<dim>> &comm_objects,
    const IteratorRange<Iterator>                       &cell_iterator_range,
    const std::vector<std::vector<Quadrature<dim - 1>>> &quadrature_vector)
  {
    // erase type
    communication_objects.clear();
    for (const auto &co : comm_objects)
      communication_objects.push_back(co);

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
   * Fill the fields stored in PrecomputedFEEvaluationData.
   */
  template <int n_components,
            typename PrecomputedFEEvaluationDataType,
            typename MeshType,
            typename VectorType>
  void update_ghost_values(
    PrecomputedFEEvaluationDataType       &dst,
    const MeshType                        &mesh,
    const VectorType                      &src,
    const EvaluationFlags::EvaluationFlags eval_flags,
    const unsigned int                     first_selected_component,
    const VectorTools::EvaluationFlags::EvaluationFlags vec_flags) const
  {
    const bool has_ghost_elements = src.has_ghost_elements();

    if (has_ghost_elements == false)
      src.update_ghost_values();


    for (const auto &communication_object : communication_objects)
      {
        if (eval_flags & EvaluationFlags::values)
          {
            std::visit(
              [&](const auto &obj) {
                copy_data(
                  dst.values,
                  VectorTools::point_values<n_components>(
                    *obj.rpe, mesh, src, vec_flags, first_selected_component),
                  view,
                  obj.get_pntrs());
              },
              communication_object);
          }

        if (eval_flags & EvaluationFlags::gradients)
          {
            std::visit(
              [&](const auto &obj) {
                copy_data(
                  dst.gradients,
                  VectorTools::point_gradients<n_components>(
                    *obj.rpe, mesh, src, vec_flags, first_selected_component),
                  view,
                  obj.get_pntrs());
              },
              communication_object);
          }

        Assert(!(eval_flags & EvaluationFlags::hessians), ExcNotImplemented());
      }

    if (has_ghost_elements == false)
      src.zero_out_ghost_values();
  }

  /**
   * Provide access to @c PrecomputedFEEvaluationDataView.
   */
  const internal::PrecomputedFEEvaluationDataView &get_view() const
  {
    return view;
  }

private:
  /**
   * CRS like data structure that describes the data positions at given
   * indices.
   */
  internal::PrecomputedFEEvaluationDataView view;

  /**
   * A variant for all possible communication objects.
   */
  std::vector<std::variant<FERemoteCommunicationObjectEntityBatches<dim>,
                           FERemoteCommunicationObjectCells<dim>,
                           FERemoteCommunicationObjectFaces<dim>>>
    communication_objects;

  // The rest of the class only deals with copying data
  /**
   * Copy data from @c src to @c dst. Overload for @c
   * FERemoteCommunicationObjectCells.
   */
  template <typename T1, typename T2>
  void copy_data(
    std::vector<T1>                                               &dst,
    const std::vector<T2>                                         &src,
    const std::vector<typename Triangulation<dim>::cell_iterator> &cells) const
  {
    dst.resize(view.size());

    unsigned int c = 0;
    for (const auto &cell : cells)
      {
        for (unsigned int j = view.get_shift(cell->active_cell_index());
             j < view.get_shift(cell->active_cell_index() + 1);
             ++j, ++c)
          {
            AssertIndexRange(j, dst.size());
            AssertIndexRange(c, src.size());
            dst[j] = src[c];
          }
      }
  }

  /**
   * Copy data from @c src to @c dst. Overload for @c
   * FERemoteCommunicationObjectFaces.
   */
  template <typename T1, typename T2>
  void copy_data(
    std::vector<T1>                                 &dst,
    const std::vector<T2>                           &src,
    const internal::PrecomputedFEEvaluationDataView &view,
    const std::vector<std::pair<typename Triangulation<dim>::cell_iterator,
                                unsigned int>>      &cell_face_nos) const
  {
    dst.resize(view.size());

    unsigned int c = 0;
    for (const auto &[cell, f] : cell_face_nos)
      {
        for (unsigned int j = view.get_shift(cell->active_cell_index(), f);
             j < view.get_shift(cell->active_cell_index(), f + 1);
             ++j, ++c)
          {
            AssertIndexRange(j, dst.size());
            AssertIndexRange(c, src.size());

            dst[j] = src[c];
          }
      }
  }

  /**
   * Copy data from @c src to @c dst. Overload for @c
   * FERemoteCommunicationObjectEntityBatches.
   */
  template <typename T1, typename T2>
  void copy_data(std::vector<T1>                                 &dst,
                 const std::vector<T2>                           &src,
                 const internal::PrecomputedFEEvaluationDataView &view,
                 const std::vector<std::pair<unsigned int, unsigned int>>
                   &batch_id_n_entities) const
  {
    dst.resize(view.size());

    unsigned int c = 0;
    for (const auto &[batch_id, n_entries] : batch_id_n_entities)
      {
        for (unsigned int v = 0; v < n_entries; ++v)
          for (unsigned int j = view.get_shift(batch_id);
               j < view.get_shift(batch_id + 1);
               ++j, ++c)
            {
              AssertIndexRange(j, dst.size());
              AssertIndexRange(c, src.size());

              copy_data_entries(dst[j], v, src[c]);
            }
      }
  }

  /**
   * Copy data to the correct position in a @c VectorizedArray.
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
   * Similar as @c copy_data_entries() above.
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
   * Similar as @c copy_data_entries() above.
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
   * Throw a run time exception if @c copy_data_entries() has not been
   * implemented for given types.
   */
  template <typename T1, typename T2>
  void copy_data_entries(T1 &, const unsigned int, const T2 &) const
  {
    Assert(false,
           ExcMessage(
             "copy_data_entries() not implemented for given arguments."));
  }
};

/**
 * Class to access data in matrix-free loops for non-matching discretizations.
 * Interfaces are named with FEEvaluation in mind.
 * The main difference is, that `gather_evaluate()` updates and caches
 * all values at once. Therefore, it has to be called on one thread before a
 * matrix-free loop.
 *
 * To access values and gradients in a thread safe way, @c get_data_accessor()
 * has to be called on every thread. It provides the functions `get_value()` and
 * `get_gradient()`.
 */
template <int dim, int n_components, typename value_type>
class FERemoteEvaluation
{
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
  FERemoteEvaluation(const FERemoteEvaluationCommunicator<dim> &comm,
                     const MeshType                            &mesh,
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
   * @c FERemoteEvaluation does not provide the functions `get_value()` and
   *`get_gradient()`. To access values and/or gradients call @c
   *get_data_accessor() on every thread. E.g. auto remote_evaluator =
   *get_data_accessor();
   *@code
   * for (unsigned int entity = range.first; entity < range.second; ++entity)
   * {
   *    remote_evaluator.reinit(entity);
   *    for(unsigned int q : quadrature_point_indices())
   *      remote_evaluator.get_value(q)
   * }
   *@endcode
   */
  internal::PrecomputedFEEvaluationDataAccessor<dim, n_components, value_type>
  get_data_accessor() const
  {
    internal::PrecomputedFEEvaluationDataAccessor data_accessor(
      data, comm->get_view());
    return data_accessor;
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
   * Precomputed values and/or gradients at remote locations.
   */
  internal::PrecomputedFEEvaluationData<dim, n_components, value_type> data;

  /**
   * Underlying communicator which handles update of the ghost values.
   */
  SmartPointer<const FERemoteEvaluationCommunicator<dim>> comm;

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

DEAL_II_NAMESPACE_CLOSE

#endif
