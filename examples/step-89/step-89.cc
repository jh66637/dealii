/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 *
 * Authors: Johannes Heinz, TU Wien, 2023
 *          Marco Feder, SISSA, 2023
 *          Peter Munch, University of Augsburg, 2023
 */

// @sect3{Include files}
//
// The program starts with including all the relevant header files.
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>


#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/non_matching/mapping_info.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>
// The following header file provides the class FERemoteEvaluation, which allows
// to access values and/or gradients at remote triangulations similar to
// FEEvaluation.
#include "fe_remote_evaluation.h"
// TODO: this file is not yet in deal.ii and will end up in
//  #include <deal.II/matrix_free/fe_remote_evaluation.h>

// TODO: clean up!!!
// TODO: inhomegenous!!!


// We pack everything that is specific for this program into a namespace
// of its own.
namespace Step89
{
  using namespace dealii;

  template <int dim>
  class InitialSolutionVibratingMembrane : public Function<dim>
  {
  public:
    InitialSolutionVibratingMembrane(const double modes)
      : Function<dim>(dim + 1, 0.0)
      , M(modes)
    {
      static_assert(dim == 2, "Only implemented for dim==2");
    }

    double value(const Point<dim> &p, const unsigned int comp) const final
    {
      if (comp == 0)
        return std::sin(M * numbers::PI * p[0]) *
               std::sin(M * numbers::PI * p[1]);

      return 0.0;
    }

  private:
    const double M;
  };

  template <int dim>
  class InitialSolutionInHomogenous : public Function<dim>
  {
  public:
    InitialSolutionInHomogenous()
      : Function<dim>(dim + 1, 0.0)
    {
      static_assert(dim == 2, "Only implemented for dim==2");
    }

    double value(const Point<dim> &p, const unsigned int comp) const final
    {
      if (comp == 0)
        return std::exp(-100.0 * (std::pow(p[0] + 0.5, 2)) +
                        (std::pow(p[1] + 0, 2)));

      return 0.0;
    }
  };

  // Free helper functions that are used in the tutorial.
  namespace HelperFunctions
  {
    // Helper function to check if a boundary ID is related to a non-matching
    // face. A @c std::set that contains all non-matching boundary IDs is
    // handed over additionaly to the face ID under question. This function
    // could certainly also be defined inline but this way the code is more easy
    // to read.
    bool is_non_matching_face(
      const std::set<types::boundary_id> &non_matching_face_ids,
      const types::boundary_id            face_id)
    {
      return non_matching_face_ids.find(face_id) != non_matching_face_ids.end();
    }

    // Helper function to set the initial conditions for the vibrating membrane
    //  test case.
    template <int dim,
              typename Number,

              typename VectorType>
    void set_initial_condition(MatrixFree<dim, Number> matrix_free,
                               const Function<dim>    &initial_solution,
                               VectorType             &dst)
    {
      VectorTools::interpolate(*matrix_free.get_mapping_info().mapping,
                               matrix_free.get_dof_handler(),
                               initial_solution,
                               dst);
    }

    // Helper function to compute the time step size according to the CFL
    // condition.
    double
    compute_dt_cfl(const double hmin, const unsigned int degree, const double c)
    {
      return hmin / (std::pow(degree, 1.5) * c);
    }

    template <typename VectorType, int dim>
    void write_vtu(const VectorType      &solution,
                   const DoFHandler<dim> &dof_handler,
                   const Mapping<dim>    &mapping,
                   const unsigned int     degree,
                   const std::string     &name_prefix)
    {
      DataOut<dim>          data_out;
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(
          dim + 1, DataComponentInterpretation::component_is_part_of_vector);
      std::vector<std::string> names(dim + 1, "U");

      interpretation[0] = DataComponentInterpretation::component_is_scalar;
      names[0]          = "P";

      data_out.add_data_vector(dof_handler, solution, names, interpretation);

      data_out.build_patches(mapping, degree, DataOut<dim>::curved_inner_cells);
      data_out.write_vtu_in_parallel(name_prefix + ".vtu",
                                     dof_handler.get_communicator());
    }
  } // namespace HelperFunctions


  template <int dim, typename Number>
  class MaterialHandler
  {
    using scalar = typename FEEvaluation<dim, -1, 0, 1, Number>::value_type;

  public:
    MaterialHandler(
      const MatrixFree<dim, Number, VectorizedArray<Number>> &matrix_free,
      const std::map<types::material_id, std::pair<double, double>>
        &material_id_map)
      : homogenous(material_id_map.size() == 1)
      , phi(matrix_free)
      , phi_face(matrix_free, true)
    {
      if (homogenous)
        {
          speed_of_sound = material_id_map.begin()->second.first;
          density        = material_id_map.begin()->second.second;
        }
      else
        {
          const auto n_cell_batches =
            matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();

          rho.resize(n_cell_batches);
          c.resize(n_cell_batches);

          for (unsigned int cell = 0; cell < n_cell_batches; ++cell)
            {
              c[cell]   = 1.;
              rho[cell] = 1.;
              for (unsigned int v = 0;
                   v < matrix_free.n_active_entries_per_cell_batch(cell);
                   ++v)
                {
                  const auto material_id =
                    matrix_free.get_cell_iterator(cell, v)->material_id();

                  c[cell][v]   = material_id_map.at(material_id).first;
                  rho[cell][v] = material_id_map.at(material_id).second;
                }
            }
        }
    }

    bool is_homogenous() const
    {
      return homogenous;
    }

    void reinit_cell(const unsigned int cell)
    {
      if (!homogenous)
        {
          phi.reinit(cell);
          speed_of_sound = phi.read_cell_data(c);
          density        = phi.read_cell_data(rho);
        }
    }

    void reinit_face(const unsigned int face)
    {
      if (!homogenous)
        {
          phi_face.reinit(face);
          speed_of_sound = phi.read_cell_data(c);
          density        = phi.read_cell_data(rho);
        }
    }

    std::pair<scalar, scalar> get_materials() const
    {
      return std::make_pair(speed_of_sound, density);
    }

    std::pair<Number, Number> get_materials(const unsigned int lane) const
    {
      return std::make_pair(speed_of_sound[lane], density[lane]);
    }


  private:
    const bool                              homogenous;
    FEEvaluation<dim, -1, 0, 1, Number>     phi;
    FEFaceEvaluation<dim, -1, 0, 1, Number> phi_face;
    AlignedVector<VectorizedArray<Number>>  c;
    AlignedVector<VectorizedArray<Number>>  rho;

    scalar speed_of_sound;
    scalar density;
  };

  template <int dim, typename Number, bool mortaring>
  class RemoteMaterialHandler
  {
  public:
    RemoteMaterialHandler(/*TODO*/)
    {
      // fill arrays that can be read by read_cell_data()
    }

    template <bool M = mortaring>
    typename std::enable_if_t<false == M, void>
    reinit_face(const unsigned int face)
    {
      phi_c.reinit(face);
      phi_rho.reinit(face);
    }

    template <bool M = mortaring>
    typename std::enable_if_t<true == M, void>
    reinit_face(const unsigned int cell, const unsigned int face)
    {
      phi_c.reinit(cell, face);
      phi_rho.reinit(cell, face);
    }

    auto get_materials(unsigned int q) const
    {
      return std::make_pair(phi_c.get_value(q), phi_rho.get_value(q));
    }

  private:
    FERemoteEvaluationBase<dim,
                           1,
                           Number,
                           VectorizedArray<Number>,
                           true,
                           !mortaring>
      phi_c;
    FERemoteEvaluationBase<dim,
                           1,
                           Number,
                           VectorizedArray<Number>,
                           true,
                           !mortaring>
      phi_rho;
  };


  // Class that defines the acoustic operator.
  template <int dim, typename Number>
  class AcousticOperator
  {
    // To be able to use the same kernel, for all face integrals we define
    //  a class that returns the needed values at boundaries. In this tutorial
    //  homogenous pressure Dirichlet boundary conditions are applied via
    //  the mirror priciple, i.e. $p_h^+=-p_h^- + 2g$ with $g=0$.
    class BCEvalP
    {
    public:
      BCEvalP(const FEFaceEvaluation<dim, -1, 0, 1, Number> &pressure_m)
        : pressure_m(pressure_m)
      {}

      typename FEFaceEvaluation<dim, -1, 0, 1, Number>::value_type
      get_value(const unsigned int q) const
      {
        return -pressure_m.get_value(q);
      }

    private:
      const FEFaceEvaluation<dim, -1, 0, 1, Number> &pressure_m;
    };

    // Similar as above. In this tutorial velocity Neumann boundary conditions
    // are applied.
    class BCEvalU
    {
    public:
      BCEvalU(const FEFaceEvaluation<dim, -1, 0, dim, Number> &velocity_m)
        : velocity_m(velocity_m)
      {}

      typename FEFaceEvaluation<dim, -1, 0, dim, Number>::value_type
      get_value(const unsigned int q) const
      {
        return velocity_m.get_value(q);
      }

    private:
      const FEFaceEvaluation<dim, -1, 0, dim, Number> &velocity_m;
    };


  public:
    // Constructor with all the needed ingredients for the operator.
    AcousticOperator(
      const MatrixFree<dim, Number>      &matrix_free_in,
      const std::set<types::boundary_id> &non_matching_face_ids,
      std::shared_ptr<FEFaceRemoteEvaluation<dim, 1, Number>>   pressure_r,
      std::shared_ptr<FEFaceRemoteEvaluation<dim, dim, Number>> velocity_r,
      std::shared_ptr<MaterialHandler<dim, Number>> material_handler,
      std::shared_ptr<RemoteMaterialHandler<dim, Number, false>>
        material_handler_remote)
      : use_mortaring(false)
      , matrix_free(matrix_free_in)
      , remote_face_ids(non_matching_face_ids)
      , pressure_r(pressure_r)
      , velocity_r(velocity_r)
      , nm_mapping_info(nullptr)
      , pressure_r_mortar(nullptr)
      , velocity_r_mortar(nullptr)
      , material_handler(material_handler)
      , material_handler_r(material_handler_remote)
      , material_handler_r_mortar(nullptr)
    {}

    // Constructor with all the needed ingredients for the operator.
    AcousticOperator(
      const MatrixFree<dim, Number>      &matrix_free_in,
      const std::set<types::boundary_id> &non_matching_face_ids,
      std::shared_ptr<NonMatching::MappingInfo<dim, dim, Number>>    nm_info,
      std::shared_ptr<FEFaceRemotePointEvaluation<dim, 1, Number>>   pressure_r,
      std::shared_ptr<FEFaceRemotePointEvaluation<dim, dim, Number>> velocity_r,
      std::shared_ptr<MaterialHandler<dim, Number>> material_handler,
      std::shared_ptr<RemoteMaterialHandler<dim, Number, true>>
        material_handler_remote)
      : use_mortaring(true)
      , matrix_free(matrix_free_in)
      , remote_face_ids(non_matching_face_ids)
      , pressure_r(nullptr)
      , velocity_r(nullptr)
      , nm_mapping_info(nm_info)
      , pressure_r_mortar(pressure_r)
      , velocity_r_mortar(velocity_r)
      , material_handler(material_handler)
      , material_handler_r(nullptr)
      , material_handler_r_mortar(material_handler_remote)
    {}

    // Function to evaluate the acoustic operator with Nitsche-type mortaring
    // at non-matching faces.
    template <typename VectorType>
    void evaluate(VectorType &dst, const VectorType &src) const
    {
      if (use_mortaring)
        {
          // Update the cached values in corresponding the RemoteEvaluation
          // objects.
          pressure_r_mortar->gather_evaluate(src, EvaluationFlags::values);
          velocity_r_mortar->gather_evaluate(src, EvaluationFlags::values);

          // Perform matrix free loop.
          matrix_free.loop(&AcousticOperator::cell_loop,
                           &AcousticOperator::face_loop,
                           &AcousticOperator::boundary_face_loop_mortaring,
                           this,
                           dst,
                           src,
                           true,
                           MatrixFree<dim, Number>::DataAccessOnFaces::values,
                           MatrixFree<dim, Number>::DataAccessOnFaces::values);
        }
      else
        {
          // Update the cached values in corresponding the RemoteEvaluation
          // objects.
          pressure_r->gather_evaluate(src, EvaluationFlags::values);
          velocity_r->gather_evaluate(src, EvaluationFlags::values);

          // Perform matrix free loop.
          matrix_free.loop(&AcousticOperator::cell_loop,
                           &AcousticOperator::face_loop,
                           &AcousticOperator::boundary_face_loop_point_to_point,
                           this,
                           dst,
                           src,
                           true,
                           MatrixFree<dim, Number>::DataAccessOnFaces::values,
                           MatrixFree<dim, Number>::DataAccessOnFaces::values);
        }
    }

  private:
    // This function evaluates the volume integrals.
    template <typename VectorType>
    void
    cell_loop(const MatrixFree<dim, Number>               &matrix_free,
              VectorType                                  &dst,
              const VectorType                            &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, 1, Number>   pressure(matrix_free, 0, 0, 0);
      FEEvaluation<dim, -1, 0, dim, Number> velocity(matrix_free, 0, 0, 1);

      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          velocity.reinit(cell);
          pressure.reinit(cell);

          pressure.gather_evaluate(src, EvaluationFlags::gradients);
          velocity.gather_evaluate(src, EvaluationFlags::gradients);

          material_handler->reinit_cell(cell);
          const auto [c, rho] = material_handler->get_materials();

          for (unsigned int q = 0; q < pressure.n_q_points; ++q)
            {
              pressure.submit_value(rho * c * c * velocity.get_divergence(q),
                                    q);
              velocity.submit_value(1.0 / rho * pressure.get_gradient(q), q);
            }

          pressure.integrate_scatter(EvaluationFlags::values, dst);
          velocity.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    // This function evaluates the fluxes at faces. If boundary faces are under
    // consideration fluxes into neighboring faces do not have to be considered
    // (there are none). For non-matching faces the fluxes into neighboring
    // faces are not considered as well. This is because we iterate over each
    // side of the non-matching face seperately (similar to a cell centric
    // loop).
    template <bool weight_neighbor,
              typename InternalFaceIntegratorPressure,
              typename InternalFaceIntegratorVelocity,
              typename ExternalFaceIntegratorPressure,
              typename ExternalFaceIntegratorVelocity>
    void evaluate_face_kernel(
      InternalFaceIntegratorPressure &pressure_m,
      InternalFaceIntegratorVelocity &velocity_m,
      ExternalFaceIntegratorPressure &pressure_p,
      ExternalFaceIntegratorVelocity &velocity_p,
      const std::pair<typename InternalFaceIntegratorPressure::value_type,
                      typename InternalFaceIntegratorPressure::value_type>
        &materials) const
    {
      const auto [c, rho] = materials;
      const auto tau      = 0.5 * rho * c;
      const auto gamma    = 0.5 / (rho * c);

      for (unsigned int q : pressure_m.quadrature_point_indices())
        {
          const auto n  = pressure_m.normal_vector(q);
          const auto pm = pressure_m.get_value(q);
          const auto um = velocity_m.get_value(q);

          const auto pp = pressure_p.get_value(q);
          const auto up = velocity_p.get_value(q);


          const auto flux_momentum =
            0.5 * (pm + pp) + 0.5 * tau * (um - up) * n;
          velocity_m.submit_value(1.0 / rho * (flux_momentum - pm) * n, q);
          if constexpr (weight_neighbor)
            velocity_p.submit_value(1.0 / rho * (flux_momentum - pp) * (-n), q);

          const auto flux_mass = 0.5 * (um + up) + 0.5 * gamma * (pm - pp) * n;
          pressure_m.submit_value(rho * c * c * (flux_mass - um) * n, q);
          if constexpr (weight_neighbor)
            pressure_p.submit_value(rho * c * c * (flux_mass - up) * (-n), q);
        }
    }

    template <bool mortaring,
              typename InternalFaceIntegratorPressure,
              typename InternalFaceIntegratorVelocity,
              typename ExternalFaceIntegratorPressure,
              typename ExternalFaceIntegratorVelocity>
    void evaluate_face_kernel(
      InternalFaceIntegratorPressure &pressure_m,
      InternalFaceIntegratorVelocity &velocity_m,
      ExternalFaceIntegratorPressure &pressure_p,
      ExternalFaceIntegratorVelocity &velocity_p,
      const std::pair<typename InternalFaceIntegratorPressure::value_type,
                      typename InternalFaceIntegratorPressure::value_type>
                                                          &materials,
      const RemoteMaterialHandler<dim, Number, mortaring> &material_handler_r)
      const
    {
      const auto [c, rho] = materials;
      const auto tau_m    = 0.5 * rho * c;
      const auto gamma_m  = 0.5 / (rho * c);

      for (unsigned int q : pressure_m.quadrature_point_indices())
        {
          const auto [c_p, rho_p]  = material_handler_r.get_materials(q);
          const auto tau_p         = 0.5 * rho_p * c_p;
          const auto gamma_p       = 0.5 / (rho_p * c_p);
          const auto tau_sum_inv   = 1.0 / (tau_m + tau_p);
          const auto gamma_sum_inv = 1.0 / (gamma_m + gamma_p);


          const auto n  = pressure_m.normal_vector(q);
          const auto pm = pressure_m.get_value(q);
          const auto um = velocity_m.get_value(q);

          const auto pp = pressure_p.get_value(q);
          const auto up = velocity_p.get_value(q);

          const auto flux_momentum =
            pm - tau_m * tau_sum_inv * (pm - pp) +
            tau_m * tau_p * tau_sum_inv * (um - up) * n;
          velocity_m.submit_value(1.0 / rho * (flux_momentum - pm) * n, q);


          const auto flux_mass =
            um - gamma_m * gamma_sum_inv * (um - up) +
            gamma_m * gamma_p * gamma_sum_inv * (pm - pp) * n;

          pressure_m.submit_value(rho * c * c * (flux_mass - um) * n, q);
        }
    }

    //  This function evaluates the inner face integrals.
    template <typename VectorType>
    void
    face_loop(const MatrixFree<dim, Number>               &matrix_free,
              VectorType                                  &dst,
              const VectorType                            &src,
              const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, 1, Number> pressure_m(
        matrix_free, true, 0, 0, 0);
      FEFaceEvaluation<dim, -1, 0, 1, Number> pressure_p(
        matrix_free, false, 0, 0, 0);
      FEFaceEvaluation<dim, -1, 0, dim, Number> velocity_m(
        matrix_free, true, 0, 0, 1);
      FEFaceEvaluation<dim, -1, 0, dim, Number> velocity_p(
        matrix_free, false, 0, 0, 1);

      for (unsigned int face = face_range.first; face < face_range.second;
           face++)
        {
          velocity_m.reinit(face);
          velocity_p.reinit(face);

          pressure_m.reinit(face);
          pressure_p.reinit(face);

          pressure_m.gather_evaluate(src, EvaluationFlags::values);
          pressure_p.gather_evaluate(src, EvaluationFlags::values);

          velocity_m.gather_evaluate(src, EvaluationFlags::values);
          velocity_p.gather_evaluate(src, EvaluationFlags::values);

          material_handler->reinit_face(face);
          evaluate_face_kernel<true>(pressure_m,
                                     velocity_m,
                                     pressure_p,
                                     velocity_p,
                                     material_handler->get_materials());

          pressure_m.integrate_scatter(EvaluationFlags::values, dst);
          pressure_p.integrate_scatter(EvaluationFlags::values, dst);
          velocity_m.integrate_scatter(EvaluationFlags::values, dst);
          velocity_p.integrate_scatter(EvaluationFlags::values, dst);
        }
    }


    // This function evaluates the boundary face integrals and the
    // non-matching face integrals using point-to-point interpolation.
    template <typename VectorType>
    void boundary_face_loop_point_to_point(
      const MatrixFree<dim, Number>               &matrix_free,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      // Standard face evaluators.
      FEFaceEvaluation<dim, -1, 0, 1, Number> pressure_m(
        matrix_free, true, 0, 0, 0);
      FEFaceEvaluation<dim, -1, 0, dim, Number> velocity_m(
        matrix_free, true, 0, 0, 1);

      // Classes which return the correct BC values.
      BCEvalP pressure_bc(pressure_m);
      BCEvalU velocity_bc(velocity_m);

      for (unsigned int face = face_range.first; face < face_range.second;
           face++)
        {
          velocity_m.reinit(face);
          pressure_m.reinit(face);

          pressure_m.gather_evaluate(src, EvaluationFlags::values);
          velocity_m.gather_evaluate(src, EvaluationFlags::values);

          if (!HelperFunctions::is_non_matching_face(
                remote_face_ids, matrix_free.get_boundary_id(face)))
            {
              // If @c face is a standard boundary face, evaluate the integral
              // as
              // usual in the matrix free context. To be able to use the same
              // kernel as for inner faces we pass the boundary condition
              // objects to the function that evaluates the kernel. As mentioned
              // above, there is no neighbor to consider in the kernel.
              material_handler->reinit_face(face);
              evaluate_face_kernel<false>(pressure_m,
                                          velocity_m,
                                          pressure_bc,
                                          velocity_bc,
                                          material_handler->get_materials());
            }
          else
            {
              // If @c face is nonmatching we have to query values via the
              // RemoteEvaluaton objects. This is done by passing the
              // corresponding RemoteEvaluaton objects to the function that
              // evaluates the kernel. As mentioned above, each side of the
              // non-matching interface is iterated seperately and we do not
              // have to consider the neighbor in the kernel. Note, that the
              // values in the RemoteEvaluaton objects are already updated at
              // this point.

              // For point-to-point interpolation we simply use the
              // corresponding RemoteEvaluaton objects in combination with the
              // standard FEFaceEvaluation objects.
              velocity_r->reinit(face);
              pressure_r->reinit(face);

              material_handler->reinit_face(face);

              if (material_handler->is_homogenous())
                {
                  evaluate_face_kernel<false>(
                    pressure_m,
                    velocity_m,
                    *pressure_r,
                    *velocity_r,
                    material_handler->get_materials());
                }
              else
                {
                  material_handler_r->reinit_face(face);
                  evaluate_face_kernel<false>(pressure_m,
                                              velocity_m,
                                              *pressure_r,
                                              *velocity_r,
                                              material_handler->get_materials(),
                                              *material_handler_r);
                }
            }

          pressure_m.integrate_scatter(EvaluationFlags::values, dst);
          velocity_m.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    // This function evaluates the boundary face integrals and the
    // non-matching face integrals using Nitsche-type mortaring.
    template <typename VectorType>
    void boundary_face_loop_mortaring(
      const MatrixFree<dim, Number>               &matrix_free,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      // Standard face evaluators for BCs.
      FEFaceEvaluation<dim, -1, 0, 1, Number> pressure_m(
        matrix_free, true, 0, 0, 0);
      FEFaceEvaluation<dim, -1, 0, dim, Number> velocity_m(
        matrix_free, true, 0, 0, 1);

      // Classes which return the correct BC values.
      BCEvalP pressure_bc(pressure_m);
      BCEvalU velocity_bc(velocity_m);

      // For Nitsche-type mortaring we are evaluating the integrals over
      // intersections. This is why, quadrature points are arbitrarely
      // distributed on every face. Thus, we can not make use of face batches
      // and FEFaceEvaluation but have to consider each face individually and
      // make use of @c FEPointEvaluation to evaluate the integrals in the
      // arbitrarely distributed quadrature points.
      FEPointEvaluation<1, dim, dim, Number> pressure_m_mortar(
        *nm_mapping_info, matrix_free.get_dof_handler().get_fe(), 0);
      FEPointEvaluation<dim, dim, dim, Number> velocity_m_mortar(
        *nm_mapping_info, matrix_free.get_dof_handler().get_fe(), 1);

      // Buffer on which FEPointEvaluation is working on.
      std::vector<Number> buffer(
        matrix_free.get_dof_handler().get_fe().dofs_per_cell);

      for (unsigned int face = face_range.first; face < face_range.second;
           face++)
        {
          if (!HelperFunctions::is_non_matching_face(
                remote_face_ids, matrix_free.get_boundary_id(face)))
            {
              // Same as in @c boundary_face_loop_point_to_point().
              velocity_m.reinit(face);
              pressure_m.reinit(face);

              pressure_m.gather_evaluate(src, EvaluationFlags::values);
              velocity_m.gather_evaluate(src, EvaluationFlags::values);

              material_handler->reinit_face(face);
              evaluate_face_kernel<false>(pressure_m,
                                          velocity_m,
                                          pressure_bc,
                                          velocity_bc,
                                          material_handler->get_materials());

              pressure_m.integrate_scatter(EvaluationFlags::values, dst);
              velocity_m.integrate_scatter(EvaluationFlags::values, dst);
            }
          else
            {
              // For mortaring we have to cosider every face from the face
              // batches seperately and have to use the FEPointEvaluation
              // objects to be able to evaluate the integrals with the
              // arbitrarily distributed quadrature points.
              for (unsigned int v = 0;
                   v < matrix_free.n_active_entries_per_face_batch(face);
                   ++v)
                {
                  const auto [cell, f] =
                    matrix_free.get_face_iterator(face, v, true);

                  velocity_m_mortar.reinit(cell->active_cell_index(), f);
                  pressure_m_mortar.reinit(cell->active_cell_index(), f);

                  cell->get_dof_values(src, buffer.begin(), buffer.end());
                  velocity_m_mortar.evaluate(buffer, EvaluationFlags::values);
                  pressure_m_mortar.evaluate(buffer, EvaluationFlags::values);

                  velocity_r_mortar->reinit(cell->active_cell_index(), f);
                  pressure_r_mortar->reinit(cell->active_cell_index(), f);

                  material_handler->reinit_face(face);

                  if (material_handler->is_homogenous())
                    {
                      evaluate_face_kernel<false>(
                        pressure_m_mortar,
                        velocity_m_mortar,
                        *pressure_r_mortar,
                        *velocity_r_mortar,
                        material_handler->get_materials(v));
                    }
                  else
                    {
                      material_handler_r_mortar->reinit_face(
                        cell->active_cell_index(), f);
                      evaluate_face_kernel<true>(
                        pressure_m_mortar,
                        velocity_m_mortar,
                        *pressure_r_mortar,
                        *velocity_r_mortar,
                        material_handler->get_materials(v),
                        *material_handler_r_mortar);
                    }

                  // First zero out buffer via sum_into_values=false
                  velocity_m_mortar.integrate(buffer,
                                              EvaluationFlags::values,
                                              /*sum_into_values=*/false);
                  // Don't zero out values again to keep already integrated
                  // values
                  pressure_m_mortar.integrate(buffer,
                                              EvaluationFlags::values,
                                              /*sum_into_values=*/true);

                  cell->distribute_local_to_global(buffer.begin(),
                                                   buffer.end(),
                                                   dst);
                }
            }
        }
    }

    // Members, needed to evaluate the acoustic operator.
    const bool use_mortaring;

    const MatrixFree<dim, Number> &matrix_free;

    // FERemoteEvaluation objects are strored as shared pointers. This way,
    // they can also be used for other operators without caching the values
    // multiple times.
    const std::set<types::boundary_id> remote_face_ids;
    const std::shared_ptr<FEFaceRemoteEvaluation<dim, 1, Number>>   pressure_r;
    const std::shared_ptr<FEFaceRemoteEvaluation<dim, dim, Number>> velocity_r;
    const std::shared_ptr<NonMatching::MappingInfo<dim, dim, Number>>
      nm_mapping_info;
    const std::shared_ptr<FEFaceRemotePointEvaluation<dim, 1, Number>>
      pressure_r_mortar;
    const std::shared_ptr<FEFaceRemotePointEvaluation<dim, dim, Number>>
      velocity_r_mortar;

    const std::shared_ptr<MaterialHandler<dim, Number>> material_handler;
    const std::shared_ptr<RemoteMaterialHandler<dim, Number, false>>
      material_handler_r;
    const std::shared_ptr<RemoteMaterialHandler<dim, Number, true>>
      material_handler_r_mortar;
  };

  // Class to apply the inverse mass operator.
  template <int dim, typename Number>
  class InverseMassOperator
  {
  public:
    // Constructor.
    InverseMassOperator(const MatrixFree<dim, Number> &matrix_free)
      : matrix_free(matrix_free)
    {}

    // Function to apply the inverse mass operator.
    template <typename VectorType>
    void apply(VectorType &dst, const VectorType &src) const
    {
      dst.zero_out_ghost_values();
      matrix_free.cell_loop(&InverseMassOperator::cell_loop, this, dst, src);
    }

  private:
    // Apply the inverse mass operator onto every cell batch.
    template <typename VectorType>
    void
    cell_loop(const MatrixFree<dim, Number>               &mf,
              VectorType                                  &dst,
              const VectorType                            &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, dim + 1, Number> phi(mf);
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 1, Number>
        minv(phi);

      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(src);
          minv.apply(phi.begin_dof_values(), phi.begin_dof_values());
          phi.set_dof_values(dst);
        }
    }

    const MatrixFree<dim, Number> &matrix_free;
  };

  // This class implements a Runge-Kutta scheme of order 2.
  template <int dim, typename Number>
  class RungeKutta2
  {
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

  public:
    // Constructor.
    RungeKutta2(
      const std::shared_ptr<InverseMassOperator<dim, Number>>
        inverse_mass_operator,
      const std::shared_ptr<AcousticOperator<dim, Number>> acoustic_operator)
      : inverse_mass_operator(inverse_mass_operator)
      , acoustic_operator(acoustic_operator)
    {}



    // Setup and run time loop.
    void run(const MatrixFree<dim, Number> &matrix_free,
             const double                   cr,
             const Function<dim>           &initial_condition)
    {
      // Get needed members of matrix free.
      const auto &dof_handler = matrix_free.get_dof_handler();
      const auto &mapping     = *matrix_free.get_mapping_info().mapping;
      const auto  degree      = dof_handler.get_fe().degree;

      // Initialize needed Vectors...
      VectorType solution;
      matrix_free.initialize_dof_vector(solution);
      VectorType solution_temp;
      matrix_free.initialize_dof_vector(solution_temp);

      // and set the initial condition.
      HelperFunctions::set_initial_condition(matrix_free,
                                             initial_condition,
                                             solution);

      // Compute time step size:

      // Compute minimum element edge length. We assume non-distorted
      // elements, therefore we only compute the distance between two vertices
      double h_local_min = std::numeric_limits<double>::max();
      for (const auto &cell : dof_handler.active_cell_iterators())
        h_local_min =
          std::min(h_local_min,
                   (cell->vertex(1) - cell->vertex(0)).norm_square());
      h_local_min = std::sqrt(h_local_min);
      const double h_min =
        Utilities::MPI::min(h_local_min, dof_handler.get_communicator());

      // Compute constant time step size via the CFL consition.
      const double dt =
        cr * HelperFunctions::compute_dt_cfl(h_min, degree, 1.0 /*TODO*/);

      // Compute end time for exactly one period duration.
      const double end_time =
        2.0 / (10.0 /*TODO*/ * std::sqrt(dim) * 1.0 /*TODO*/);

      // Perform time integration loop.
      double       time     = 0.0;
      unsigned int timestep = 0;
      while (time < end_time)
        {
          // Write ouput.
          HelperFunctions::write_vtu(solution,
                                     matrix_free.get_dof_handler(),
                                     mapping,
                                     degree,
                                     "step_89-" + std::to_string(timestep));

          // Perform a single time step.
          std::swap(solution, solution_temp);
          time += dt;
          timestep++;
          perform_time_step(dt, solution, solution_temp);
        }
    }

  private:
    // Perform one Runge-Kutta 2 time step.
    void
    perform_time_step(const double dt, VectorType &dst, const VectorType &src)
    {
      VectorType k1 = src;

      // stage 1
      evaluate_stage(k1, src);

      // stage 2
      k1.sadd(0.5 * dt, 1.0, src);
      evaluate_stage(dst, k1);
      dst.sadd(dt, 1.0, src);
    }

    // Evaluate a single Runge-Kutta stage.
    void evaluate_stage(VectorType &dst, const VectorType &src)
    {
      // Evaluate the stage
      acoustic_operator->evaluate(dst, src);
      dst *= -1.0;
      inverse_mass_operator->apply(dst, dst);
    }

    // Needed operators.
    const std::shared_ptr<InverseMassOperator<dim, Number>>
                                                         inverse_mass_operator;
    const std::shared_ptr<AcousticOperator<dim, Number>> acoustic_operator;
  };


  // @sect3{Construct a mesh with non-matching interfaces}
  template <int dim>
  void build_non_matching_triangulation(
    parallel::distributed::Triangulation<dim> &tria,
    std::set<types::boundary_id>              &non_matching_faces,
    const unsigned int                         refinements)
  {
    const double length = 1.0;

    const types::boundary_id non_matching_id_left  = 98;
    const types::boundary_id non_matching_id_right = 99;

    non_matching_faces.insert(non_matching_id_left);
    non_matching_faces.insert(non_matching_id_right);

    // left part of mesh
    Triangulation<dim> tria_left;
    const unsigned int subdiv_left = 1;
    GridGenerator::subdivided_hyper_rectangle(tria_left,
                                              {subdiv_left, 2 * subdiv_left},
                                              {0.0, 0.0},
                                              {0.5 * length, length});

    for (const auto &cell : tria_left.active_cell_iterators())
      cell->set_material_id(0);

    for (const auto &face : tria_left.active_face_iterators())
      if (face->at_boundary())
        {
          face->set_boundary_id(0);
          if (face->center()[0] > 0.5 * length - 1e-6)
            face->set_boundary_id(non_matching_id_left);
        }

    // right part of mesh
    Triangulation<dim> tria_right;
    const unsigned int subdiv_right = 3;
    GridGenerator::subdivided_hyper_rectangle(tria_right,
                                              {subdiv_right, 2 * subdiv_right},
                                              {0.5 * length, 0.0},
                                              {length, length});

    for (const auto &cell : tria_left.active_cell_iterators())
      cell->set_material_id(1);

    for (const auto &face : tria_right.active_face_iterators())
      if (face->at_boundary())
        {
          face->set_boundary_id(0);
          if (face->center()[0] < 0.5 * length + 1e-6)
            face->set_boundary_id(non_matching_id_right);
        }

    // merge triangulations with tolerance 0 to ensure no vertices
    //  are merged.
    GridGenerator::merge_triangulations(
      tria_left, tria_right, tria, /*tolerance*/ 0., false, true);
    tria.refine_global(refinements);
  }

  // @sect3{Point-to-point interpolation}
  template <int dim, typename Number>
  void point_to_point_interpolation(
    const MatrixFree<dim, Number>      &matrix_free,
    const std::set<types::boundary_id> &non_matching_faces,
    const std::map<types::material_id, std::pair<double, double>> &materials,
    const Function<dim> &initial_condition)
  {
    const auto &dof_handler = matrix_free.get_dof_handler();
    const auto &tria        = dof_handler.get_triangulation();
    const auto &mapping     = *matrix_free.get_mapping_info().mapping;

    const QGauss<dim - 1> face_quad(matrix_free.get_quadrature().size());


    // TODO: check if everything is correct(that means if the instable
    // result is expected in this configuration)!

    std::vector<
      std::pair<std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>>,
                std::vector<std::pair<unsigned int, unsigned int>>>>
      comm_objects;

    const auto face_batch_range =
      std::make_pair(matrix_free.n_inner_face_batches(),
                     matrix_free.n_inner_face_batches() +
                       matrix_free.n_boundary_face_batches());

    std::vector<Quadrature<dim>> global_quadrature_vector(
      face_batch_range.second - face_batch_range.first);

    for (const auto &nm_face : non_matching_faces)
      {
        FEFaceValues<dim> phi(mapping,
                              dof_handler.get_fe(),
                              face_quad,
                              update_quadrature_points);

        std::vector<std::pair<unsigned int, unsigned int>> face_lane;
        std::vector<Point<dim>>                            points;


        for (unsigned int bface = 0;
             bface < face_batch_range.second - face_batch_range.first;
             ++bface)
          {
            const unsigned int face = face_batch_range.first + bface;
            if (matrix_free.get_boundary_id(face) == nm_face)
              {
                const unsigned int n_lanes =
                  matrix_free.n_active_entries_per_face_batch(face);
                face_lane.push_back(std::make_pair(face, n_lanes));
                for (unsigned int v = 0; v < n_lanes; ++v)
                  {
                    const auto [cell, f] =
                      matrix_free.get_face_iterator(face, v, true);
                    phi.reinit(cell, f);
                    for (unsigned int q = 0; q < phi.n_quadrature_points; ++q)
                      {
                        points.push_back(phi.quadrature_point(q));
                      }
                  }

                Assert(global_quadrature_vector[bface].size() == 0,
                       ExcMessage(
                         "Quadrature for given face already provided."));

                global_quadrature_vector[bface] =
                  Quadrature<dim>(phi.get_quadrature_points());
              }
          }

        auto rpe = std::make_shared<Utilities::MPI::RemotePointEvaluation<dim>>(
          1.0e-9, false, 0, [&]() {
            // only search points at cells that are not connected to
            // nm_face
            std::vector<bool> mask(tria.n_vertices(), false);

            for (const auto &face : tria.active_face_iterators())
              if (face->at_boundary() && face->boundary_id() != nm_face)
                for (const auto v : face->vertex_indices())
                  mask[face->vertex_index(v)] = true;

            return mask;
          });

        rpe->reinit(points, tria, mapping);
        Assert(rpe->all_points_found(),
               ExcMessage("Not all remote points found."));

        comm_objects.push_back(std::make_pair(rpe, face_lane));
      }

    FERemoteEvaluationCommunicator<dim, true, true> remote_communicator;
    remote_communicator.reinit_faces(comm_objects,
                                     face_batch_range,
                                     global_quadrature_vector);

    const auto pressure_r =
      std::make_shared<FEFaceRemoteEvaluation<dim, 1, Number>>(
        remote_communicator, dof_handler, 0);
    const auto velocity_r =
      std::make_shared<FEFaceRemoteEvaluation<dim, dim, Number>>(
        remote_communicator, dof_handler, 1);

    const auto inverse_mass_operator =
      std::make_shared<InverseMassOperator<dim, Number>>(matrix_free);

    const auto material_handler =
      std::make_shared<MaterialHandler<dim, Number>>(matrix_free, materials);

    const auto acoustic_operator =
      std::make_shared<AcousticOperator<dim, Number>>(
        matrix_free,
        non_matching_faces,
        pressure_r,
        velocity_r,
        material_handler,
        nullptr); // TODO:remotematerialhandler

    RungeKutta2<dim, Number> time_integrator(inverse_mass_operator,
                                             acoustic_operator);
    time_integrator.run(matrix_free, 0.1, initial_condition);
  }

  // TODO: clean up
  // // @sect3{Nitsche-type mortaring}
  // //
  // // Description
  template <int dim, typename Number>
  void nitsche_type_mortaring(
    const MatrixFree<dim, Number>      &matrix_free,
    const std::set<types::boundary_id> &non_matching_faces,
    const std::map<types::material_id, std::pair<double, double>> &materials,
    const Function<dim> &initial_condition)
  {
    const auto &dof_handler       = matrix_free.get_dof_handler();
    const auto &tria              = dof_handler.get_triangulation();
    const auto &mapping           = *matrix_free.get_mapping_info().mapping;
    const auto  n_quadrature_pnts = matrix_free.get_quadrature().size();

    std::vector<std::vector<Quadrature<dim - 1>>> global_quadrature_vector;
    for (const auto &cell : tria.active_cell_iterators())
      global_quadrature_vector.emplace_back(
        std::vector<Quadrature<dim - 1>>(cell->n_faces()));

    std::vector<std::pair<
      std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>>,
      std::vector<
        std::pair<typename Triangulation<dim>::cell_iterator, unsigned int>>>>
      comm_objects;

    for (const auto &nm_face : non_matching_faces)
      {
        // 1) compute cell face pairs
        std::vector<
          std::pair<typename Triangulation<dim>::cell_iterator, unsigned int>>
          cell_face_pairs;

        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            for (unsigned int f = 0; f < cell->n_faces(); ++f)
              if (cell->face(f)->at_boundary() &&
                  cell->face(f)->boundary_id() == nm_face)
                cell_face_pairs.emplace_back(std::make_pair(cell, f));

        // 2) create RPE
        // create bounding boxes to search in
        std::vector<BoundingBox<dim>> local_boxes;
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            local_boxes.emplace_back(mapping.get_bounding_box(cell));

        // create r-tree of bounding boxes
        const auto local_tree = pack_rtree(local_boxes);

        // compress r-tree to a minimal set of bounding boxes
        std::vector<std::vector<BoundingBox<dim>>> global_bboxes(1);
        global_bboxes[0] = extract_rtree_level(local_tree, 0);

        const GridTools::Cache<dim, dim> cache(tria, mapping);

        // build intersection requests. Intersection requests
        // correspond to vertices at faces.
        std::vector<std::vector<Point<dim>>> intersection_requests;
        for (const auto &[cell, f] : cell_face_pairs)
          {
            std::vector<Point<dim>> vertices(cell->face(f)->n_vertices());
            std::copy_n(mapping.get_vertices(cell, f).begin(),
                        cell->face(f)->n_vertices(),
                        vertices.begin());
            intersection_requests.emplace_back(vertices);
          }

        // compute intersection data
        auto intersection_data =
          GridTools::internal::distributed_compute_intersection_locations<dim -
                                                                          1>(
            cache,
            intersection_requests,
            global_bboxes,
            [&]() {
              std::vector<bool> mask(tria.n_vertices(), false);

              for (const auto &face : tria.active_face_iterators())
                if (face->at_boundary() && face->boundary_id() != nm_face)
                  for (const auto v : face->vertex_indices())
                    mask[face->vertex_index(v)] = true;

              return mask;
            }(),
            1.0e-9);

        // convert to rpe
        auto rpe =
          std::make_shared<Utilities::MPI::RemotePointEvaluation<dim>>();
        rpe->reinit(
          intersection_data
            .template convert_to_distributed_compute_point_locations_internal<
              dim>(n_quadrature_pnts, tria, mapping),
          tria,
          mapping);

        // TODO: Most of the following is currently done twice in
        //  convert_to_distributed_compute_point_locations_internal.
        //  We have to adapt
        //  convert_to_distributed_compute_point_locations_internal to be
        //  able to retrieve relevant information.

        // TODO: NonMatchingMappingInfo should be able to work with
        //  Quadrature<dim> instead <dim-1>. Currently we are constructing
        //  dim-1 from dim and inside MappingInfo it is converted back.

        // 3) fill quadrature vector.
        for (unsigned int i = 0; i < intersection_requests.size(); ++i)
          {
            const auto &[cell, f] = cell_face_pairs[i];

            const unsigned int begin = intersection_data.recv_ptrs[i];
            const unsigned int end   = intersection_data.recv_ptrs[i + 1];

            std::vector<
              typename GridTools::internal::
                DistributedComputeIntersectionLocationsInternal<dim - 1, dim>::
                  IntersectionType>
              found_intersections(end - begin);

            unsigned int c = 0;
            for (unsigned int ptr = begin; ptr < end; ++ptr, ++c)
              found_intersections[c] =
                std::get<2>(intersection_data.recv_components[ptr]);

            const auto quad = QGaussSimplex<dim - 1>(n_quadrature_pnts)
                                .mapped_quadrature(found_intersections);

            std::vector<Point<dim - 1>> face_points(quad.size());
            for (uint q = 0; q < quad.size(); ++q)
              {
                face_points[q] =
                  mapping.project_real_point_to_unit_point_on_face(
                    cell, f, quad.point(q));
              }

            Assert(global_quadrature_vector[cell->active_cell_index()][f]
                       .size() == 0,
                   ExcMessage("Quadrature for given face already provided."));

            global_quadrature_vector[cell->active_cell_index()][f] =
              Quadrature<dim - 1>(face_points, quad.get_weights());
          }
        comm_objects.push_back(std::make_pair(rpe, cell_face_pairs));
      }

    FERemoteEvaluationCommunicator<dim, true, false> remote_communicator;
    remote_communicator.reinit_faces(
      comm_objects,
      matrix_free.get_dof_handler().get_triangulation().active_cell_iterators(),
      global_quadrature_vector);

    typename NonMatching::MappingInfo<dim, dim, Number>::AdditionalData
      additional_data;
    additional_data.use_global_weights = true;
    auto nm_mapping_info =
      std::make_shared<NonMatching::MappingInfo<dim, dim, Number>>(
        mapping,
        update_values | update_JxW_values | update_normal_vectors |
          update_quadrature_points,
        additional_data);
    nm_mapping_info->reinit_faces(
      matrix_free.get_dof_handler().get_triangulation().active_cell_iterators(),
      global_quadrature_vector);



    const auto pressure_r =
      std::make_shared<FEFaceRemotePointEvaluation<dim, 1, Number>>(
        remote_communicator, dof_handler, 0);
    const auto velocity_r =
      std::make_shared<FEFaceRemotePointEvaluation<dim, dim, Number>>(
        remote_communicator, dof_handler, 1);

    const auto inverse_mass_operator =
      std::make_shared<InverseMassOperator<dim, Number>>(matrix_free);

    const auto material_handler =
      std::make_shared<MaterialHandler<dim, Number>>(matrix_free, materials);

    const auto acoustic_operator =
      std::make_shared<AcousticOperator<dim, Number>>(
        matrix_free,
        non_matching_faces,
        nm_mapping_info,
        pressure_r,
        velocity_r,
        material_handler,
        nullptr); // TODO:remotematerialhandler

    RungeKutta2<dim, Number> time_integrator(inverse_mass_operator,
                                             acoustic_operator);
    time_integrator.run(matrix_free, 0.1, initial_condition);
  }



} // namespace Step89


// @sect3{Driver}
//
// Finally, the driver executes the different versions of handling non-matching
// interfaces.



int main(int argc, char *argv[])
{
  using namespace dealii;
  constexpr int dim = 2;
  using Number      = double;

  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  std::cout.precision(5);

  // Homogenous pressure DBCs are applied for simplicity. Therefore,
  // modes can not be chosen arbitrarily.
  const double                                            modes = 10.0;
  std::map<types::material_id, std::pair<double, double>> homogenous_material;
  homogenous_material[numbers::invalid_material_id] = std::make_pair(1.0, 1.0);

  const unsigned int refinements = 2;
  const unsigned int degree      = 3;

  // Construct triangulation and fill non-matching boundary IDs.
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  std::set<types::boundary_id>              non_matching_faces;
  Step89::build_non_matching_triangulation(tria,
                                           non_matching_faces,
                                           refinements);

  // Setup MatrixFree.
  MatrixFree<dim, Number> matrix_free;

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FESystem<dim>(FE_DGQ<dim>(degree), dim + 1));

  AffineConstraints<Number> constraints;
  constraints.close();

  typename MatrixFree<dim, Number>::AdditionalData data;
  data.mapping_update_flags = update_gradients | update_values;
  data.mapping_update_flags_inner_faces =
    update_quadrature_points | update_values;
  data.mapping_update_flags_boundary_faces =
    data.mapping_update_flags_inner_faces;

  matrix_free.reinit(
    MappingQ1<dim>(), dof_handler, constraints, QGauss<dim>(degree + 1), data);


  // Run vibrating membrane testcase using point-to-point interpolation:
  Step89::point_to_point_interpolation(
    matrix_free,
    non_matching_faces,
    homogenous_material,
    Step89::InitialSolutionVibratingMembrane<dim>(modes));
  
  // Run vibrating membrane testcase using Nitsche-type mortaring:
  Step89::nitsche_type_mortaring(matrix_free,
                                 non_matching_faces,
                                 homogenous_material,
                                 Step89::InitialSolutionVibratingMembrane<dim>(
                                   modes));

  // TODO:
  //  Run simple testcase with in-homogenous material:
  //  Step89::inhomogenous_material();

  return 0;
}
