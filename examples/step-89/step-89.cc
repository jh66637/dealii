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
#include <deal.II/matrix_free/fe_remote_evaluation.h>

// We pack everything that is specific for this program into a namespace
// of its own.
namespace Step89
{
  using namespace dealii;

  // @sect3{Initial conditions for vibrating membrane}
  //
  // Function that provides the initial condition for the vibrating membrane
  // testcase.
  template <int dim>
  class InitialConditionVibratingMembrane : public Function<dim>
  {
  public:
    InitialConditionVibratingMembrane(const double modes)
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

    double get_period_duration(const double speed_of_sound) const
    {
      return 2.0 / (M * std::sqrt(dim) * speed_of_sound);
    }

  private:
    const double M;
  };

  // @sect3{Gauss pulse}
  //
  // Function that provides the values of a pressure Gauss pulse.
  template <int dim>
  class GaussPulse : public Function<dim>
  {
  public:
    GaussPulse(const double shift_x, const double shift_y)
      : Function<dim>(dim + 1, 0.0)
      , shift_x(shift_x)
      , shift_y(shift_y)
    {
      static_assert(dim == 2, "Only implemented for dim==2");
    }

    double value(const Point<dim> &p, const unsigned int comp) const final
    {
      if (comp == 0)
        return std::exp(-1000.0 * ((std::pow(p[0] - shift_x, 2)) +
                                   (std::pow(p[1] - shift_y, 2))));

      return 0.0;
    }

  private:
    const double shift_x;
    const double shift_y;
  };

  // @sect3{Helper functions}
  //
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
    // test case.
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

    // Helper function that writes vtu output.
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

  //@sect3{Material access}
  //
  // This class stores the information if the fluid is homogenous
  // as well as the material properties at every cell.
  // This class helps to access the correct values without accessing
  // a large vector of materials in the homogenous case.
  template <typename Number>
  class CellwiseMaterialData
  {
  public:
    template <int dim>
    CellwiseMaterialData(
      const MatrixFree<dim, Number, VectorizedArray<Number>> &matrix_free,
      const std::map<types::material_id, std::pair<double, double>>
        &material_id_map)
      // If the map is of size 1, the material is constant in every cell.
      : homogenous(material_id_map.size() == 1)
    {
      Assert(material_id_map.size() > 0,
             ExcMessage("No materials given to CellwiseMaterialData"));

      if (homogenous)
        {
          // In the homogenous case we know the materials in the whole domain.
          speed_of_sound_homogenous = material_id_map.begin()->second.first;
          density_homogenous        = material_id_map.begin()->second.second;
        }
      else
        {
          // In the in-homogenous case materials vary between cells. We are
          // filling a vector with the correct materials, that can be processed
          // via
          // @c read_cell_data().
          const auto n_cell_batches =
            matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();

          speed_of_sound.resize(n_cell_batches);
          density.resize(n_cell_batches);

          for (unsigned int cell = 0; cell < n_cell_batches; ++cell)
            {
              speed_of_sound[cell] = 1.;
              density[cell]        = 1.;
              for (unsigned int v = 0;
                   v < matrix_free.n_active_entries_per_cell_batch(cell);
                   ++v)
                {
                  const auto material_id =
                    matrix_free.get_cell_iterator(cell, v)->material_id();

                  speed_of_sound[cell][v] =
                    material_id_map.at(material_id).first;
                  density[cell][v] = material_id_map.at(material_id).second;
                }
            }
        }
    }

    bool is_homogenous() const
    {
      return homogenous;
    }

    AlignedVector<VectorizedArray<Number>> get_speed_of_sound() const
    {
      Assert(!homogenous, ExcMessage("Use get_homogenous_speed_of_sound()"));
      return speed_of_sound;
    }

    AlignedVector<VectorizedArray<Number>> get_density() const
    {
      Assert(!homogenous, ExcMessage("Use get_homogenous_density()"));
      return density;
    }

    VectorizedArray<Number> get_homogenous_speed_of_sound() const
    {
      Assert(homogenous, ExcMessage("Use get_speed_of_sound()"));
      return speed_of_sound_homogenous;
    }

    VectorizedArray<Number> get_homogenous_density() const
    {
      Assert(homogenous, ExcMessage("Use get_density()"));
      return density_homogenous;
    }

  private:
    const bool homogenous;

    // Materials in the inhomogenous case.
    AlignedVector<VectorizedArray<Number>> speed_of_sound;
    AlignedVector<VectorizedArray<Number>> density;

    // Materials in the homogenous case.
    VectorizedArray<Number> speed_of_sound_homogenous;
    VectorizedArray<Number> density_homogenous;
  };

  // To be able to access the material data in every cell in a thread safe way
  // @c MaterialEvaluation is used. Similar to @c FEEvaluation, every thread
  // creates its own instance and thus, there are no race conditions. For
  // in-homogenous materials, a @c reinit_cell() or @c reinit_face() function is
  // used to set the correct material at the current cell batch. In the
  // homogenous case the @c _reinit() functions don't have to reset the
  // materials.
  template <int dim, typename Number>
  class MaterialEvaluation
  {
  public:
    MaterialEvaluation(
      const MatrixFree<dim, Number, VectorizedArray<Number>> &matrix_free,
      const CellwiseMaterialData<Number>                     &material_data)
      : phi(matrix_free)
      , phi_face(matrix_free, true)
      , material_data(material_data)
    {
      if (material_data.is_homogenous())
        {
          // Set the material that is used in every cell.
          speed_of_sound = material_data.get_homogenous_speed_of_sound();
          density        = material_data.get_homogenous_density();
        }
    }

    bool is_homogenous() const
    {
      return material_data.is_homogenous();
    }

    // Update the cell data, given a cell batch index.
    void reinit_cell(const unsigned int cell)
    {
      // In the homogenous case we do not have to reset the cell data.
      if (!material_data.is_homogenous())
        {
          // Reinit the FEEvaluation object and set the cell data.
          phi.reinit(cell);
          speed_of_sound =
            phi.read_cell_data(material_data.get_speed_of_sound());
          density = phi.read_cell_data(material_data.get_density());
        }
    }

    // Update the cell data, given a face batch index.
    void reinit_face(const unsigned int face)
    {
      // In the homogenous case we do not have to reset the cell data.
      if (!material_data.is_homogenous())
        {
          // Reinit the FEFaceEvaluation object and set the cell data.
          phi_face.reinit(face);
          speed_of_sound =
            phi_face.read_cell_data(material_data.get_speed_of_sound());
          density = phi_face.read_cell_data(material_data.get_density());
        }
    }

    // Return the speed of sound at the current cell batch.
    VectorizedArray<Number> get_speed_of_sound() const
    {
      return speed_of_sound;
    }

    // Return the density at the current cell batch.
    VectorizedArray<Number> get_density() const
    {
      return density;
    }

  private:
    // Members needed for the inhomogenous case.
    FEEvaluation<dim, -1, 0, 1, Number>     phi;
    FEFaceEvaluation<dim, -1, 0, 1, Number> phi_face;

    // Material defined at every cell.
    const CellwiseMaterialData<Number> &material_data;

    // Materials at current cell.
    VectorizedArray<Number> speed_of_sound;
    VectorizedArray<Number> density;
  };


  //@sect3{Boundary conditions}
  //
  // To be able to use the same kernel, for all face integrals we define
  // a class that returns the needed values at boundaries. In this tutorial
  // homogenous pressure Dirichlet boundary conditions are applied via
  // the mirror priciple, i.e. $p_h^+=-p_h^- + 2g$ with $g=0$.
  template <int dim, typename Number>
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
  template <int dim, typename Number>
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

  //@sect3{Acoustic operator}
  //
  // Class that defines the acoustic operator.
  template <int dim, typename Number, typename remote_value_type>
  class AcousticOperator
  {
    // If the remote evaluators are set up with a VectorizedArray we are
    // using point-to-point interpolation. Otherwise we make use of
    // Nitsche-type mortaring.
    static constexpr bool use_mortaring =
      std::is_floating_point_v<remote_value_type>;

  public:
    // Constructor.
    AcousticOperator(
      const MatrixFree<dim, Number>                &matrix_free,
      std::shared_ptr<CellwiseMaterialData<Number>> material_data,
      const std::set<types::boundary_id>           &remote_face_ids,
      std::shared_ptr<FERemoteEvaluation<dim, 1, remote_value_type>>
        pressure_r_eval,
      std::shared_ptr<FERemoteEvaluation<dim, dim, remote_value_type>>
        velocity_r_eval,
      std::shared_ptr<FERemoteEvaluation<dim, 1, remote_value_type>> c_r_eval,
      std::shared_ptr<FERemoteEvaluation<dim, 1, remote_value_type>> rho_r_eval,
      std::shared_ptr<NonMatching::MappingInfo<dim, dim, Number>>    nm_info =
        nullptr)
      : matrix_free(matrix_free)
      , material_data(material_data)
      , remote_face_ids(remote_face_ids)
      , pressure_r_eval(pressure_r_eval)
      , velocity_r_eval(velocity_r_eval)
      , c_r_eval(c_r_eval)
      , rho_r_eval(rho_r_eval)
      , nm_mapping_info(nm_info)
    {
      if (use_mortaring)
        Assert(
          nm_info,
          ExcMessage(
            "In case of Nitsche-type mortaring NonMatching::MappingInfo has to be provided."));
    }

    // Function to evaluate the acoustic operator.
    template <typename VectorType>
    void evaluate(VectorType &dst, const VectorType &src) const
    {
      // Update the cached values in corresponding the FERemoteEvaluation
      // objects. The material parameters do not change and thus, we do
      // not have to update cached values in @c c_r_eval and @c rho_r_eval.
      pressure_r_eval->gather_evaluate(src, EvaluationFlags::values);
      velocity_r_eval->gather_evaluate(src, EvaluationFlags::values);

      if constexpr (use_mortaring)
        {
          // Perform matrix free loop with Nitsche-type mortaring at
          // non-matching faces.
          matrix_free.loop(
            &AcousticOperator::local_apply_cell,
            &AcousticOperator::local_apply_face,
            &AcousticOperator::local_apply_boundary_face_mortaring,
            this,
            dst,
            src,
            true,
            MatrixFree<dim, Number>::DataAccessOnFaces::values,
            MatrixFree<dim, Number>::DataAccessOnFaces::values);
        }
      else
        {
          // Perform matrix free loop with point-to-point interpolation at
          // non-matching faces.
          matrix_free.loop(
            &AcousticOperator::local_apply_cell,
            &AcousticOperator::local_apply_face,
            &AcousticOperator::local_apply_boundary_face_point_to_point,
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
    void local_apply_cell(
      const MatrixFree<dim, Number>               &matrix_free,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, 1, Number>   pressure(matrix_free, 0, 0, 0);
      FEEvaluation<dim, -1, 0, dim, Number> velocity(matrix_free, 0, 0, 1);

      // Class that gives access to material at each cell
      MaterialEvaluation material(matrix_free, *material_data);

      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          velocity.reinit(cell);
          pressure.reinit(cell);

          pressure.gather_evaluate(src, EvaluationFlags::gradients);
          velocity.gather_evaluate(src, EvaluationFlags::gradients);

          // Get the materials at the corresponding cell. Since we
          // introduced @c MaterialEvaluation we can write the code
          // independent if the material is homogenous or inhomogenous.
          material.reinit_cell(cell);
          const auto c   = material.get_speed_of_sound();
          const auto rho = material.get_density();
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

    // This function evaluates the fluxes at faces between cells with the same
    // material. If boundary faces are under consideration fluxes into
    // neighboring faces do not have to be considered (there are none). For
    // non-matching faces the fluxes into neighboring faces are not considered
    // as well. This is because we iterate over each side of the non-matching
    // face seperately (similar to a cell
    // centric loop).
    template <bool weight_neighbor,
              typename InternalFaceIntegratorPressure,
              typename InternalFaceIntegratorVelocity,
              typename ExternalFaceIntegratorPressure,
              typename ExternalFaceIntegratorVelocity>
    inline DEAL_II_ALWAYS_INLINE void evaluate_face_kernel(
      InternalFaceIntegratorPressure                           &pressure_m,
      InternalFaceIntegratorVelocity                           &velocity_m,
      ExternalFaceIntegratorPressure                           &pressure_p,
      ExternalFaceIntegratorVelocity                           &velocity_p,
      const typename InternalFaceIntegratorPressure::value_type c,
      const typename InternalFaceIntegratorPressure::value_type rho) const
    {
      // Compute penalty parameters from material parameters.
      const auto tau   = 0.5 * rho * c;
      const auto gamma = 0.5 / (rho * c);

      for (unsigned int q : pressure_m.quadrature_point_indices())
        {
          const auto n  = pressure_m.normal_vector(q);
          const auto pm = pressure_m.get_value(q);
          const auto um = velocity_m.get_value(q);

          const auto pp = pressure_p.get_value(q);
          const auto up = velocity_p.get_value(q);

          // Compute homogenous local Lax-Friedrichs fluxes and submit the
          // corrsponding values to the integrators.
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

    // This function evaluates the fluxes at faces between cells with differnet
    // materials. This can only happen over non-matching interfaces. Therefore,
    // it is clear that weight_neighbor=false.
    template <typename InternalFaceIntegratorPressure,
              typename InternalFaceIntegratorVelocity,
              typename ExternalFaceIntegratorPressure,
              typename ExternalFaceIntegratorVelocity,
              typename MaterialIntegrator>
    void evaluate_face_kernel_inhomogeneous(
      InternalFaceIntegratorPressure                           &pressure_m,
      InternalFaceIntegratorVelocity                           &velocity_m,
      const ExternalFaceIntegratorPressure                     &pressure_p,
      const ExternalFaceIntegratorVelocity                     &velocity_p,
      const typename InternalFaceIntegratorPressure::value_type c,
      const typename InternalFaceIntegratorPressure::value_type rho,
      const MaterialIntegrator                                 &c_r,
      const MaterialIntegrator                                 &rho_r) const
    {
      // interior material information is constant over quadrature points
      const auto tau_m   = 0.5 * rho * c;
      const auto gamma_m = 0.5 / (rho * c);

      for (unsigned int q : pressure_m.quadrature_point_indices())
        {
          // The material at the neighboring face might vary in every quadrature
          // point.
          const auto c_p           = c_r.get_value(q);
          const auto rho_p         = rho_r.get_value(q);
          const auto tau_p         = 0.5 * rho_p * c_p;
          const auto gamma_p       = 0.5 / (rho_p * c_p);
          const auto tau_sum_inv   = 1.0 / (tau_m + tau_p);
          const auto gamma_sum_inv = 1.0 / (gamma_m + gamma_p);

          const auto n  = pressure_m.normal_vector(q);
          const auto pm = pressure_m.get_value(q);
          const auto um = velocity_m.get_value(q);

          const auto pp = pressure_p.get_value(q);
          const auto up = velocity_p.get_value(q);


          // Compute in-homogenous fluxes and submit the corrsponding values
          // to the integrators.
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

    // This function evaluates the inner face integrals.
    template <typename VectorType>
    void local_apply_face(
      const MatrixFree<dim, Number>               &matrix_free,
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

      // Class that gives access to material at each cell
      MaterialEvaluation material(matrix_free, *material_data);

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

          material.reinit_face(face);
          evaluate_face_kernel<true>(pressure_m,
                                     velocity_m,
                                     pressure_p,
                                     velocity_p,
                                     material.get_speed_of_sound(),
                                     material.get_density());

          pressure_m.integrate_scatter(EvaluationFlags::values, dst);
          pressure_p.integrate_scatter(EvaluationFlags::values, dst);
          velocity_m.integrate_scatter(EvaluationFlags::values, dst);
          velocity_p.integrate_scatter(EvaluationFlags::values, dst);
        }
    }


    //@sect4{Matrix-free boundary function for point-to-point interpolation}
    //
    // This function evaluates the boundary face integrals and the
    // non-matching face integrals using point-to-point interpolation.
    template <typename VectorType>
    void local_apply_boundary_face_point_to_point(
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

      // Class that gives access to material at each cell
      MaterialEvaluation material(matrix_free, *material_data);

      // Remote evaluatiors
      auto pressure_r = pressure_r_eval->get_data_accessor();
      auto velocity_r = velocity_r_eval->get_data_accessor();
      auto c_r        = c_r_eval->get_data_accessor();
      auto rho_r      = rho_r_eval->get_data_accessor();

      for (unsigned int face = face_range.first; face < face_range.second;
           face++)
        {
          velocity_m.reinit(face);
          pressure_m.reinit(face);

          pressure_m.gather_evaluate(src, EvaluationFlags::values);
          velocity_m.gather_evaluate(src, EvaluationFlags::values);

          if (HelperFunctions::is_non_matching_face(
                remote_face_ids, matrix_free.get_boundary_id(face)))
            {
              // If @c face is nonmatching we have to query values via the
              // FERemoteEvaluaton objects. This is done by passing the
              // corresponding FERemoteEvaluaton objects to the function that
              // evaluates the kernel. As mentioned above, each side of the
              // non-matching interface is iterated seperately and we do not
              // have to consider the neighbor in the kernel. Note, that the
              // values in the FERemoteEvaluaton objects are already updated at
              // this point.

              // For point-to-point interpolation we simply use the
              // corresponding FERemoteEvaluaton objects in combination with the
              // standard FEFaceEvaluation objects.
              velocity_r.reinit(face);
              pressure_r.reinit(face);

              material.reinit_face(face);

              if (material.is_homogenous())
                {
                  // If homogenous material is considered do not use the
                  // in-homogenous fluxes. While it would be possible
                  // to use the in-homogenous fluxes they are more expensive to
                  // compute.
                  evaluate_face_kernel<false>(pressure_m,
                                              velocity_m,
                                              pressure_r,
                                              velocity_r,
                                              material.get_speed_of_sound(),
                                              material.get_density());
                }
              else
                {
                  // If in-homogenous material is considered use the
                  // in-homogenous fluxes.
                  c_r.reinit(face);
                  rho_r.reinit(face);
                  evaluate_face_kernel_inhomogeneous(
                    pressure_m,
                    velocity_m,
                    pressure_r,
                    velocity_r,
                    material.get_speed_of_sound(),
                    material.get_density(),
                    c_r,
                    rho_r);
                }
            }
          else
            {
              // If @c face is a standard boundary face, evaluate the integral
              // as usual in the matrix free context. To be able to use the same
              // kernel as for inner faces we pass the boundary condition
              // objects to the function that evaluates the kernel. As mentioned
              // above, there is no neighbor to consider in the kernel.
              material.reinit_face(face);
              evaluate_face_kernel<false>(pressure_m,
                                          velocity_m,
                                          pressure_bc,
                                          velocity_bc,
                                          material.get_speed_of_sound(),
                                          material.get_density());
            }

          pressure_m.integrate_scatter(EvaluationFlags::values, dst);
          velocity_m.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    //@sect4{Matrix-free boundary function for Nitsche-type mortaring}
    //
    // This function evaluates the boundary face integrals and the
    // non-matching face integrals using Nitsche-type mortaring.
    template <typename VectorType>
    void local_apply_boundary_face_mortaring(
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

      // For Nitsche-type mortaring we are evaluating the integrals over
      // intersections. This is why, quadrature points are arbitrarely
      // distributed on every face. Thus, we can not make use of face batches
      // and FEFaceEvaluation but have to consider each face individually and
      // make use of @c FEPointEvaluation to evaluate the integrals in the
      // arbitrarely distributed quadrature points.
      // TODO: The setup of FEPointEvaluation is more
      //  expensive than that of FEEvaluation (since MatrixFree stores
      //  all the precomputed data). Should we use ThreadLocalStorage?
      FEPointEvaluation<1, dim, dim, Number> pressure_m_mortar(
        *nm_mapping_info, matrix_free.get_dof_handler().get_fe(), 0);
      FEPointEvaluation<dim, dim, dim, Number> velocity_m_mortar(
        *nm_mapping_info, matrix_free.get_dof_handler().get_fe(), 1);
      AlignedVector<Number> buffer(
        matrix_free.get_dof_handler().get_fe().dofs_per_cell);

      BCEvalP pressure_bc(pressure_m);
      BCEvalU velocity_bc(velocity_m);

      MaterialEvaluation material(matrix_free, *material_data);

      auto pressure_r_mortar = pressure_r_eval->get_data_accessor();
      auto velocity_r_mortar = velocity_r_eval->get_data_accessor();
      auto c_r               = c_r_eval->get_data_accessor();
      auto rho_r             = rho_r_eval->get_data_accessor();

      for (unsigned int face = face_range.first; face < face_range.second;
           ++face)
        {
          if (HelperFunctions::is_non_matching_face(
                remote_face_ids, matrix_free.get_boundary_id(face)))
            {
              // For mortaring, we have to cosider every face from the face
              // batches seperately and have to use the FEPointEvaluation
              // objects to be able to evaluate the integrals with the
              // arbitrarily distributed quadrature points.
              for (unsigned int v = 0;
                   v < matrix_free.n_active_entries_per_face_batch(face);
                   ++v)
                {
                  const auto [cell, f] =
                    matrix_free.get_face_iterator(face, v, true);

                  // TODO: we will be able to simplify this in a follow up PR
                  // once there is FEFacePointEvaluation.
                  velocity_m_mortar.reinit(cell->active_cell_index(), f);
                  pressure_m_mortar.reinit(cell->active_cell_index(), f);

                  cell->get_dof_values(src, buffer.begin(), buffer.end());
                  velocity_m_mortar.evaluate(buffer, EvaluationFlags::values);
                  pressure_m_mortar.evaluate(buffer, EvaluationFlags::values);

                  velocity_r_mortar.reinit(cell->active_cell_index(), f);
                  pressure_r_mortar.reinit(cell->active_cell_index(), f);

                  material.reinit_face(face);

                  if (material.is_homogenous())
                    {
                      // If homogenous material is considered do not use the
                      // in-homogenous fluxes. While it would be possible
                      // to use the in-homogenous fluxes they are more expensive
                      // to compute.
                      // Since we are operating on face @c v we call
                      //@c material.get_density()[v]
                      evaluate_face_kernel<false>(
                        pressure_m_mortar,
                        velocity_m_mortar,
                        pressure_r_mortar,
                        velocity_r_mortar,
                        material.get_speed_of_sound()[v],
                        material.get_density()[v]);
                    }
                  else
                    {
                      c_r.reinit(cell->active_cell_index(), f);
                      rho_r.reinit(cell->active_cell_index(), f);
                      evaluate_face_kernel_inhomogeneous(
                        pressure_m_mortar,
                        velocity_m_mortar,
                        pressure_r_mortar,
                        velocity_r_mortar,
                        material.get_speed_of_sound()[v],
                        material.get_density()[v],
                        c_r,
                        rho_r);
                    }

                  // @c integrate(sum_into_values=false) zeroes out the
                  // whole buffer and writes the integrated values in the
                  // correct palces of the buffer.
                  velocity_m_mortar.integrate(buffer,
                                              EvaluationFlags::values,
                                              /*sum_into_values=*/false);

                  // We have to call @c integrate(sum_into_values=true) to
                  // avoid that the vales written by
                  // velocity_m_mortar.integrate() are zeroed out.
                  // TODO: should integrate only zero out the values it writes
                  //  to or are there cases in which all values have to be
                  //  zeroed out? If not the whole buffer should be zeroed out:
                  //  Follow up PR.
                  pressure_m_mortar.integrate(buffer,
                                              EvaluationFlags::values,
                                              /*sum_into_values=*/true);

                  cell->distribute_local_to_global(buffer.begin(),
                                                   buffer.end(),
                                                   dst);
                }
            }
          else
            {
              // Same as in @c local_apply_boundary_face_point_to_point().
              velocity_m.reinit(face);
              pressure_m.reinit(face);

              pressure_m.gather_evaluate(src, EvaluationFlags::values);
              velocity_m.gather_evaluate(src, EvaluationFlags::values);

              material.reinit_face(face);
              evaluate_face_kernel<false>(pressure_m,
                                          velocity_m,
                                          pressure_bc,
                                          velocity_bc,
                                          material.get_speed_of_sound(),
                                          material.get_density());

              pressure_m.integrate_scatter(EvaluationFlags::values, dst);
              velocity_m.integrate_scatter(EvaluationFlags::values, dst);
            }
        }
    }

    const MatrixFree<dim, Number> &matrix_free;

    // CellwiseMaterialData is stored as shared pointer with the same
    // argumentation.
    const std::shared_ptr<CellwiseMaterialData<Number>> material_data;

    const std::set<types::boundary_id> remote_face_ids;

    // FERemoteEvaluation objects are strored as shared pointers. This way,
    // they can also be used for other operators without caching the values
    // multiple times.
    const std::shared_ptr<FERemoteEvaluation<dim, 1, remote_value_type>>
      pressure_r_eval;
    const std::shared_ptr<FERemoteEvaluation<dim, dim, remote_value_type>>
      velocity_r_eval;

    const std::shared_ptr<FERemoteEvaluation<dim, 1, remote_value_type>>
      c_r_eval;
    const std::shared_ptr<FERemoteEvaluation<dim, 1, remote_value_type>>
      rho_r_eval;

    const std::shared_ptr<NonMatching::MappingInfo<dim, dim, Number>>
      nm_mapping_info;
  };

  //@sect3{Inverse mass operator}
  //
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
      matrix_free.cell_loop(&InverseMassOperator::local_apply_cell,
                            this,
                            dst,
                            src);
    }

  private:
    // Apply the inverse mass operator onto every cell batch.
    template <typename VectorType>
    void local_apply_cell(
      const MatrixFree<dim, Number>               &mf,
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

  //@sect3{Runge-Kutta timestepping}
  //
  // This class implements a Runge-Kutta scheme of order 2.
  template <int dim, typename Number, typename remote_value_type>
  class RungeKutta2
  {
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

  public:
    // Constructor.
    RungeKutta2(
      const std::shared_ptr<InverseMassOperator<dim, Number>>
        inverse_mass_operator,
      const std::shared_ptr<AcousticOperator<dim, Number, remote_value_type>>
        acoustic_operator)
      : inverse_mass_operator(inverse_mass_operator)
      , acoustic_operator(acoustic_operator)
    {}

    // Setup and run time loop.
    void run(const MatrixFree<dim, Number> &matrix_free,
             const double                   cr,
             const double                   end_time,
             const double                   speed_of_sound,
             const Function<dim>           &initial_condition,
             const std::string             &vtk_prefix)
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
        cr * HelperFunctions::compute_dt_cfl(h_min, degree, speed_of_sound);

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
                                     "step_89-" + vtk_prefix +
                                       std::to_string(timestep));

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
    const std::shared_ptr<AcousticOperator<dim, Number, remote_value_type>>
      acoustic_operator;
  };


  // @sect3{Construction of non-matching triangulations}
  //
  // This function creates a two dimensional squared triangulation
  // that spans from (0,0) to (1,1). It consists of two subdomains.
  // The left subdomain spans from (0,0) to (0.5,1). The right
  // subdomain spans from (0.5,0) to (1,1). The left subdomain has
  // three times smaller elements compared to the right subdomain.
  template <int dim>
  void build_non_matching_triangulation(
    parallel::distributed::Triangulation<dim> &tria,
    std::set<types::boundary_id>              &non_matching_faces,
    const unsigned int                         refinements)
  {
    const double length = 1.0;

    // At non-matching interfaces, we provide different boundary
    // IDs. These boundary IDs have to differ because later on
    // RemotePointEvaluation has to search for remote points for
    // each face, that are defined in the same mesh (since we merge
    // the mesh) but not on the same side of the non-matching interface.
    const types::boundary_id non_matching_id_left  = 98;
    const types::boundary_id non_matching_id_right = 99;

    // Provide this information to the caller.
    non_matching_faces.insert(non_matching_id_left);
    non_matching_faces.insert(non_matching_id_right);

    // Construct left part of mesh.
    Triangulation<dim> tria_left;
    const unsigned int subdiv_left = 3;
    GridGenerator::subdivided_hyper_rectangle(tria_left,
                                              {subdiv_left, 2 * subdiv_left},
                                              {0.0, 0.0},
                                              {0.5 * length, length});

    // The left part of the mesh has a material ID of 0.
    for (const auto &cell : tria_left.active_cell_iterators())
      cell->set_material_id(0);

    // The right face is non-matching. All other boundary IDs
    // are set to 0.
    for (const auto &face : tria_left.active_face_iterators())
      if (face->at_boundary())
        {
          face->set_boundary_id(0);
          if (face->center()[0] > 0.5 * length - 1e-6)
            face->set_boundary_id(non_matching_id_left);
        }

    // Construct right part of mesh.
    Triangulation<dim> tria_right;
    const unsigned int subdiv_right = 1;
    GridGenerator::subdivided_hyper_rectangle(tria_right,
                                              {subdiv_right, 2 * subdiv_right},
                                              {0.5 * length, 0.0},
                                              {length, length});

    // The right part of the mesh has a material ID of 1.
    for (const auto &cell : tria_right.active_cell_iterators())
      cell->set_material_id(1);

    // The left face is non-matching. All other boundary IDs
    // are set to 0.
    for (const auto &face : tria_right.active_face_iterators())
      if (face->at_boundary())
        {
          face->set_boundary_id(0);
          if (face->center()[0] < 0.5 * length + 1e-6)
            face->set_boundary_id(non_matching_id_right);
        }

    // Merge triangulations with tolerance 0 to ensure no vertices
    // are merged.
    GridGenerator::merge_triangulations(tria_left,
                                        tria_right,
                                        tria,
                                        /*tolerance*/ 0.,
                                        /*copy_manifold_ids*/ false,
                                        /*copy_boundary_ids*/ true);
    tria.refine_global(refinements);
  }

  // @sect3{Point-to-point interpolation}
  //
  // The main purpose of this function is to fill the remote communicator that
  // is needed for point-to-point interpolation. Using this remote communicator
  // also the corresponding remote evaluators are setup. Eventually, the
  // operators are handed to the time integrator that runs the simulation.
  //
  template <int dim, typename Number>
  void run_with_point_to_point_interpolation(
    const MatrixFree<dim, Number>      &matrix_free,
    const std::set<types::boundary_id> &non_matching_faces,
    const std::map<types::material_id, std::pair<double, double>> &materials,
    const double                                                   end_time,
    const Function<dim> &initial_condition,
    const std::string   &vtk_prefix)
  {
    const auto &dof_handler = matrix_free.get_dof_handler();
    const auto &tria        = dof_handler.get_triangulation();
    const auto &mapping     = *matrix_free.get_mapping_info().mapping;

    // Communication objects know about the communication pattern. I.e.,
    // they know about the cells and quadrature ponts that have to be
    // evaluated at remote faces. This information is given via
    // RemotePointEvaluation. Additionally, the communication objects
    // have to be able to match the quadrature points of the remote
    // points (that provide exterior information) to the quadrature points
    // defined at the interior cell. In case of point-to-point interpolation
    // a vector of pairs with face batch Ids and the number of faces in the
    // batch is needed. @c FERemoteCommunicationObjectEntityBatches
    // is a container to store this information.
    //
    // The information is filled outside of the actual class since in some cases
    // the information is available from some heuristic and
    // it is possible to skip some expensive operations. This is for example
    // the case for sliding rotating interfaces with equally spaced elements on
    // both sides of the non-matching interface @cite duerrwaechter2021an.
    //
    // We need multiple communication objects (one for each non-matching face
    // ID).
    std::vector<FERemoteCommunicationObjectEntityBatches<dim>> comm_objects;

    // Additionally to the communicaton objects we need a vector
    // that stores quadrature rules for every face batch.
    // The quadrature can be empty in case of non non-matching faces,
    // i.e. boundary faces. Internally this information is needed to correctly
    // access values over multiple communication objects.
    // TODO: This interface is up for discussion. It is currently morivated by
    // NonMatchingMapping info but it is not optimal.
    std::vector<Quadrature<dim>> global_quadrature_vector(
      matrix_free.n_boundary_face_batches());

    // Get the range of face batches we have to look at during construction of
    // the communication objects. We only have to look at boundary faces.
    const auto face_batch_range =
      std::make_pair(matrix_free.n_inner_face_batches(),
                     matrix_free.n_inner_face_batches() +
                       matrix_free.n_boundary_face_batches());

    // Iterate over all sides of the non-matching interface.
    for (const auto &nm_face : non_matching_faces)
      {
        // Construct the communication object for every face ID:

        // 1) RemotePointEvaluation with lambda that rules out all cells
        // that are connected to the current side of the non-matching interface.
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

        // 2) Face batch IDs and number of faces in batch.
        std::vector<std::pair<unsigned int, unsigned int>>
          face_batch_id_n_faces;

        // Points that are searched by rpe.
        std::vector<Point<dim>> points;

        // Temporarily setup FEFaceEvaluation to access the quadrature points at
        // the faces on the non-matching interface.
        FEFaceEvaluation<dim, -1, 0, 1, Number> phi(matrix_free, true, 0, 0, 0);

        // Iterate over the boundary faces.
        // TODO:!!!!!!!! why do results differ if run with MPI?!
        // IT HAS TO BE EITHER THIS SETUP; REINIT_FACES OR COPY DATA
        // DEBUG BY PROVDIDING INITIAL CONDITION OF VELOCITY THAT IS
        // THE COODRINATES AND CHECK IF QUADRATURE POINTS AND REMOTE
        // VALUES MATCH!
        // THIS IS THE LAST TODO BEFORE FINISHING DOCUMENATATION AND FILING A PR.
        for (unsigned int bface = 0;
             bface < face_batch_range.second - face_batch_range.first;
             ++bface)
          {
            const unsigned int face = face_batch_range.first + bface;

            if (matrix_free.get_boundary_id(face) == nm_face)
              {
                phi.reinit(face);

                // If face is on the current side of the non-matching interface.
                // Add face batch ID and number of faces in batch to the
                // corresponding data structure.
                const unsigned int n_faces =
                  matrix_free.n_active_entries_per_face_batch(face);
                face_batch_id_n_faces.push_back(std::make_pair(face, n_faces));

                // Append the quadrature points to the points we need to search
                // for.
                for (unsigned int v = 0; v < n_faces; ++v)
                  {
                    for (unsigned int q = 0; q < phi.n_q_points; ++q)
                      {
                        const auto         point = phi.quadrature_point(q);
                        dealii::Point<dim> temp;
                        for (unsigned int i = 0; i < dim; ++i)
                          temp[i] = point[i][v];

                        points.push_back(temp);
                      }
                  }

                // Insert a quadrature rule of correct size into the global
                // quadrature vector. First check that each face is only
                // considered once.
                Assert(global_quadrature_vector[bface].size() == 0,
                       ExcMessage(
                         "Quadrature for given face already provided."));

                global_quadrature_vector[bface] =
                  Quadrature<dim>(phi.n_q_points);
              }
          }

        // Reinit RPE and ensure all points are found.
        rpe->reinit(points, tria, mapping);
        Assert(rpe->all_points_found(),
               ExcMessage("Not all remote points found."));

        // Add communication object to the list of objects.
        FERemoteCommunicationObjectEntityBatches<dim> co;
        co.batch_id_n_entities = face_batch_id_n_faces;
        co.rpe                 = rpe;
        comm_objects.push_back(co);
      }

    // Renit the communicator with the communication objects.
    // FERemoteEvaluationCommunicator<dim> remote_communicator;
    FERemoteEvaluationCommunicator<dim> remote_communicator;
    remote_communicator.reinit_faces(comm_objects,
                                     face_batch_range,
                                     global_quadrature_vector);


    // We are using point-to-point interpolation and can therefore
    // easily access the corresponding data at face batches. This
    // is why we use a @c VectorizedArray as @c remote_value_type
    using remote_value_type = VectorizedArray<Number>;

    // Set up FERemoteEvaluation object that accesses the pressure
    // at remote faces.
    const auto pressure_r =
      std::make_shared<FERemoteEvaluation<dim, 1, remote_value_type>>(
        remote_communicator, dof_handler, /*first_selected_component*/ 0);

    // Set up FERemoteEvaluation object that accesses the velocity
    // at remote faces.
    const auto velocity_r =
      std::make_shared<FERemoteEvaluation<dim, dim, remote_value_type>>(
        remote_communicator, dof_handler, /*first_selected_component*/ 1);

    // Set up cellwise material data.
    const auto material_data =
      std::make_shared<CellwiseMaterialData<Number>>(matrix_free, materials);

    // If we have an inhomogenous problem, we have to setup the
    // material handler that accesses the materials at remote faces.
    const auto c_r =
      std::make_shared<FERemoteEvaluation<dim, 1, remote_value_type>>(
        remote_communicator,
        matrix_free.get_dof_handler().get_triangulation(),
        /*first_selected_component*/ 0);
    const auto rho_r =
      std::make_shared<FERemoteEvaluation<dim, 1, remote_value_type>>(
        remote_communicator,
        matrix_free.get_dof_handler().get_triangulation(),
        /*first_selected_component*/ 0);

    if (!material_data->is_homogenous())
      {
        // Initialize and fill DoF vectors that contain the materials.
        Vector<Number> c(
          matrix_free.get_dof_handler().get_triangulation().n_active_cells());
        Vector<Number> rho(
          matrix_free.get_dof_handler().get_triangulation().n_active_cells());

        for (const auto &cell : matrix_free.get_dof_handler()
                                  .get_triangulation()
                                  .active_cell_iterators())
          {
            c[cell->active_cell_index()] =
              materials.at(cell->material_id()).first;
            rho[cell->active_cell_index()] =
              materials.at(cell->material_id()).second;
          }

        // Values are constant during the simulation, therefore after
        // caching the values here there is no need in calling
        // @c gather_evaluate() again.
        c_r->gather_evaluate(c, EvaluationFlags::values);
        rho_r->gather_evaluate(rho, EvaluationFlags::values);
      }


    // Setup inverse mass operator.
    const auto inverse_mass_operator =
      std::make_shared<InverseMassOperator<dim, Number>>(matrix_free);

    // Setup the acoustic operator. Using this constructor makes the
    //  operator use point-to-point interpolation.
    const auto acoustic_operator =
      std::make_shared<AcousticOperator<dim, Number, remote_value_type>>(
        matrix_free,
        material_data,
        non_matching_faces,
        pressure_r,
        velocity_r,
        c_r,
        rho_r);

    // Compute the the maximum speed of sound, needed for the computation of
    // the time-step size.
    double speed_of_sound_max = 0.0;
    for (const auto &mat : materials)
      speed_of_sound_max = std::max(speed_of_sound_max, mat.second.first);

    // Set up time integrator.
    RungeKutta2<dim, Number, remote_value_type> time_integrator(
      inverse_mass_operator, acoustic_operator);

    // Run time loop with Courant number 0.1.
    time_integrator.run(matrix_free,
                        /*Cr*/ 0.2,
                        end_time,
                        speed_of_sound_max,
                        initial_condition,
                        vtk_prefix);
  }

  // @sect3{Nitsche-type mortaring}
  //
  // The main purpose of this function is to fill the remote communicator that
  // is needed for Nitsche-type mortaring. Using this remote communicator also
  // the corresponding remote evaluators are setup. Eventually, the operators
  // are handed to the time integrator that runs the simulation.
  template <int dim, typename Number>
  void run_with_nitsche_type_mortaring(
    const MatrixFree<dim, Number>      &matrix_free,
    const std::set<types::boundary_id> &non_matching_faces,
    const std::map<types::material_id, std::pair<double, double>> &materials,
    const double                                                   end_time,
    const Function<dim> &initial_condition,
    const std::string   &vtk_prefix)
  {
    const auto &dof_handler       = matrix_free.get_dof_handler();
    const auto &tria              = dof_handler.get_triangulation();
    const auto &mapping           = *matrix_free.get_mapping_info().mapping;
    const auto  n_quadrature_pnts = matrix_free.get_quadrature().size();

    std::vector<std::vector<Quadrature<dim - 1>>> global_quadrature_vector;
    for (const auto &cell : tria.active_cell_iterators())
      global_quadrature_vector.emplace_back(
        std::vector<Quadrature<dim - 1>>(cell->n_faces()));

    // In case of Nitsche-type mortaring a vector of pairs with cell iterators
    // and face number is needed as communication object.
    // @c FERemoteCommunicationObjectFaces is a container to store this
    // infromation.
    //
    // We need multiple communication objects (one for each non-matching face
    // ID).
    std::vector<FERemoteCommunicationObjectFaces<dim>> comm_objects;

    // Iterate over all sides of the non-matching interface.
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

        // 2) Create RPE
        // In the Nitsche-type case we do not collect points for the setup
        // of RemotePointEvaluation. Instead we compute intersections between
        // the faces and setup RemotePointEvaluation with the computed
        // intersections.

        // Create bounding boxes to search in
        std::vector<BoundingBox<dim>> local_boxes;
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            local_boxes.emplace_back(mapping.get_bounding_box(cell));

        // Create r-tree of bounding boxes
        const auto local_tree = pack_rtree(local_boxes);

        // Compress r-tree to a minimal set of bounding boxes
        std::vector<std::vector<BoundingBox<dim>>> global_bboxes(1);
        global_bboxes[0] = extract_rtree_level(local_tree, 0);

        const GridTools::Cache<dim, dim> cache(tria, mapping);

        // Build intersection requests. Intersection requests
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

        // Compute intersection data and rule out intersections between
        // identic faces by the same lambda as in the case of point-to-point
        // interpolation.
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

        // Convert to RPE
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
        //  able to retrieve relevant information. This will be done in a
        //  follow up PR.

        // TODO: NonMatchingMappingInfo should be able to work with
        //  Quadrature<dim> instead <dim-1>. Currently we are constructing
        //  dim-1 from dim and inside MappingInfo it is converted back.
        //  As far as we know Max is working on that. If not: follow up
        //  PR.

        // 3) Fill global quadrature vector.
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

        // Add communication object
        FERemoteCommunicationObjectFaces<dim> co;
        co.cell_face_nos = cell_face_pairs;
        co.rpe           = rpe;
        comm_objects.push_back(co);
      }

    // Renit the communicator with the communication objects.
    FERemoteEvaluationCommunicator<dim> remote_communicator;
    remote_communicator.reinit_faces(
      comm_objects,
      matrix_free.get_dof_handler().get_triangulation().active_cell_iterators(),
      global_quadrature_vector);


    // Same as above but since quadrature points are aribtrarily distributed
    // we have to considere each face in a batch seperately and can not make
    // use of @c VecorizedArray.
    using remote_value_type = Number;

    const auto pressure_r =
      std::make_shared<FERemoteEvaluation<dim, 1, remote_value_type>>(
        remote_communicator, dof_handler, /*first_selected_component*/ 0);

    const auto velocity_r =
      std::make_shared<FERemoteEvaluation<dim, dim, remote_value_type>>(
        remote_communicator, dof_handler, /*first_selected_component*/ 1);

    const auto material_data =
      std::make_shared<CellwiseMaterialData<Number>>(matrix_free, materials);

    const auto c_r =
      std::make_shared<FERemoteEvaluation<dim, 1, remote_value_type>>(
        remote_communicator,
        matrix_free.get_dof_handler().get_triangulation(),
        /*first_selected_component*/ 0);
    const auto rho_r =
      std::make_shared<FERemoteEvaluation<dim, 1, remote_value_type>>(
        remote_communicator,
        matrix_free.get_dof_handler().get_triangulation(),
        /*first_selected_component*/ 0);

    if (!material_data->is_homogenous())
      {
        Vector<Number> c(
          matrix_free.get_dof_handler().get_triangulation().n_active_cells());
        Vector<Number> rho(
          matrix_free.get_dof_handler().get_triangulation().n_active_cells());

        for (const auto &cell : matrix_free.get_dof_handler()
                                  .get_triangulation()
                                  .active_cell_iterators())
          {
            c[cell->active_cell_index()] =
              materials.at(cell->material_id()).first;
            rho[cell->active_cell_index()] =
              materials.at(cell->material_id()).second;
          }

        c_r->gather_evaluate(c, EvaluationFlags::values);
        rho_r->gather_evaluate(rho, EvaluationFlags::values);
      }

    // Quadrature points are arbitrarily distributed on each non-matching
    // face. Therefore, we have to make use of FEPointEvaluation.
    // FEPointEvaluation needs NonMatching::MappingInfo to work at the correct
    // quadrature points that are in sync with used FERemoteEvaluation object.
    // In the case of mortaring, we have to use the weights provided by the
    // quadrature rules that are used to setup NonMatching::MappingInfo.
    // Therefore we have to set the flag use_global_weights.
    typename NonMatching::MappingInfo<dim, dim, Number>::AdditionalData
      additional_data;
    additional_data.use_global_weights = true;

    // Setup NonMatching::MappingInfo with needed update flags and
    // additional_data.
    auto nm_mapping_info =
      std::make_shared<NonMatching::MappingInfo<dim, dim, Number>>(
        mapping,
        update_values | update_JxW_values | update_normal_vectors |
          update_quadrature_points,
        additional_data);

    // Reinit faces with the same vector of quadratures that is used to setup
    // the remote communicator.
    nm_mapping_info->reinit_faces(
      matrix_free.get_dof_handler().get_triangulation().active_cell_iterators(),
      global_quadrature_vector);

    // Setup inverse mass operator.
    const auto inverse_mass_operator =
      std::make_shared<InverseMassOperator<dim, Number>>(matrix_free);

    // Setup the acoustic operator. Using this constructor makes the
    // operator use Nitsche-type mortaring.
    const auto acoustic_operator =
      std::make_shared<AcousticOperator<dim, Number, remote_value_type>>(
        matrix_free,
        material_data,
        non_matching_faces,
        pressure_r,
        velocity_r,
        c_r,
        rho_r,
        nm_mapping_info);

    // Compute the the maximum speed of sound, needed for the computation of
    // the time-step size.
    double speed_of_sound_max = 0.0;
    for (const auto &mat : materials)
      speed_of_sound_max = std::max(speed_of_sound_max, mat.second.first);

    ConditionalOStream pcout(
      std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

    // Setup time integrator.
    RungeKutta2<dim, Number, remote_value_type> time_integrator(
      inverse_mass_operator, acoustic_operator);

    // Run time loop with Courant number 0.1.
    time_integrator.run(matrix_free,
                        /*Cr*/ 0.2,
                        end_time,
                        speed_of_sound_max,
                        initial_condition,
                        vtk_prefix);
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
  ConditionalOStream pcout(std::cout,
                           (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                            0));

  const unsigned int refinements = 2;
  const unsigned int degree      = 5;

  // Construct non-matching triangulation and fill non-matching boundary IDs.
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  pcout << "Create non-matching grid..." << std::endl;

  std::set<types::boundary_id> non_matching_faces;
  Step89::build_non_matching_triangulation(tria,
                                           non_matching_faces,
                                           refinements);

  pcout << " - Refinement level: " << refinements << std::endl;
  pcout << " - Number of cells: " << tria.n_cells() << std::endl;

  // Setup MatrixFree.

  pcout << "Create DoFHandler..." << std::endl;
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FESystem<dim>(FE_DGQ<dim>(degree), dim + 1));
  pcout << " - Number of DoFs: " << dof_handler.n_dofs() << std::endl;

  AffineConstraints<Number> constraints;
  constraints.close();

  pcout << "Setup MatrixFree..." << std::endl;
  typename MatrixFree<dim, Number>::AdditionalData data;
  data.mapping_update_flags = update_gradients | update_values;
  data.mapping_update_flags_inner_faces =
    update_quadrature_points | update_values;
  data.mapping_update_flags_boundary_faces =
    data.mapping_update_flags_inner_faces;

  MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(
    MappingQ1<dim>(), dof_handler, constraints, QGauss<dim>(degree + 1), data);


  pcout << "Run vibrating membrane testcase..." << std::endl;
  // Vibrating membrane testcase:
  //
  // Homogenous pressure DBCs are applied for simplicity. Therefore,
  // modes can not be chosen arbitrarily.
  const double                                            modes = 10.0;
  std::map<types::material_id, std::pair<double, double>> homogenous_material;
  homogenous_material[numbers::invalid_material_id] = std::make_pair(1.0, 1.0);
  const auto initial_solution_membrane =
    Step89::InitialConditionVibratingMembrane<dim>(modes);

  pcout << " - Point-to-point interpolation: " << std::endl;
  // Run vibrating membrane testcase using point-to-point interpolation:

  Step89::run_with_point_to_point_interpolation(
    matrix_free,
    non_matching_faces,
    homogenous_material,
    2.0 * initial_solution_membrane.get_period_duration(
            homogenous_material.begin()->second.first),
    initial_solution_membrane,
    "vm-p2p");

  pcout << " - Nitsche-type mortaring: " << std::endl;
  // Run vibrating membrane testcase using Nitsche-type mortaring:
  Step89::run_with_nitsche_type_mortaring(
    matrix_free,
    non_matching_faces,
    homogenous_material,
    2.0 * initial_solution_membrane.get_period_duration(
            homogenous_material.begin()->second.first),
    initial_solution_membrane,
    "vm-nitsche");

  pcout << "Run testcase with inhomogenous material..." << std::endl;
  // In-homogenous material testcase:
  //
  // Run simple testcase with in-homogenous material and Nitsche-type mortaring:
  std::map<types::material_id, std::pair<double, double>> inhomogenous_material;
  inhomogenous_material[0] = std::make_pair(1.0, 1.0);
  inhomogenous_material[1] = std::make_pair(3.0, 1.0);
  Step89::run_with_nitsche_type_mortaring(matrix_free,
                                          non_matching_faces,
                                          inhomogenous_material,
                                          /*runtime*/ 0.3,
                                          Step89::GaussPulse<dim>(0.3, 0.5),
                                          "inhomogenous");


  return 0;
}
