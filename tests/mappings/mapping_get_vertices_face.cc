// ---------------------------------------------------------------------
//
// Copyright (C) 2019 - 2022 by the deal.II authors
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

// Test Mapping*::get_vertices(face)

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q1_eulerian.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"

template <int dim>
void
do_test(Mapping<dim> mapping, const unsigned int refinement)
{
  Triangulation<dim> tria;
  if (dim > 1)
    GridGenerator::hyper_ball(tria);
  else
    GridGenerator::hyper_cube(tria, -1, 1);

  tria.refine_global(refinement);
  // mapping_cache.initialize(tria, position_lambda);

  deallog << "Testing tria" << std::endl;
  for (const auto &cell : tria.cell_iterators())
    {
      for (auto const &face : cell->face_iterators())
        {
          std::vector<Point<spacedim>> vertices_trafo;
          for (unsigned int i = 0; i < face->n_vertices(); ++i)
            {
              vertices_trafo[i] = mapping.transform_unit_to_real_cell(
                cell,
                mapping.transform_real_to_unit_cell(cell, face->vertex(i)));
            }
          deallog << "Testing cell" << std::endl;
          const auto &vertices_direct = mapping.get_vertices(face);

          for (unsigned int i = 0; i < face->n_vertices(); ++i)
            deallog << vertices_trafo[i] << " vs. " << vertices_direct[i]
                    << std::endl;
        }
    }
  deallog << std::endl;
}


int
main()
{
  // TODO: Test for
  /*   #include <deal.II/fe/mapping.h> */
  /* #include <deal.II/fe/mapping_q_cache.h> */
  /* #include <deal.II/fe/mapping_q1_eulerian.h> */
  /* #include <deal.II/fe/mapping_q_eulerian.h> */
  /* #include <deal.II/fe/mapping_fe_field.h> */


  initlog();
  do_test(MappingQ<1>(1), 1);
  do_test(MappingQ<1>(1), 2);
  do_test(MappingQ<1>(3), 1);
  do_test(MappingQ<1>(3), 2);
  do_test(MappingQ<2>(1), 1);
  do_test(MappingQ<2>(1), 2);
  do_test(MappingQ<2>(3), 1);
  do_test(MappingQ<2>(3), 2);
  do_test(MappingQ<3>(1), 1);
  do_test(MappingQ<3>(1), 2);
  do_test(MappingQ<3>(3), 1);
  do_test(MappingQ<3>(3), 2);
}
