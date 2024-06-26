// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2004 - 2019 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// check that begin() works if the first row(s) is(are) empty

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_sparse_matrix.h>

#include "../tests.h"


void
test()
{
  // test the case that the first row is empty
  dealii::DynamicSparsityPattern dsp_1(2);
  dsp_1.add(1, 1);

  PETScWrappers::SparseMatrix K_1(dsp_1);

  if (K_1.begin()->row() == 1)
    deallog << "OK" << std::endl;

  // test the case that the entire matrix is empty
  dealii::DynamicSparsityPattern dsp_2(2);

  PETScWrappers::SparseMatrix K_2(dsp_2);

  if (K_2.begin() == K_2.end())
    deallog << "OK" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  initlog();
  test();
}
