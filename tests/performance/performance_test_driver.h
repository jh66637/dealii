// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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

#ifndef dealii_performance_test_driver_h
#define dealii_performance_test_driver_h

#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <boost/range/adaptor/indexed.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>


/**
 * An enum class that describes the environment in which performance tests
 * are running.
 */
enum class TestingEnvironment
{
  /**
   * A lightweight mobile laptop with at least 2 physical cores and at
   * least 8 GB of main memory.
   */
  light,
  /**
   * A medium sized work station with at least 8 physical cores and at
   * least 32 GB of main memory.
   */
  medium,
  /**
   * A heavyweight compute node with at least 32 physical cores and at
   * least 128 GB of main memory.
   */
  heavy,
};


/**
 * Return the current testing environment
 */
TestingEnvironment
get_testing_envrionment()
{
#ifndef TESTING_ENVIRONMENT
#  define TESTING_ENVIRONMENT light
#endif
  return TestingEnvironment::TESTING_ENVIRONMENT;
}


/**
 * Supported reporting metrics of the driver.
 */
enum class Metric
{
  /** a timing (wall clock time / cpu time) in seconds */
  timing,
};


/**
 * Returns the chosen metric, number of repetitions, and a list of timer
 * descriptions.
 *
 * This function needs to be implemented by the individual performance
 * tests.
 */
std::tuple<Metric, unsigned int, std::vector<std::string>>
describe_measurements();


/**
 * The Measurement type returned by perform_single_measurement(). This is
 * currently just a typedef for <code>std::vector<double></code> but could
 * be easily extended to a std::variant (or similar) data type supporting
 * something else than double.
 */
using Measurement = std::vector<double>;


/**
 * Perform a single benchmark run returning a Measurement.
 *
 * This function needs to be implemented by the individual performance
 * tests.
 */
Measurement
perform_single_measurement();


int
main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

#ifdef ENABLE_MPI
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  const auto                               this_mpi_process =
    dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
#else
  const auto this_mpi_process = 0;
#endif

  dealii::ConditionalOStream pout(std::cout, this_mpi_process == 0);

#ifndef ENABLE_PERFORMANCE_TESTS
  pout << "Performance test framework disabled, do not run test." << std::endl;
  return 1;
#endif

  const auto description           = describe_measurements();
  const auto metric                = std::get<0>(description);
  const auto number_of_repetitions = std::get<1>(description);
  const auto names                 = std::get<2>(description);

  Assert(number_of_repetitions > 0, dealii::ExcInternalError);
  if (number_of_repetitions == 0)
    {
      pout << "Test specified an invalid number of measurements." << std::endl;
      return 1;
    }

  // Print the measurement type:

  switch (metric)
    {
      case Metric::timing:
        pout << "timing\n";
        break;
    }

  // Header:

  if (number_of_repetitions == 1)
    {
      pout << "name\tvalue\n";
    }
  else
    {
      pout << "name\tmin\tmax\tmean\tstd_dev\tsamples\n";
    }

  // Perform measurements:

  std::vector<Measurement> measurements;
  std::generate_n(std::back_inserter(measurements),
                  number_of_repetitions,
                  perform_single_measurement);

  for (std::size_t i = 0; i < names.size(); ++i)
    {
      const double x = measurements[0][i];

      if (number_of_repetitions == 1)
        {
          pout << names[i] << "\t" << x << "\n";
        }
      else
        {
          double min      = x;
          double max      = x;
          double mean     = x;
          double variance = 0.;

          for (unsigned int k = 1; k < number_of_repetitions; ++k)
            {
              const double x = measurements[k][i];

              min = std::min(min, x);
              max = std::max(max, x);

              const auto new_mean = mean + (x - mean) / (k + 1);

              variance = variance + (x - mean) * (x - new_mean);
              mean     = new_mean;
            }

          const double std_dev =
            number_of_repetitions == 1 ?
              0. :
              std::sqrt(variance / (number_of_repetitions - 1));

          pout << names[i] << "\t" << min << "\t" << max << "\t" << mean << "\t"
               << std_dev << "\t" << number_of_repetitions << "\n";
        }
    }

  return 0;
}

#endif // dealii_performance_test_driver_h