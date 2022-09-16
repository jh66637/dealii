#pragma once
#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/mpi_consensus_algorithms.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/numerics/rtree.h>

#include <deal.II/cgal/intersections.h>
#include <deal.II/cgal/surface_mesh.h>
#include <deal.II/cgal/utilities.h>


namespace internaL
{
  template <int spacedim>
  std::tuple<std::vector<unsigned int>,
             std::vector<unsigned int>,
             std::vector<unsigned int>>
  guess_intersection_owner(
    const std::vector<std::vector<dealii::BoundingBox<spacedim>>>
      &                                                 global_bboxes_slave,
    const dealii::RTree<dealii::BoundingBox<spacedim>> &local_tree_master)
  {
    std::vector<std::pair<unsigned int, unsigned int>> ranks_and_indices;
    ranks_and_indices.reserve(local_tree_master.size());

    // check which processor might hold each simplex
    unsigned int i = 0;
    for (const auto &master_box : local_tree_master)
      {
        for (unsigned int rank = 0; rank < global_bboxes_slave.size(); ++rank)
          for (const auto &slave_box : global_bboxes_slave[rank])
            {
              if (slave_box.get_neighbor_type(master_box) !=
                  dealii::NeighborType::not_neighbors)
                {
                  ranks_and_indices.emplace_back(rank, i);
                  break;
                }
            }
        ++i;
      }

    // convert to CRS
    std::sort(ranks_and_indices.begin(), ranks_and_indices.end());

    std::vector<unsigned int> ranks;
    std::vector<unsigned int> ptr;
    std::vector<unsigned int> indices;

    unsigned int dummy_rank = dealii::numbers::invalid_unsigned_int;

    for (const auto &i : ranks_and_indices)
      {
        if (dummy_rank != i.first)
          {
            dummy_rank = i.first;
            ranks.push_back(dummy_rank);
            ptr.push_back(indices.size());
          }

        indices.push_back(i.second);
      }
    ptr.push_back(indices.size());

    return std::make_tuple(std::move(ranks),
                           std::move(ptr),
                           std::move(indices));
  }
} // namespace internaL


template <int dim>
class ExactFaceIntersections
{
  const std::set<dealii::types::boundary_id> bnd_ids_master;
  const std::function<std::vector<bool>()> & marked_vertices_slave;
  const double                               tolerance;
  const unsigned int                         rtree_level;

  dealii::SmartPointer<dealii::Mapping<dim> const> mapping_master_;

  struct MasterCellData
  {
    static constexpr const unsigned int n_face_verts = std::pow(2, dim - 1);
    std::vector<unsigned int>           face_indices;
    std::vector<typename dealii::Triangulation<dim>::cell_iterator> cells;
    std::vector<std::array<dealii::Point<dim>, n_face_verts>>
                                          master_simplex_buffer;
    std::vector<dealii::BoundingBox<dim>> local_boxes_master;
  };

public:
  ExactFaceIntersections(
    const std::set<dealii::types::boundary_id> bnd_ids_master,
    const std::function<std::vector<bool>()> & marked_vertices_slave = {},
    const double                               tolerance             = 1e-6,
    const unsigned int                         rtree_level           = 0)
    : bnd_ids_master(bnd_ids_master)
    , marked_vertices_slave(marked_vertices_slave)
    , tolerance(tolerance)
    , rtree_level(rtree_level)
  {}

  void
  reinit(const dealii::Triangulation<dim> &tria_master,
         const dealii::Mapping<dim> &      mapping_master,
         const dealii::Triangulation<dim> &tria_slave,
         const dealii::Mapping<dim> &      mapping_slave)
  {
    mapping_master_ = &mapping_master;

    // TODO:
    // dealii::GridTools::Cache<dim> cache_slave(tria_slave, mapping_slave);
    // dealii::GridTools::Cache<dim> cache_master(tria_master, mapping_master);

    // collect master intersection cells
    MasterCellData master_cell_data;

    for (auto const &cell : tria_master.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (auto const &face : cell->face_iterators())
          {
            if (face->at_boundary() &&
                bnd_ids_master.find(face->boundary_id()) !=
                  bnd_ids_master.end())
              {
                master_cell_data.face_indices.emplace_back(face->index());
                master_cell_data.cells.emplace_back(cell);
                master_cell_data.local_boxes_master.emplace_back(
                  mapping_master.get_bounding_box(cell));

                master_cell_data.master_simplex_buffer.emplace_back(
                  dealii::CGALWrappers::get_vertices_in_cgal_order<
                    MasterCellData::n_face_verts>(cell, face, mapping_master));
              }
          }
      }

    // create r-tree of bounding boxes
    const auto local_tree_master =
      dealii::pack_rtree(master_cell_data.local_boxes_master);

    // collect slave bounding boxes
    std::vector<bool> marked_vertices;
    if (marked_vertices_slave)
      marked_vertices = marked_vertices_slave();
    else
      marked_vertices =
        std::move(std::vector<bool>(tria_slave.n_vertices(), true));

    std::vector<
      std::pair<std::pair<unsigned int, unsigned int>,
                std::array<dealii::Point<dim>, int(std::pow(2, dim))>>>
      slave_simplex_buffer;

    std::vector<dealii::BoundingBox<dim>> local_boxes_slave;
    for (const auto &cell : tria_slave.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (unsigned int i = 0; i < cell->n_vertices(); ++i)
          if (marked_vertices[cell->vertex_index(i)])
            {
              slave_simplex_buffer.emplace_back(
                std::make_pair(std::make_pair(cell->level(), cell->index()),
                               dealii::CGALWrappers::get_vertices_in_cgal_order<
                                 int(std::pow(2, dim))>(cell, mapping_slave)));

              local_boxes_slave.emplace_back(
                mapping_slave.get_bounding_box(cell));
              break;
            }
      }

    // create r-tree of bounding boxes
    const auto local_tree_slave = dealii::pack_rtree(local_boxes_slave);

    // compress r-tree to a minimal set of bounding boxes
    const auto local_reduced_box_slave =
      dealii::extract_rtree_level(local_tree_slave, rtree_level);

    // gather bounding boxes of other processes
    const auto global_bboxes_slave =
      dealii::Utilities::MPI::all_gather(tria_slave.get_communicator(),
                                         local_reduced_box_slave);

    const auto &[potential_owners_ranks,
                 potential_owners_ptrs,
                 potential_owners_indices] =
      internaL::guess_intersection_owner(global_bboxes_slave,
                                         local_tree_master);

    const auto translate = [&](const unsigned int other_rank) {
      const auto ptr = std::find(potential_owners_ranks.begin(),
                                 potential_owners_ranks.end(),
                                 other_rank);

      Assert(ptr != potential_owners_ranks.end(), dealii::ExcInternalError());

      const auto other_rank_index =
        std::distance(potential_owners_ranks.begin(), ptr);

      return other_rank_index;
    };


    Assert(
      (marked_vertices.size() == 0) ||
        (marked_vertices.size() == tria_slave.n_vertices()),
      dealii::ExcMessage(
        "The marked_vertices vector has to be either empty or its size has "
        "to equal the number of vertices of the triangulation."));

    using RequestType = std::vector<
      std::pair<unsigned int,
                std::array<dealii::Point<dim>, MasterCellData::n_face_verts>>>;
    using AnswerType =
      std::vector<std::tuple<unsigned int,        // index of owning process
                             std::pair<int, int>, // cell level,index
                             std::array<dealii::Point<dim>, dim - 1 + 1>>>;

    const auto create_request = [&](const unsigned int other_rank) {
      const auto other_rank_index = translate(other_rank);

      RequestType request;
      request.reserve(potential_owners_ptrs[other_rank_index + 1] -
                      potential_owners_ptrs[other_rank_index]);

      for (unsigned int i = potential_owners_ptrs[other_rank_index];
           i < potential_owners_ptrs[other_rank_index + 1];
           ++i)
        request.emplace_back(
          potential_owners_indices[i],
          master_cell_data.master_simplex_buffer[potential_owners_indices[i]]);

      return request;
    };

    const auto answer_request = [&](const unsigned int &other_rank,
                                    const RequestType & request) -> AnswerType {
      AnswerType answer;

      for (unsigned int i = 0; i < request.size(); ++i)
        {
          const auto &index_and_simplex = request[i];

          unsigned int n_intersections = 0;

          for (const auto &slave_simplex : slave_simplex_buffer)
            {
              // slave_simplex_buffer only contains candidates with marked
              // vertices
              // TODO: would this be good: check first with bounding boxes if
              // intersection can happeng using RTree??

              const auto intersection_simplices =
                dealii::CGALWrappers::compute_intersection_of_cells<
                  dim,
                  dim - 1,
                  dim,
                  int(std::pow(2, dim)), // TODO: we assume hex only here
                  int(std::pow(2, dim - 1))>(slave_simplex.second,
                                             index_and_simplex.second,
                                             tolerance);

              for (const auto &intersection_simplex : intersection_simplices)
                answer.emplace_back(
                  index_and_simplex
                    .first,            // index referring to master cell//TODO:!
                  slave_simplex.first, //<-level and index of slave cell
                  intersection_simplex); //<-intersection
            }
        }

      return answer;
    };

    std::vector<std::pair<unsigned int, AnswerType>> answer_buffer;
    const auto process_answer = [&](const unsigned int other_rank,
                                    const AnswerType & answer) {
      answer_buffer.emplace_back(std::make_pair(other_rank, answer));
    };

    dealii::Utilities::MPI::ConsensusAlgorithms::selector<RequestType,
                                                          AnswerType>(
      potential_owners_ranks,
      create_request,
      answer_request,
      process_answer,
      tria_slave.get_communicator());

    // scrub infos which are not needed for the easy approach
    for (auto &ranks_answer : answer_buffer)
      {
        for (auto &ans : ranks_answer.second)
          {
            const auto &index   = std::get<0>(ans);
            const auto &face    = master_cell_data.face_indices[index];
            const auto &cell    = master_cell_data.cells[index];
            const auto &simplex = std::get<2>(ans);

            if (faces_to_simplices.find(face) == faces_to_simplices.end())
              {
                faces_to_simplices.insert({face, {simplex}});
                faces_to_cell.insert({face, cell});
              }
            else
              faces_to_simplices.at(face).emplace_back(simplex);
          }
      }
  }

  std::map<unsigned int, std::vector<std::array<dealii::Point<dim>, dim>>>
    faces_to_simplices;
  std::map<unsigned int, typename dealii::Triangulation<dim>::cell_iterator>
    faces_to_cell;

  dealii::Quadrature<dim>
  get_mapped_quadrule(const unsigned int face,
                      const unsigned int n_quadrature_pnts) const
  {
    return dealii::QGaussSimplex<dim - 1>(n_quadrature_pnts)
      .mapped_quadrature(faces_to_simplices.at(face));
  }

  std::vector<dealii::Point<dim>>
  get_quadrature_unit_points(const unsigned int face,
                             const unsigned int n_quadrature_pnts) const
  {
    std::vector<dealii::Point<dim>> points =
      get_mapped_quadrule(face, n_quadrature_pnts).get_points();
    for (auto &p : points)
      p =
        mapping_master_->transform_real_to_unit_cell(faces_to_cell.at(face), p);

    return points;
  }
};
// TODO:
// benchmark: using caches and trees
// look at extract_boundary_mesh for parallel distributed trias
// remote point eval doubles the work
// benchmark: add unit points and quadrature rules as members
