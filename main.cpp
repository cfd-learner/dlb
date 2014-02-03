#include <cmath>
#include <cstdio>
#include "lbm.hpp"

// grid size in x- and y-dimension
static const int lx = 600;
static const int ly = 300;

static const int t_max = 2;           // maximum number of iterations
int time_it = 0;             // iteration counter

// void write_distributions(bool stop = false) {
//   printf("# WRITING DISTRIBUTIONS AT TIME=%d\n", time_it);
//   for (auto c: cells.cell_ids()) {
//     for (auto n : cells.node_ids()) {
//       printf("x=%ld \t y=%d \t n=%ld \t f=%e \t fh=%e \n", cells.x(c),
//              cells.y(c), n, cells.nodes(c, n), cells.nodes_hlp(c, n));
//     }
//   }
//   if (stop) { std::terminate(); }
// }


int main() {

  lbm::data_structure cells(lx, ly);
  while(time_it < t_max) {
    ++time_it;
    lbm::Num d_sum = cells.advance(1);
    printf("# %d | integral density: %8.f\n", time_it, d_sum);

    write_vtk(cells, time_it);
  }

  return 0;
}
