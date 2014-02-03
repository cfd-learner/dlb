#include <cmath>
#include <cstdio>
#include "lbm.hpp"

// grid size in x- and y-dimension
static const int lx = 600;
static const int ly = 300;

static const int t_max = 10001;  // maximum number of iterations
int time_it = 0;             // iteration counter

int main() {

  lbm::data_structure cells(lx, ly);
  while(time_it < t_max) {
    ++time_it;
    const lbm::Num d_sum = cells.advance(1);
    printf("# %d | integral density: %8.f\n", time_it, d_sum);
    if (time_it == 10000) {
      write_vtk(cells, time_it);
    }
  }

  return 0;
}
