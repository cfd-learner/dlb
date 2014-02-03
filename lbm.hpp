#ifndef LBM_HPP_
#define LBM_HPP_
////////////////////////////////////////////////////////////////////////////////
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/range.h>
////////////////////////////////////////////////////////////////////////////////

namespace lbm {

typedef double Num;
typedef Num& NumRef;
typedef long Idx;

template<class I> struct Range {
  typedef thrust::iterator_range<thrust::counting_iterator<I> > type;
};

typedef Range<Idx>::type CellRange;

struct cylinder {
  __host__ __device__
  cylinder(const Idx lx, const Idx ly)
      : x_c(lx / 2. - 0.2 * lx), y_c(ly / 2.), r(0.125 * ly) {}
  const Num x_c;
  const Num y_c;
  const Num r;
};

struct node_vars {};
struct hlp_vars  {};

struct device_data {
  const Idx lx;
  const Idx ly;
  const cylinder cyl;

  const Num density; // fluid density per link
  const Num c_squ;   // square speed-of-sound c^2 = 1/3
  const Num omega;   // relaxation parameter
  const Num accel;   // accelleration

  Num* nodes_;
  Num* nodes_hlp_;

  /// host-device kernels
  __host__ __device__ Idx n_size() const { return 9; }
  __host__ __device__ Idx x_size() const { return lx; }
  __host__ __device__ Idx y_size() const { return ly; }

  __host__ __device__ Idx no_cells() const { return x_size() * y_size(); }
  __host__ __device__ Idx no_nodes() const { return n_size() * no_cells(); }

  __host__ __device__
  device_data(const Idx size_x, const Idx size_y,
              thrust::device_vector<Num>& nodes,
              thrust::device_vector<Num>& nodes_hlp)
      : lx(size_x), ly(size_y), cyl(lx, ly)
      , density(0.1)
      , c_squ(1. / 3.)
      , omega(1.85)
      , accel(0.015)
      , nodes_(thrust::raw_pointer_cast(nodes.data()))
      , nodes_hlp_(thrust::raw_pointer_cast(nodes_hlp.data()))
  {}

  // device kernels
  __host__ __device__
  CellRange cell_ids() const {
    return thrust::make_counting_range<Idx>(0, no_cells());
  }
  __device__
  CellRange node_ids() const {
    return thrust::make_counting_range<Idx>(0, n_size());
  }

  __device__
  CellRange direct_node_ids() const
  { return thrust::make_counting_range<Idx>(1, 5); }

  __device__
  CellRange diagonal_node_ids() const
  { return thrust::make_counting_range<Idx>(5, 9); }

  __device__
  Idx opposite_node(const Idx n) const {
    const Idx opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    return opp[n];
  }

  __device__
  Idx cell_at(const Idx xIdx, const Idx yIdx) const {
    return xIdx + x_size() * yIdx;
  }

  __host__ __device__
  Idx x(      Idx cIdx) const {
    while (cIdx > lx - 1) { cIdx -= lx; }
    return cIdx;
  }

  __host__ __device__
  int y(const Idx cIdx) const { return cIdx / lx; }

  __device__
  Idx neighbor(const Idx c, const Idx n) const {
    const Idx x_i = x(c);
    const Idx y_i = y(c);
    // compute upper and right next neighbour nodes with regard to periodic
    // boundaries:
    const Idx y_n = (y_i + 1) % ly + 1 - 1;  // todo: recheck!
    const Idx x_e = (x_i + 1) % lx + 1 - 1;  // todo: recheck!
    // compute lower and left next neighbour nodes with regard to periodic
    // boundaries:
    const Idx y_s = ly - (ly + 1 - (y_i + 1)) % ly - 1;
    const Idx x_w = lx - (lx + 1 - (x_i + 1)) % lx - 1;

    const Idx nghbrs[9] = {
        c,
        cell_at(x_e, y_i),
        cell_at(x_i, y_n),
        cell_at(x_w, y_i),
        cell_at(x_i, y_s),
        cell_at(x_e, y_n),
        cell_at(x_w, y_n),
        cell_at(x_w, y_s),
        cell_at(x_e, y_s)
      };

    return nghbrs[n];
  }

  __device__
  NumRef vars(node_vars, const Idx cIdx, const Idx nIdx)
  { return nodes(cIdx, nIdx); }

  __device__
  Num vars(node_vars, const Idx cIdx, const Idx nIdx) const
  { return nodes(cIdx, nIdx); }

  __device__
  NumRef vars(hlp_vars, const Idx cIdx, const Idx nIdx)
  { return nodes_hlp(cIdx, nIdx); }

  __device__
  Num vars(hlp_vars, const Idx cIdx, const Idx nIdx) const
  { return nodes_hlp(cIdx, nIdx); }

  __device__
  NumRef nodes(const Idx cIdx, const Idx nIdx)
  { return nodes_[cIdx * n_size() + nIdx]; }

  __device__
  Num nodes(const Idx cIdx, const Idx nIdx) const
  { return nodes_[cIdx * n_size() + nIdx]; }

  __device__
  NumRef nodes_hlp(const Idx cIdx, const Idx nIdx)
  { return nodes_hlp_[cIdx * n_size() + nIdx]; }

  __device__
  Num nodes_hlp(const Idx cIdx, const Idx nIdx) const
  { return nodes_hlp_[cIdx * n_size() + nIdx]; }

  __device__
  NumRef nodes(const Idx nIdx, const Idx xIdx, const Idx yIdx)
  { return nodes_[idx(nIdx, xIdx, yIdx)]; }

  __device__
  NumRef nodes_hlp(const Idx nIdx, const Idx xIdx, const Idx yIdx)
  { return nodes_hlp_[idx(nIdx, xIdx, yIdx)]; }

  __device__
  Idx idx(const Idx nIdx, const Idx xIdx, const Idx yIdx)
  { return nIdx + n_size() * (xIdx + x_size() * yIdx); }

  __host__ __device__
  bool in_cylinder(const Idx xi, const Idx yi) const {
    return std::sqrt(std::pow(cyl.x_c - xi, 2.) + std::pow(cyl.y_c - yi, 2))
        - cyl.r < 0.;
  }

  __host__ __device__
  bool obst(const Idx x, const Idx y) const
  { return in_cylinder(x, y) || y == 0 || y == ly - 1; }

  __host__ __device__
  bool obst(const Idx cIdx) const { return obst(x(cIdx), y(cIdx)); }
};

namespace kernel {


template<class T>
struct density : thrust::unary_function<Idx, Num> {
  device_data d;

  __host__ __device__
  density(device_data data) : d(data) {}

  __device__
  Num operator()(const Idx c) const {
    Num tmp = 0;
    typedef thrust::counting_iterator<Idx> It;
    for (It i = d.node_ids().begin(), e = d.node_ids().end(); i != e; ++i) {
      tmp += d.vars(T(), c, *i);
    }
    return tmp;
  }
};

template<class T> struct pressure : thrust::unary_function<Idx, Num> {
  device_data d;

  __host__ __device__
  pressure(device_data data) : d(data) {}
  __device__ Num operator()(const Idx c) const {
    if (d.obst(c)) {
      return d.density * d.c_squ;
    } else {
      return density<T>(d)(c) * d.c_squ;
    }
  }
};

template<class T> struct u : thrust::unary_function<Idx, Num> {
  device_data d;

  __host__ __device__
  u(device_data data) : d(data) {}

  __device__
  Num operator()(const Idx c) const {
    if (d.obst(c)) {
      return 0.;
    } else {
      return (d.vars(T(), c, 1) + d.vars(T(), c, 5) + d.vars(T(), c, 8)
              -(d.vars(T(), c, 3) + d.vars(T(), c, 6) + d.vars(T(), c, 7)))
          / density<T>(d)(c);
    }
  }
};

template<class T> struct v : thrust::unary_function<Idx, Num> {
  device_data d;

  __host__ __device__
  v(device_data data) : d(data) {}

  __device__
  Num operator()(const Idx c) const {
    if (d.obst(c)) {
      return 0.;
    } else {
      return (d.vars(T(), c, 2) + d.vars(T(), c, 5) + d.vars(T(), c, 6)
              -(d.vars(T(), c, 4) + d.vars(T(), c, 7) + d.vars(T(), c, 8)))
          / density<T>(d)(c);
    }
  }
};

struct init_variables {
  device_data d;
  const Num t_0;
  const Num t_1;
  const Num t_2;

  __host__ __device__
  init_variables(device_data data)
      : d(data)
      , t_0(d.density * 4. / 9)
      , t_1(d.density / 9.)
      , t_2(d.density / 36.) {}

  __device__
  void operator()(const Idx c) {
    d.nodes(c, 0) = t_0;  // zero velocity density
    for (thrust::counting_iterator<Idx> bIt = d.direct_node_ids().begin(),
                                        eIt = d.direct_node_ids().end();
         bIt != eIt; ++bIt)
    {
      const Idx i = *bIt;
      d.nodes(c, i) = t_1;  // equilibrium densities for axis speeds
    }
    for (thrust::counting_iterator<Idx> bIt = d.diagonal_node_ids().begin(),
                                        eIt = d.diagonal_node_ids().end();
         bIt != eIt; ++bIt)
    {
      const Idx i = *bIt;
      d.nodes(c, i) = t_2;  // equilibrium densities for diagonal speeds
    }
  }
};

struct redistribute {
  device_data d;
  const Num t_1;
  const Num t_2;

  __host__ __device__
  redistribute(device_data data)
      : d(data)
      , t_1(d.density * d.accel / 9.)
      , t_2(d.density * d.accel / 36.)
  {}

  __device__
  void operator()(const Idx c) {
    const Idx xi = d.x(c);
    if (xi != 0) { return; }
    const Idx yi = d.y(c);
    if (!d.obst(0,yi)  // accelerate flow only on non-occupied nodes
        // check to avoid negative densities:
        && d.nodes(3, 0, yi) - t_1 > 0.
        && d.nodes(6, 0, yi) - t_2 > 0.
        && d.nodes(7, 0, yi) - t_2 > 0.
        ) {
      d.nodes(1, 0, yi) += t_1;  // increase east
      d.nodes(5, 0, yi) += t_2;  // increase north-east
      d.nodes(8, 0, yi) += t_2;  // increase south-east

      d.nodes(3, 0, yi) -= t_1;  // decrease west
      d.nodes(6, 0, yi) -= t_2;  // decrease north-west
      d.nodes(7, 0, yi) -= t_2;  // decrease south-west
    }
  }
};

struct propagate {
  device_data d;

  __host__ __device__
  propagate(device_data data) : d(data) {}

  __device__
  void operator()(const Idx c) {
    for (thrust::counting_iterator<Idx> bIt = d.node_ids().begin(),
                                        eIt = d.node_ids().end();
         bIt != eIt; ++bIt)
    {
      const Idx n = *bIt;
      d.nodes_hlp(d.neighbor(c, n), n) = d.nodes(c, n);
    }
  }
};

struct bounceback {
  device_data d;

  __host__ __device__
  bounceback(device_data data) : d(data) {}

  __device__
  void operator()(const Idx c) {
    if (d.obst(c)) {
      for (thrust::counting_iterator<Idx> bIt = d.node_ids().begin(),
                                          eIt = d.node_ids().end();
           bIt != eIt; ++bIt)
      {
        const Idx n = *bIt;
        d.nodes(c, n) = d.nodes_hlp(c, d.opposite_node(n));
      }
    }
  }
};

struct relaxation {
  device_data d;
  const Num c_squ;
  const Num omega;
  const Num t_0;
  const Num t_1;
  const Num t_2;

  __host__ __device__
  relaxation(device_data data)
        : d(data)
        , c_squ(d.c_squ)
        , omega(d.omega)
        , t_0(4. / 9.)
        , t_1(1. /  9.)
        , t_2(1. / 36.)
  {}

  __device__
  void operator()(const Idx c) {
    if (d.obst(c)) { return; }

    // integral local density:
    const Num dloc = density<hlp_vars>(d)(c);

    // x-, and y- velocity components:
    const Num u_x = u<hlp_vars>(d)(c);
    const Num u_y = v<hlp_vars>(d)(c);

    // n- velocity compnents (n = lattice node connection vectors)
    // this is only necessary for clearence, and only 3 speeds would
    // be necessary
    const Num sx[9] = { 0., 1., 0., -1., 0., 1., -1., -1., 1.};
    const Num sy[9] = { 0., 0., 1., 0., -1., 1., 1., -1., -1.};
    Num u_n[9];
    for (thrust::counting_iterator<Idx> bIt = d.node_ids().begin(),
                                        eIt = d.node_ids().end();
         bIt != eIt; ++bIt)
    {
      const Idx i = *bIt;
      u_n[i] = sx[i] * u_x + sy[i] * u_y;
    }

    // equilibrium densities:
    // (this can be rewritten to improve computational performance
    // considerabely !)
    const Num f0 = 2. * c_squ * c_squ;
    const Num f1 = (2. * c_squ);
    const Num u_squ = std::pow(u_x, 2.) + std::pow(u_y, 2.); // square velocity
    const Num f2 = u_squ / f1;
    const Num f3 = t_1 * dloc;
    const Num f4 = t_2 * dloc;

    Num n_equ[9];
    n_equ[0] = t_0 * dloc * (1. - f2);  // zero velocity density

    // axis speeds (factor: t_1)
    for (thrust::counting_iterator<Idx> bIt = d.direct_node_ids().begin(),
                                        eIt = d.direct_node_ids().end();
         bIt != eIt; ++bIt)
    {
      const Idx i = *bIt;

      n_equ[i] = f3 * (1. + u_n[i] / c_squ + std::pow(u_n[i], 2.) / f0 - f2);
    }

    // diagonal speeds (factor: t_2)
    for (thrust::counting_iterator<Idx> bIt = d.diagonal_node_ids().begin(),
                                        eIt = d.diagonal_node_ids().end();
         bIt != eIt; ++bIt)
    {
      const Idx i = *bIt;

      n_equ[i] = f4 * (1. + u_n[i] / c_squ + std::pow(u_n[i], 2.) / f0 - f2);
    }

    // relaxation step
    for (thrust::counting_iterator<Idx> bIt = d.node_ids().begin(),
                                        eIt = d.node_ids().end();
         bIt != eIt; ++bIt)
    {
      const Idx n = *bIt;

      d.nodes(c, n) = d.nodes_hlp(c, n)
                       + omega * (n_equ[n] - d.nodes_hlp(c, n));
    }
  }
};

}  // namespace kernel


struct data_structure {
  thrust::device_vector<Num> nodes;
  thrust::device_vector<Num> nodes_hlp;
  device_data data_;

  __host__
  data_structure(const Idx lx, const Idx ly)
      : nodes(lx * ly * 9)
      , nodes_hlp(lx * ly * 9)
      , data_(lx, ly, nodes, nodes_hlp)
  { init_variables(); }

  // Initialize density distribution function n with equilibrium
  // for zero velocity
  __host__ void init_variables()
  { thrust::for_each(data_.cell_ids(), kernel::init_variables(data_)); }

  // density redistribution in first lattice column
  // inlet boundary condition?
  __host__ void redistribute()
  { thrust::for_each(data_.cell_ids(), kernel::redistribute(data_)); }

  // Propagate fluid densities to their next neighbour nodes
  // streaming operator?
  __host__ void propagate()
  { thrust::for_each(data_.cell_ids(), kernel::propagate(data_)); }

  // Fluid densities are rotated by the next propagation step, this results in a
  // bounce back from cells.obstacle nodes.
  // solid-wall condition?
  __host__ void bounceback()
  { thrust::for_each(data_.cell_ids(), kernel::bounceback(data_)); }

  // One-step density relaxation process
  __host__ void relaxation()
  { thrust::for_each(data_.cell_ids(), kernel::relaxation(data_)); }

  // Compute integral density (should remain constant)
  __host__ Num check_density() {
    Num n_sum = thrust::transform_reduce
                (data_.cell_ids(), kernel::density<node_vars>(data_));
    return n_sum;
  }

  __host__ Num advance(Idx no_iters) {
    for(;;) {
      redistribute();
      propagate();
      bounceback();
      relaxation();
      const Num d = check_density();
      --no_iters;
      if (no_iters == 0) {
        return d;
      }
    }
  }

  /// Output functions
  template<class T>
  __host__
  thrust::host_vector<Num> ps(T) const {
    thrust::device_vector<Num> ps_(data_.no_cells());
    return thrust::transform(data_.cell_ids(), ps_, kernel::pressure<T>(data_));
  }

  template<class T>
  __host__
  thrust::host_vector<Num> us(T) const {
    thrust::device_vector<Num> ps_(data_.no_cells());
    return thrust::transform(data_.cell_ids(), ps_, kernel::u<T>(data_));
  }

  template<class T>
  __host__
  thrust::host_vector<Num> vs(T) const {
    thrust::device_vector<Num> ps_(data_.no_cells());
    return thrust::transform(data_.cell_ids(), ps_, kernel::v<T>(data_));
  }
};

__host__
void write_vtk(const data_structure& cs, int step) {
  char fname[100];
  sprintf(fname, "output_%d.vtk", step);
  const Idx noCells = cs.data_.no_cells();

  FILE *f = fopen(fname, "w");
  fprintf(f, "# vtk DataFile Version 2.0\n");
  fprintf(f, "LBM test output\n");
  fprintf(f, "ASCII\n");
  fprintf(f, "DATASET UNSTRUCTURED_GRID\n");

  Num x_stencil[4] = { -1, 1, -1, 1};
  Num y_stencil[4] = { -1, -1, 1, 1};
  Num length = 1;
  fprintf(f, "POINTS %ld FLOAT\n", noCells * 4);
  for(thrust::counting_iterator<Idx> it = cs.data_.cell_ids().begin(),
        it_e = cs.data_.cell_ids().end();
      it != it_e; ++it) {
    const Idx i = *it;
    Num x = static_cast<Num>(cs.data_.x(i));
    Num y = static_cast<Num>(cs.data_.y(i));
    for (int j = 0; j < 4; ++j) {
      Num xp = x + x_stencil[j] * length / 2;
      Num yp = y + y_stencil[j] * length / 2;
      fprintf(f, "%f %f 0.0\n", xp, yp);
    }
  }

  fprintf(f, "CELLS %ld %ld\n", noCells, noCells*5);
  for(thrust::counting_iterator<Idx> it = cs.data_.cell_ids().begin(),
        it_e = cs.data_.cell_ids().end();
      it != it_e; ++it) {
    const Idx i = *it;
    fprintf(f, "4 %ld %ld %ld %ld\n", 4*i, 4*i+1, 4*i+2, 4*i+3);
  }

  fprintf(f, "CELL_TYPES %ld\n", noCells);
  {
    typedef thrust::counting_iterator<Idx> It;
    for (It i = cs.data_.cell_ids().begin(),
            e = cs.data_.cell_ids().end(); i != e; ++i) {
      {
        fprintf(f, "8\n");
      }
    }
  }

  fprintf(f, "CELL_DATA %ld\n", noCells);

  fprintf(f, "SCALARS p float\n");
  fprintf(f, "LOOKUP_TABLE default\n");
  {
    thrust::host_vector<Num> ps = cs.ps(node_vars());
    for(thrust::counting_iterator<Idx> it = cs.data_.cell_ids().begin(),
          it_e = cs.data_.cell_ids().end();
        it != it_e; ++it) {
      const Idx i = *it;
      fprintf(f, "%f\n", ps[i]);
    }
  }

  fprintf(f, "SCALARS u float\n");
  fprintf(f, "LOOKUP_TABLE default\n");
  {
    thrust::host_vector<Num> us = cs.us(node_vars());
    for(thrust::counting_iterator<Idx> it = cs.data_.cell_ids().begin(),
          it_e = cs.data_.cell_ids().end();
        it != it_e; ++it) {
      const Idx i = *it;
      fprintf(f, "%f\n", us[i]);
    }
  }

  fprintf(f, "SCALARS v float\n");
  fprintf(f, "LOOKUP_TABLE default\n");
  {
    thrust::host_vector<Num> vs = cs.vs(node_vars());
    for(thrust::counting_iterator<Idx> it = cs.data_.cell_ids().begin(),
          it_e = cs.data_.cell_ids().end();
        it != it_e; ++it) {
      const Idx i = *it;
      fprintf(f, "%f\n", vs[i]);
    }
  }

  fprintf(f, "SCALARS active int\n");
  fprintf(f, "LOOKUP_TABLE default\n");
  for(thrust::counting_iterator<Idx> it = cs.data_.cell_ids().begin(),
        it_e = cs.data_.cell_ids().end();
      it != it_e; ++it) {
    const Idx i = *it;
    fprintf(f, "%d\n", !cs.data_.obst(i));
  }

  fclose(f);
}



}  // namespace lbm

////////////////////////////////////////////////////////////////////////////////
#endif  // LBM_HPP_
