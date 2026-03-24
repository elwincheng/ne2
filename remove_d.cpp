#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace py = pybind11;

// Equivalent of your Python remove_extreme_D_average, but works on a std::vector<double>
static inline double remove_extreme_D_average_vec(std::vector<double>& vals,
                                                  double agent_value,
                                                  int D) {
    if (vals.empty()) return agent_value;

    std::sort(vals.begin(), vals.end());
    const int n = static_cast<int>(vals.size());

    int low_index = 0;
    int high_index = n - 1;

    const int d = std::min(D, n);

    // Remove up to D highest values that are >= agent_value
    for (int j_highest = 0; j_highest < d; ++j_highest) {
        const int idx = (n - 1) - j_highest;
        if (vals[idx] >= agent_value) {
            high_index -= 1;
        } else {
            break;
        }
    }

    // Remove up to D lowest values that are <= agent_value
    for (int j_lowest = 0; j_lowest < d; ++j_lowest) {
        if (vals[j_lowest] <= agent_value) {
            low_index += 1;
        } else {
            break;
        }
    }

    // If we trimmed everything away, return agent_value (average over just itself)
    if (high_index < low_index) return agent_value;

    double sum = agent_value;
    for (int idx = low_index; idx <= high_index; ++idx) {
        sum += vals[idx];
    }

    const int kept = (high_index - low_index + 1);
    const int denom = kept + 1; // +1 for agent_value
    return sum / static_cast<double>(denom);
}

static double median_sorted(const std::vector<double>& sorted) {
    const size_t n = sorted.size();
    if (n == 0) return 0.0;
    if (n % 2 == 1) return sorted[n / 2];
    return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

static double median_aggregate_vec(std::vector<double> vals, double agent_value) {
    vals.push_back(agent_value);
    std::sort(vals.begin(), vals.end());
    return median_sorted(vals);
}

/** median_window None -> pure median; "mad" via use_mad; else fixed window w. */
static double median_window_aggregate_vec(std::vector<double> vals,
                                          double agent_value,
                                          double median_window,
                                          bool use_mad,
                                          bool window_none) {
    vals.push_back(agent_value);
    std::vector<double> sorted = vals;
    std::sort(sorted.begin(), sorted.end());
    const double m = median_sorted(sorted);

    if (window_none && !use_mad) return m;

    double w = 0.0;
    if (use_mad) {
        std::vector<double> absdev;
        absdev.reserve(vals.size());
        for (double v : vals) absdev.push_back(std::abs(v - m));
        std::sort(absdev.begin(), absdev.end());
        const double mad = median_sorted(absdev);
        w = 1.4826 * mad + 1e-6;
    } else {
        w = median_window;
    }

    double sum = 0.0;
    int cnt = 0;
    for (double v : vals) {
        if (std::abs(v - m) <= w) {
            sum += v;
            ++cnt;
        }
    }
    if (cnt == 0) return m;
    return sum / static_cast<double>(cnt);
}

/**
 * Weiszfeld in R^2 — matches resilient.py module-level geometric_median (maxiter=50, tol=1e-6,
 * eps=1e-12): mean init, if min distance to iterate < eps return closest data point (np.argmin),
 * stop when ||y_next - y|| <= tol.
 */
static std::array<double, 2> geometric_median_2d(std::vector<std::array<double, 2>> pts) {
    const size_t n = pts.size();
    if (n == 0) return {0.0, 0.0};

    double cx = 0.0, cy = 0.0;
    for (const auto& p : pts) {
        cx += p[0];
        cy += p[1];
    }
    cx /= static_cast<double>(n);
    cy /= static_cast<double>(n);

    constexpr int kMaxIter = 50;
    constexpr double kTol = 1e-6;
    constexpr double kEps = 1e-12;

    for (int iter = 0; iter < kMaxIter; ++iter) {
        double min_d = std::numeric_limits<double>::infinity();
        size_t argmin_i = 0;
        for (size_t i = 0; i < n; ++i) {
            const double d = std::hypot(pts[i][0] - cx, pts[i][1] - cy);
            if (d < min_d) {
                min_d = d;
                argmin_i = i;
            }
        }
        if (min_d < kEps) {
            return pts[argmin_i];
        }

        double num_x = 0.0, num_y = 0.0, den = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double dx = pts[i][0] - cx;
            const double dy = pts[i][1] - cy;
            const double d = std::hypot(dx, dy);
            const double inv = 1.0 / std::max(d, kEps);
            num_x += pts[i][0] * inv;
            num_y += pts[i][1] * inv;
            den += inv;
        }
        const double nx = num_x / den;
        const double ny = num_y / den;
        if (std::hypot(nx - cx, ny - cy) <= kTol) {
            return {nx, ny};
        }
        cx = nx;
        cy = ny;
    }
    return {cx, cy};
}

/** Matches median.py robust_geom_median_2d (mean of in-window points; else center). */
static std::array<double, 2> robust_geom_median_2d(const std::vector<std::array<double, 2>>& pts,
                                                   double median_window,
                                                   bool use_mad) {
    if (pts.empty()) return {0.0, 0.0};

    std::array<double, 2> center = geometric_median_2d(pts);

    const size_t n = pts.size();
    std::vector<double> dists(n);
    for (size_t i = 0; i < n; ++i) {
        const double dx = pts[i][0] - center[0];
        const double dy = pts[i][1] - center[1];
        dists[i] = std::hypot(dx, dy);
    }

    std::vector<double> dists_sorted = dists;
    std::sort(dists_sorted.begin(), dists_sorted.end());
    const double med_d = median_sorted(dists_sorted);

    std::vector<char> keep(n, 0);
    if (use_mad) {
        std::vector<double> absdev;
        absdev.reserve(n);
        for (double d : dists) absdev.push_back(std::abs(d - med_d));
        std::sort(absdev.begin(), absdev.end());
        const double mad2 = median_sorted(absdev);
        const double w = 1.4826 * mad2 + 1e-12;
        for (size_t i = 0; i < n; ++i) {
            if (dists[i] <= med_d + w) keep[i] = 1;
        }
    } else {
        const double w = median_window;
        for (size_t i = 0; i < n; ++i) {
            if (dists[i] <= w) keep[i] = 1;
        }
    }

    double sx = 0.0, sy = 0.0;
    int kcnt = 0;
    for (size_t i = 0; i < n; ++i) {
        if (keep[i]) {
            sx += pts[i][0];
            sy += pts[i][1];
            ++kcnt;
        }
    }
    if (kcnt == 0) return center;
    return {sx / static_cast<double>(kcnt), sy / static_cast<double>(kcnt)};
}

/**
 * filter_communicated_message_cpp(state_y, Go, offsets, neighbors, D, ...)
 *
 * state_y: float64 ndarray shape (dim_state, N)
 * Go:      uint8/int32/bool ndarray shape (N, N) where Go[i,j] is truthy if direct trust
 * offsets: int32/int64 ndarray shape (N+1) CSR row offsets for adj_list_gc
 * neighbors: int32/int64 ndarray shape (E) flat neighbor list
 * D: int (trim only)
 * aggregation: 0 = trim, 1 = median, 2 = median_window
 * use_geometric_median: if true with aggregation==1, use 2D robust geometric median per (i,j)
 * median_window: fixed radius for 2D (non-MAD) or scalar median_window; NaN = None for agg 2
 * median_window_mad: use MAD rule (scalar agg 2 or 2D geom)
 *
 * Returns: float64 ndarray shape (dim_state, 1)
 */
py::array_t<double> filter_communicated_message_cpp(py::array state_y,
                                                    py::array Go,
                                                    py::array offsets,
                                                    py::array neighbors,
                                                    int D,
                                                    int aggregation,
                                                    bool use_geometric_median,
                                                    double median_window,
                                                    bool median_window_mad) {
    if (aggregation < 0 || aggregation > 2) {
        throw std::runtime_error("aggregation must be 0 (trim), 1 (median), or 2 (median_window)");
    }
    if (aggregation == 1 && use_geometric_median && !median_window_mad &&
        std::isnan(median_window)) {
        throw std::runtime_error(
            "median_window must be finite when use_geometric_median is true unless "
            "median_window_mad is true");
    }

    // ---- Validate state_y ----
    auto y = py::array_t<double, py::array::c_style | py::array::forcecast>(state_y);
    auto ybuf = y.request();
    if (ybuf.ndim != 2) throw std::runtime_error("state_y must be 2D (dim_state, N)");
    const int64_t dim_state = ybuf.shape[0];
    const int64_t N = ybuf.shape[1];
    if (N <= 0) throw std::runtime_error("N must be positive");

    // ---- Validate Go ----
    auto go_arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(Go);
    auto gobuf = go_arr.request();
    if (gobuf.ndim != 2) throw std::runtime_error("Go must be 2D (N, N)");
    if (gobuf.shape[0] != N || gobuf.shape[1] != N) throw std::runtime_error("Go shape must be (N, N)");

    // ---- Validate CSR adjacency ----
    auto off = py::array_t<int64_t, py::array::c_style | py::array::forcecast>(offsets);
    auto nbr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>(neighbors);
    auto offbuf = off.request();
    auto nbrbuf = nbr.request();
    if (offbuf.ndim != 1) throw std::runtime_error("offsets must be 1D (N+1,)");
    if (nbrbuf.ndim != 1) throw std::runtime_error("neighbors must be 1D (E,)");
    if (offbuf.shape[0] != N + 1) throw std::runtime_error("offsets length must be N+1");

    const auto* Y = static_cast<const double*>(ybuf.ptr);
    const auto* GO = static_cast<const uint8_t*>(gobuf.ptr);
    const auto* OFF = static_cast<const int64_t*>(offbuf.ptr);
    const auto* NBR = static_cast<const int64_t*>(nbrbuf.ptr);

    const int64_t E = nbrbuf.shape[0];
    if (OFF[0] != 0 || OFF[N] != E) {
        throw std::runtime_error("CSR offsets must start at 0 and end at len(neighbors)");
    }

    py::array_t<double> out(py::array::ShapeContainer{(py::ssize_t)dim_state, (py::ssize_t)1});

    auto outbuf = out.request();
    auto* V = static_cast<double*>(outbuf.ptr);

    const int64_t dim_action_i = 2;
    const int64_t dim_action = 2 * N;

    auto Y_at = [&](int64_t row, int64_t col) -> double { return Y[row * N + col]; };
    auto GO_at = [&](int64_t i, int64_t j) -> uint8_t { return GO[i * N + j]; };

    const bool window_none = std::isnan(median_window) && !median_window_mad;

    #pragma omp parallel for schedule(static)
    for (int64_t agent_i = 0; agent_i < N; ++agent_i) {
        const int64_t off0 = OFF[agent_i];
        const int64_t off1 = OFF[agent_i + 1];

        for (int64_t state_j = 0; state_j < N; ++state_j) {
            const int64_t base_offset = dim_action_i * state_j;

            if (aggregation == 1 && use_geometric_median) {
                const int64_t offset_x = base_offset + 0;
                const int64_t offset_y = base_offset + 1;
                const int64_t idx_i_x = dim_action * agent_i + offset_x;
                const int64_t idx_i_y = dim_action * agent_i + offset_y;

                if (GO_at(agent_i, state_j)) {
                    const int64_t idx_j_x = dim_action * state_j + offset_x;
                    const int64_t idx_j_y = dim_action * state_j + offset_y;
                    V[idx_i_x] = Y_at(idx_j_x, state_j);
                    V[idx_i_y] = Y_at(idx_j_y, state_j);
                } else {
                    std::vector<std::array<double, 2>> pts;
                    pts.reserve(static_cast<size_t>(off1 - off0 + 1));
                    for (int64_t p = off0; p < off1; ++p) {
                        const int64_t X = NBR[p];
                        const double px = Y_at(dim_action * X + offset_x, agent_i);
                        const double py = Y_at(dim_action * X + offset_y, agent_i);
                        pts.push_back({px, py});
                    }
                    pts.push_back({Y_at(idx_i_x, agent_i), Y_at(idx_i_y, agent_i)});
                    std::array<double, 2> agg =
                        robust_geom_median_2d(pts, median_window, median_window_mad);
                    V[idx_i_x] = agg[0];
                    V[idx_i_y] = agg[1];
                }
                continue;
            }

            for (int64_t component_k = 0; component_k < dim_action_i; ++component_k) {
                const int64_t offset = base_offset + component_k;
                const int64_t state_index_i = dim_action * agent_i + offset;

                if (GO_at(agent_i, state_j)) {
                    const int64_t state_index_j = dim_action * state_j + offset;
                    V[state_index_i] = Y_at(state_index_j, state_j);
                } else {
                    std::vector<double> vals;
                    vals.reserve(static_cast<size_t>(off1 - off0));

                    for (int64_t p = off0; p < off1; ++p) {
                        const int64_t X = NBR[p];
                        vals.push_back(Y_at(dim_action * X + offset, agent_i));
                    }

                    const double agent_value = Y_at(state_index_i, agent_i);

                    if (aggregation == 0) {
                        V[state_index_i] = remove_extreme_D_average_vec(vals, agent_value, D);
                    } else if (aggregation == 1) {
                        V[state_index_i] = median_aggregate_vec(std::move(vals), agent_value);
                    } else {
                        V[state_index_i] = median_window_aggregate_vec(
                            std::move(vals), agent_value, median_window, median_window_mad, window_none);
                    }
                }
            }
        }
    }

    return out;
}

PYBIND11_MODULE(remove_d, m) {
    m.doc() = "C++ bindings for robust filtering (filter_communicated_message)";
    m.def("filter_communicated_message",
          &filter_communicated_message_cpp,
          py::arg("state_y"),
          py::arg("Go"),
          py::arg("offsets"),
          py::arg("neighbors"),
          py::arg("D"),
          py::arg("aggregation") = 0,
          py::arg("use_geometric_median") = false,
          py::arg("median_window") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("median_window_mad") = false);
}
