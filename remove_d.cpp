#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
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

/**
 * filter_communicated_message_cpp(state_y, Go, offsets, neighbors, D)
 *
 * state_y: float64 ndarray shape (dim_state, N)
 * Go:      uint8/int32/bool ndarray shape (N, N) where Go[i,j] is truthy if direct trust
 * offsets: int32/int64 ndarray shape (N+1) CSR row offsets for adj_list_gc
 * neighbors: int32/int64 ndarray shape (E) flat neighbor list
 * D: int
 *
 * Returns: float64 ndarray shape (dim_state, 1)
 */
py::array_t<double> filter_communicated_message_cpp(py::array state_y,
                                                    py::array Go,
                                                    py::array offsets,
                                                    py::array neighbors,
                                                    int D) {
    // ---- Validate state_y ----
    auto y = py::array_t<double, py::array::c_style | py::array::forcecast>(state_y);
    auto ybuf = y.request();
    if (ybuf.ndim != 2) throw std::runtime_error("state_y must be 2D (dim_state, N)");
    const int64_t dim_state = ybuf.shape[0];
    const int64_t N = ybuf.shape[1];
    if (N <= 0) throw std::runtime_error("N must be positive");

    // dim_state should be 2*N*N (since dim_action_i=2 and dim_action=2N)
    // We won't hard-fail if mismatch, but it's usually a bug.
    // const int64_t expected = 2 * N * N;

    // ---- Validate Go ----
    // Accept bool, uint8, int32, int64; we’ll read as uint8 via forcecast.
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

    const auto* Y  = static_cast<const double*>(ybuf.ptr);
    const auto* GO = static_cast<const uint8_t*>(gobuf.ptr);
    const auto* OFF = static_cast<const int64_t*>(offbuf.ptr);
    const auto* NBR = static_cast<const int64_t*>(nbrbuf.ptr);

    const int64_t E = nbrbuf.shape[0];
    if (OFF[0] != 0 || OFF[N] != E) {
        throw std::runtime_error("CSR offsets must start at 0 and end at len(neighbors)");
    }

    // Output: (dim_state, 1)
    py::array_t<double> out(py::array::ShapeContainer{(py::ssize_t)dim_state, (py::ssize_t)1});

    auto outbuf = out.request();
    auto* V = static_cast<double*>(outbuf.ptr);

    // Constants derived from your layout
    const int64_t dim_action_i = 2;
    const int64_t dim_action   = 2 * N;      // per-agent block length

    // Helper lambdas for indexing into flattened row-major arrays:
    // state_y is (dim_state, N) in C-order => row-major: idx = row*N + col
    auto Y_at = [&](int64_t row, int64_t col) -> double {
        return Y[row * N + col];
    };
    auto GO_at = [&](int64_t i, int64_t j) -> uint8_t {
        return GO[i * N + j];
    };

    // Main loops: receiver agent_i, target state_j, component_k
    // Writes into V at row=state_index_i, col=0
    #pragma omp parallel for schedule(static)
    for (int64_t agent_i = 0; agent_i < N; ++agent_i) {
        const int64_t off0 = OFF[agent_i];
        const int64_t off1 = OFF[agent_i + 1];

        for (int64_t state_j = 0; state_j < N; ++state_j) {
            const int64_t base_offset = dim_action_i * state_j;

            for (int64_t component_k = 0; component_k < dim_action_i; ++component_k) {
                const int64_t offset = base_offset + component_k;
                const int64_t state_index_i = dim_action * agent_i + offset;

                if (GO_at(agent_i, state_j)) {
                    // Trust j's self value:
                    const int64_t state_index_j = dim_action * state_j + offset;
                    V[state_index_i] = Y_at(state_index_j, state_j);
                } else {
                    // Gather neighbor messages about (state_j, component_k) received by agent_i
                    std::vector<double> vals;
                    vals.reserve(static_cast<size_t>(off1 - off0));

                    for (int64_t p = off0; p < off1; ++p) {
                        const int64_t X = NBR[p]; // neighbor id
                        // row = dim_action*X + offset, col = agent_i
                        vals.push_back(Y_at(dim_action * X + offset, agent_i));
                    }

                    // agent's own value is at row=state_index_i, col=agent_i
                    const double agent_value = Y_at(state_index_i, agent_i);

                    const double filtered = remove_extreme_D_average_vec(vals, agent_value, D);
                    V[state_index_i] = filtered;
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
          py::arg("D"));
}
