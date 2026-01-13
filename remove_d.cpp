#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // enables std::vector <-> Python list conversion
#include <algorithm>
#include <vector>

namespace py = pybind11;

// C++ version of remove_extreme_D_average
double remove_extreme_D_average(std::vector<double> received_values,
                                double agent_value,
                                int D) {
    // Sort in-place (we take received_values by value so Python input isn't mutated)
    std::sort(received_values.begin(), received_values.end());

    int n = static_cast<int>(received_values.size());
    if (n == 0) {
        // If no neighbor values, just return agent_value (matches "average starts at agent_value")
        return agent_value;
    }

    int low_index = 0;
    int high_index = n - 1;

    int d = std::min(D, n);

    // Remove up to D highest values that are >= agent_value
    for (int j_highest = 0; j_highest < d; ++j_highest) {
        int idx = (n - 1) - j_highest;
        if (received_values[idx] >= agent_value) {
            high_index -= 1;
        } else {
            break;
        }
    }

    // Remove up to D lowest values that are <= agent_value
    for (int j_lowest = 0; j_lowest < d; ++j_lowest) {
        if (received_values[j_lowest] <= agent_value) {
            low_index += 1;
        } else {
            break;
        }
    }

    // Sum kept values + agent_value
    double sum = agent_value;
    for (int idx = low_index; idx <= high_index; ++idx) {
        sum += received_values[idx];
    }

    // Denominator = (#kept neighbor values) + 1 for agent_value
    int kept = (high_index - low_index + 1);
    int denom = kept + 1;

    // Edge case: if trimming removed everything, kept could be 0 (still fine)
    if (kept < 0) {
        // everything removed => only agent_value remains
        return agent_value;
    }

    return sum / static_cast<double>(denom);
}

PYBIND11_MODULE(remove_d, m) {
    m.doc() = "C++ bindings for resilient functions";
    m.def("remove_extreme_D_average",
          &remove_extreme_D_average,
          py::arg("received_values"),
          py::arg("agent_value"),
          py::arg("D"),
          "Robust trimmed average: drop up to D lows <= agent_value and D highs >= agent_value, then average with agent_value.");
}


