#include <algorithm>
#include <torch/extension.h>
#include "const.h"

static inline bool in_range(int r, int c, int nrows, int ncols) {
    return r >= 0 && r < nrows && c >= 0 && c < ncols;
}

static const std::pair<int, int> chessboard_structure[8] = {
    {-1, -1},
    {-1,  0},
    {-1, +1},
    { 0, -1},
    { 0, +1},
    {+1, -1},
    {+1,  0},
    {+1, +1},
};

static void init(const torch::Tensor &input, torch::Tensor &output) {
    auto in_acc = input.accessor<float, 4>();
    auto out_acc = output.accessor<float, 4>();

    const int64_t batch_size = output.size(0);
    const int64_t nrows = output.size(2);
    const int64_t ncols = output.size(3);

#pragma omp parallel for collapse(3)
    for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t r = 0; r < nrows; r++) {
            for (int64_t c = 0; c < ncols; c++) {
                if (in_acc[i][0][r][c] == 0) {
                    out_acc[i][0][r][c] = 0;
                } else {
                    out_acc[i][0][r][c] = INF;
                }
            }
        }
    }
}

template<typename T>
static T nbhood_minimum(
        const torch::TensorAccessor<T, 4> &X,
        int64_t i,
        int64_t r,
        int64_t c) {
    const auto nrows = X.size(2);
    const auto ncols = X.size(3);

    auto min_val = X[i][0][r][c];
    for (const auto &offset : chessboard_structure) {
        const auto off_row = r + offset.first;
        const auto off_col = c + offset.second;
        if (in_range(off_row, off_col, nrows, ncols)) {
            const auto off_val = X[i][0][off_row][off_col];
            min_val = std::min(min_val, off_val + 1);
        } }
    return min_val;
}

template<bool forward>
static void loop(torch::Tensor &output) {
    auto accessor = output.accessor<float, 4>();

    const int64_t batch_size = output.size(0);
    const int64_t nrows = output.size(2);
    const int64_t ncols = output.size(3);

    int64_t start_row, start_col, dir;
    if constexpr (forward) {
        start_row = 0;
        start_col = 0;
        dir = +1;
    } else {
        start_row = nrows - 1;
        start_col = ncols - 1;
        dir = -1;
    }

#pragma omp parallel for
    for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t r = start_row; r != start_row + dir * nrows; r += dir) {
            for (int64_t c = start_col; c != start_col + dir * ncols; c += dir) {
                accessor[i][0][r][c] = nbhood_minimum(accessor, i, r, c);
            }
        }
    }
}

torch::Tensor chdt_cpu(const torch::Tensor &input) {
    TORCH_CHECK(input.device().is_cpu(), input, " must be a CPU tensor");
    TORCH_CHECK(input.is_contiguous(), input, " must be contiguous");

    auto D = torch::zeros_like(input);
    init(input, D);
    loop<true>(D);
    loop<false>(D);
    return D;
}

