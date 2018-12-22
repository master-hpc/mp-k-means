//
// Created by mastermind on 12/25/17.
//

#ifndef K_MEANS_CUDA_KMEANS_CUDA_H
#define K_MEANS_CUDA_KMEANS_CUDA_H

#include "external/Color.h"

namespace app {
    namespace k_means_cuda {
        void run(data_type::Color k_colors[], const unsigned int &k, data_type::Color color_array[],
                 const unsigned int &n);
    }
}

#endif //K_MEANS_CUDA_KMEANS_CUDA_H
