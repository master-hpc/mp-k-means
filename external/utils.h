//
// Created by mastermind on 12/25/17.
//

#ifndef K_MEANS_CUDA_UTILS_H
#define K_MEANS_CUDA_UTILS_H

#include "bitmap/bitmap_image.hpp"
#include "Color.h"

#define DEFAULT_SEED 1994

namespace app {
    namespace utils {

        bool is_cuda_capable();

        void print_device_information();

        void get_cwd(std::string &cwd);

        void from_image_to_color_array(const bitmap_image &, data_type::Color *);

        void from_color_array_to_image(data_type::Color *color_array, bitmap_image &image);

        void init_k_colors(const unsigned int &k, app::data_type::Color k_colors[],
                           const data_type::Color color_array[], const unsigned int &n,
                           const double &seed = DEFAULT_SEED);
    }
}


#endif //K_MEANS_CUDA_UTILS_H
