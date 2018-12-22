//
// Created by mastermind on 12/25/17.
//

#include <cassert>
#include <set>
#include <cuda_runtime_api.h>
#include <zconf.h>
#include "utils.h"

#define STRING_BUFFER 1024

namespace app {
    namespace utils {
        
        bool is_cuda_capable() {
            int n_devices = 0;
            cudaGetDeviceCount(&n_devices);
            return n_devices > 0;
        }

        void print_device_information() {
            // Source: https://devblogs.nvidia.com/parallelforall/how-query-device-properties-and-handle-errors-cuda-cc/
            int n_devices = 0;
            cudaGetDeviceCount(&n_devices);
            std::cout << "n_devices: " << n_devices << std::endl;
            for (int i = 0; i < n_devices; i++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                printf("Device Number: %d\n", i);
                printf("  Device name: %s\n", prop.name);
                printf("  Memory Clock Rate (KHz): %d\n",
                       prop.memoryClockRate);
                printf("  Memory Bus Width (bits): %d\n",
                       prop.memoryBusWidth);
                printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
                       2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
            }
        }

        void get_cwd(std::string &cwd) {
            char buffer[STRING_BUFFER];
            assert(getcwd(buffer, sizeof(buffer)) != NULL);
            cwd = buffer;
        }

        void from_image_to_color_array(const bitmap_image &image, data_type::Color *color_array) {
            for (unsigned int y = 0; y < image.height(); y++)
                for (unsigned int x = 0; x < image.width(); x++) {
                    image.get_pixel(x, y,
                                    color_array[x + y * image.width()].r,
                                    color_array[x + y * image.width()].g,
                                    color_array[x + y * image.width()].b);
                }
        }

        void from_color_array_to_image(data_type::Color *color_array, bitmap_image &image) {
            for (unsigned int y = 0; y < image.height(); y++)
                for (unsigned int x = 0; x < image.width(); x++) {
                    image.set_pixel(x, y,
                                    color_array[x + y * image.width()].r,
                                    color_array[x + y * image.width()].g,
                                    color_array[x + y * image.width()].b);
                }
        }

        void init_k_colors(const unsigned int &k, data_type::Color *k_colors, const data_type::Color color_array[],
                           const unsigned int &n, const double &seed) {
            assert(k <= n);
            srand(DEFAULT_SEED);
            std::set<app::data_type::Color> picked_colors;
            while (picked_colors.size() < k) {
                // FIXME: In rare cases, this might run_on_gpu forever...
                unsigned int random_idx = rand() % n;
                app::data_type::Color random_color = color_array[random_idx];
                if (picked_colors.find(random_color) == picked_colors.end()) {
                    picked_colors.insert(random_color);
                }
            }

            unsigned int i = 0;
            for (std::set<app::data_type::Color>::iterator color_iterator = picked_colors.begin();
                 color_iterator != picked_colors.end(); color_iterator++) {
                k_colors[i] = *color_iterator;
                i++;
            }
        }
    }
}
