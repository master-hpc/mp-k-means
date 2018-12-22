//
// Created by mastermind on 12/21/17.
//

#ifndef K_MEANS_CUDA_PIXEL_H
#define K_MEANS_CUDA_PIXEL_H

namespace app {
    namespace data_type {
        struct Color {

            Color();

            unsigned char r;
            unsigned char g;
            unsigned char b;

            Color(const unsigned char &r, const unsigned char &g, unsigned const char &b);

            bool operator==(const Color &other) const;

            bool operator!=(const Color &other) const;

            bool operator<(const Color &other) const;
        };
    }
}

#endif //K_MEANS_CUDA_PIXEL_H
