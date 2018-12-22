//
// Created by mastermind on 12/21/17.
//

#include <cmath>
#include "Color.h"

namespace app {
    namespace data_type {
        Color::Color() : r(0), g(0), b(0) {}
        Color::Color(const unsigned char &r, unsigned const char &g, const unsigned char &b) : r(r), g(g), b(b) {
        }
        bool Color::operator==(const Color &other) const {
            return r == other.r &&
                   g == other.g &&
                   b == other.b;
        }
        bool Color::operator!=(const Color &other) const {
            return r != other.r ||
                   g != other.g ||
                   b != other.b;
        }
        bool Color::operator<(const Color &other) const {
            return r * r + g * g + b * b < other.r * other.r + other.g * other.g + other.b * other.b;
        }
    }
}
