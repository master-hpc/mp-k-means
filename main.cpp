#include <iostream>
#include <zconf.h>
#include <cassert>
#include "external/bitmap/bitmap_image.hpp"
#include "external/utils.h"
#include "k_means_cuda.h"

#define INPUT_FILE_SMALL "data/input_small.bmp"
#define INPUT_FILE_LARGE "data/input_large.bmp"
#define OUTPUT_FILE_SMALL "data/output_small.bmp"
#define OUTPUT_FILE_LARGE "data/output_large.bmp"
#define ERR 1

int main(int argc, char **argv) {

    // Vérifier si le matériel peut utiliser le GPU.
    if (!app::utils::is_cuda_capable())
        std::cout << "This hardware is not CUDA capable." << std::endl;

    // Imprimer les informations relatives au GPU(s) disponnible(s).
    std::cout << "Printing device information" << std::endl;
    app::utils::print_device_information();

    std::string cwd;
    // Récupération du cwd (Current Working Directory)
    app::utils::get_cwd(cwd);

    std::cout << "Enter" << std::endl
              << "s/S                    : For small image segmentation." << std::endl
              << "l/L (or any other key) : For large image segmentation." << std::endl;

    std::string key;
    std::cin >> key;

    std::string input_file;
    std::string output_file;

    if (key == "s" || key == "S") {
        input_file = cwd + "/" + INPUT_FILE_SMALL;
        output_file = cwd + "/" + OUTPUT_FILE_SMALL;
    } else {
        input_file = cwd + "/" + INPUT_FILE_LARGE;
        output_file = cwd + "/" + OUTPUT_FILE_LARGE;
    }


    bitmap_image image(input_file);
    if (!image) {
        // Vérifier que l'image a bien été importée
        std::cout << "Error - Failed to open: " << input_file << std::endl;
        return ERR;
    }
    unsigned int n = image.height() * image.width();
    std::cout << "Successfully imported image. - " << input_file << std::endl
              << "Number of loaded pixels 'n: " << n << std::endl;

    unsigned int k;
    // Lecture du K
    std::cout << "Enter k" << std::endl;
    std::cin >> k;
    std::cout << "Segmenting image with 'k: " << k << std::endl;
    // ALlocation du vecteur de couleur
    app::data_type::Color *color_array = (app::data_type::Color *) malloc(sizeof(app::data_type::Color) * n);
    assert(color_array != NULL);
    // Conversion de l'image en vecteur de couleur
    app::utils::from_image_to_color_array(image, color_array);
    // Allocation du vecteur des k_couleurs moyennes (les k-means)
    app::data_type::Color *k_colors = (app::data_type::Color *) malloc(sizeof(app::data_type::Color) * k);
    assert(k_colors != NULL);
    app::utils::init_k_colors(k, k_colors, color_array, n);
    try {
        app::k_means_cuda::run(k_colors, k, color_array, n);
    } catch (std::exception &e) {
        // En cas d'erreur, imprimer un message
        std::cout << "Err - " << e.what() << std::endl;
        return ERR;
    }
    app::utils::from_color_array_to_image(color_array, image);
    // Libération de la mémoire allouée
    free(color_array);
    free(k_colors);
    // Sauvegarde de l'image segmentée
    image.save_image(output_file);
    std::cout << "Successfully exported image. - " << output_file << std::endl;
    return 0;
}
