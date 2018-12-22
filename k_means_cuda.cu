//
// Created by mastermind on 12/25/17.
//

#include <iostream>
#include <cassert>
#include "k_means_cuda.h"
#include "external/utils.h"

namespace app {
    namespace k_means_cuda {
        using namespace app;
        using namespace app::data_type;

        __global__ void run_on_gpu(Color *k_colors, unsigned int k, Color *color_array, unsigned int n,
                                   unsigned int *classification_array, unsigned int *k_cardinals,
                                   unsigned int *red_sums,
                                   unsigned int *green_sums,
                                   unsigned int *blue_sums,
                                   Color *k_new_colors
        );

        void run(Color *k_colors, const unsigned int &k, Color color_array[], const unsigned int &n
        ) {
            // Allocation de la mémoire sur GPU

            Color *d_k_colors = NULL;
            Color *d_color_array = NULL;

            cudaMalloc(&d_k_colors, k * sizeof(Color));
            cudaMalloc(&d_color_array, n * sizeof(Color));

            // On s'assure que les allocaltions ont été satisfaites
            assert(d_k_colors != NULL &&
                   d_color_array != NULL
            );

            // Transfert des données depuis le 'host' vers le 'device'
            cudaMemcpy(d_k_colors, k_colors, k * sizeof(Color), cudaMemcpyHostToDevice);
            cudaMemcpy(d_color_array, color_array, n * sizeof(Color), cudaMemcpyHostToDevice);

            // Déclaration des mémoires intérmédiaires
            unsigned int *d_classification_array = NULL;
            unsigned int *d_k_cardinals = NULL;
            unsigned int *d_red_sums = NULL;
            unsigned int *d_green_sums = NULL;
            unsigned int *d_blue_sums = NULL;
            Color *d_k_new_colors = NULL;

            // Allocation des mémoires intermédiaires
            cudaMalloc(&d_classification_array, n * sizeof(unsigned int));
            cudaMalloc(&d_k_cardinals, k * sizeof(unsigned int));
            cudaMalloc(&d_red_sums, k * sizeof(unsigned int));
            cudaMalloc(&d_green_sums, k * sizeof(unsigned int));
            cudaMalloc(&d_blue_sums, k * sizeof(unsigned int));
            cudaMalloc(&d_k_new_colors, k * sizeof(Color));

            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                // print the CUDA error message and exit
                printf("CUDA error: %s\n", cudaGetErrorString(error));
                exit(-1);
            }

            // On s'assure que toutes les allocations on été satisfaites.
            assert(d_classification_array != NULL &&
                   d_k_cardinals != NULL &&
                   d_red_sums != NULL &&
                   d_green_sums != NULL &&
                   d_blue_sums != NULL &&
                   d_k_new_colors != NULL);
            // Dimension des la grille et des blocks.
            dim3 grid(1, 1, 1);
            dim3 block(1, 1, 1);

            // Mesure du temps d'execution
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            // On lance le calcul sur le CPU.
            run_on_gpu <<< grid, block >>> (d_k_colors, k, d_color_array, n,
                    d_classification_array, d_k_cardinals, d_red_sums, d_green_sums, d_blue_sums, d_k_new_colors);

            // Arrêter la mesure du temps
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            // Impression du temps d'execution
            std::cout << "Kernel execution time (ms): " << milliseconds << std::endl;

            // Copie des résultats du 'device' vers le 'host'
            cudaMemcpy(color_array, d_color_array, n * sizeof(Color), cudaMemcpyDeviceToHost);

            // Libération de la mémoire allouée précédement
            // Mémoires intérmédiaires.
            cudaFree(d_classification_array);
            cudaFree(d_k_cardinals);
            cudaFree(d_red_sums);
            cudaFree(d_green_sums);
            cudaFree(d_blue_sums);
            cudaFree(d_k_new_colors);

            // Libération de la mémoire allouée précédement.
            cudaFree(d_k_colors);
            cudaFree(d_color_array);
        }

        __device__ float rgb_distance(const Color &color_1, const Color &color_2);

        __device__ unsigned int closest_color_idx(const Color &color, const unsigned int &k, const Color k_colors[]);

        __device__ bool same_colors(const Color &color_1, const Color &color_2);

        // k_colors: Vecteur contenant les k-couleurs dominantes initialisées comme décrit dans le sujet.
        // k: Le nombre de régions qu'on souhaite extraire de l'image.
        // color_array: Un Vecteur de couleurs RGB contenant l'ensemble des pixel de l'image à segmenter.
        // n: Le nombre total de pixels de l'image à segmenter.
        // classification_array: Vecteur qui va servir à classifier les pixels de l'image.
        // k_cardinals: Vecteur temporaire qui va servir pour compter le nombre de pixel appartenant à chaque région.
        // red_sums: Vecteur de dimension 'k' qui va servir pour calculer les moyennes de rouge 'r' pour chaque région.
        // green_sums: Vecteur de dimension 'k' qui va servir pour calculer les moyennes de vert 'g' pour chaque région.
        // blue_sums: Vecteur de dimension 'k' qui va servir pour calculer les moyennes de bleu 'b' pour chaque région.

        __global__ void run_on_gpu(Color *k_colors, unsigned int k, Color *color_array, unsigned int n,
                                   unsigned int *classification_array, unsigned int *k_cardinals,
                                   unsigned int *red_sums,
                                   unsigned int *green_sums,
                                   unsigned int *blue_sums,
                                   Color *k_new_colors
        ) {
            // stop_event: Bool qui va servir pour signaler la convergence de la méthode.
            bool stop;
            // Execution de la méthode.
            do {

                // Initialisation du vecteur k_cardinals.
                for (unsigned int i = 0; i < k; i++) {
                    k_cardinals[i] = 0;
                }
                // Classification de tout les pixels.
                // Pour chaque pixel, on détermine sa classe en déterminant la couleur dominante vers
                // laquelle il se rapproche le plus.
                // Pour cela, on utilise le vecteur 'classification_array'. Pour chaque pixel d'index 'i' on écrit
                // dans classfication_array[i] l'indice de la couleur se trouveant dans 'k_colors' vers
                // laquelle il se rapproche le plus.
                for (unsigned int i = 0; i < n; i++) {
                    // Déterminer l'indice de la couleur dominante qui se rapproche le plus du pixel.
                    unsigned int closest_k_color_idx = closest_color_idx(color_array[i], k, k_colors);
                    // Classifier le pixel
                    classification_array[i] = closest_k_color_idx;
                    // Incrémenter le nombre de pixels trouvés appartenant à la région d'indice 'closest_k_color_idx'
                    ++k_cardinals[closest_k_color_idx];
                }
                // Une fois tout les pixel classifiés, on commence à calculer la couleur moyenne pour chaque région.
                // Pour cela, on utilise 3 vecteurs (r,g,b) de dimension 'k' dans lesquels on va stocker les sommes
                // des trois composantes (r,g,b) de chaque pixel.

                // Initialisation des vecteurs de sommes des composantes.
                for (unsigned int i = 0; i < k; i++) {
                    red_sums[i] = 0;
                    green_sums[i] = 0;
                    blue_sums[i] = 0;
                }
                // Calcul des sommes.
                for (unsigned int i = 0; i < n; i++) {
                    red_sums[classification_array[i]] += color_array[i].r;
                    green_sums[classification_array[i]] += color_array[i].g;
                    blue_sums[classification_array[i]] += color_array[i].b;
                }
                // Calcul des couleurs moyennes dans le vecteur 'k_new_colors' de dimension 'k'.
                for (unsigned int i = 0; i < k; i++) {
                    k_new_colors[i].r = (float) red_sums[i] / (float) k_cardinals[i];
                    k_new_colors[i].g = (float) green_sums[i] / (float) k_cardinals[i];
                    k_new_colors[i].b = (float) blue_sums[i] / (float) k_cardinals[i];
                }
                // On suppose que le processus a convergé.
                stop = true;
                // On vérifie si le processus a convergé. Si les couleurs moyennes sont les mêmes que celles précedement
                // calculées, le processus va s'arréter (stop_event == true). Sinon, on écrase la couleur moyenne qui
                // a changée avec la nouvelle couleur 'k_new_colors[i]'.
                for (unsigned int i = 0; i < k; i++) {
                    if (!same_colors(k_new_colors[i], k_colors[i])) {
                        k_colors[i] = k_new_colors[i];
                        stop = false;
                    }
                }

            } while (!stop); // Voir là ou 'stop_event' est modifiée.
            // On change les couleurs de régions. On colorie chacune par la couleur dominante correspondante.
            for (unsigned int i = 0; i < n; i++) {
                color_array[i] = k_colors[classification_array[i]];
            }
        }

        // Retourner l'index de la couleur qui se trouve dans 'k_colors' qui est la plus proche de la couleur 'color'
        __device__ unsigned int closest_color_idx(const Color &color, const unsigned int &k, const Color k_colors[]) {
            // On s'assure que le vecteur contient au moins une couleur ('k > 0')
            // assert (k > 0);
            unsigned int closest_color_idx = 0;
            double min_rgb_distance = rgb_distance(color, (k_colors[closest_color_idx]));
            // Skipping the first step (starting from i = 1).
            for (unsigned int i = 1; i < k; i++) {
                double current_rgb_distance = rgb_distance(color, (k_colors[i]));
                if (current_rgb_distance < min_rgb_distance) {
                    min_rgb_distance = current_rgb_distance;
                    closest_color_idx = i;
                }
            }
            return closest_color_idx;
        }

        // Calculer la distance rgb est deux couleurs.
        __device__ float rgb_distance(const Color &color_1, const Color &color_2) {
            return sqrtf(powf(color_1.r - color_2.r, 2) +
                         powf(color_1.g - color_2.g, 2) +
                         powf(color_1.b - color_2.b, 2));
        }

        // Comparer si deux couleurs ont sont les mêmes.
        __device__ bool same_colors(const Color &color_1, const Color &color_2) {
            return color_1.r == color_2.r &&
                   color_1.g == color_2.g &&
                   color_1.b == color_2.b;
        }
    }
}
