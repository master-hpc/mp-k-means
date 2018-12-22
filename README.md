# "K-Means"
K-means image segmentation with CUDA (a "to be optimized" single threaded kernel implementation).

## Homework assignment

Students are asked to parallelize a naive implementation of the `K-Means` image segmentation algorithm.
A single threaded kernel is provided in `k_means_cuda.cu` and corresponds to the code that needs to be optimized.
Everything else can be seen as a boiler plate code.

## Requirements

  - a CUDA capable device with `compute capability >= 2.0`
  - the appropriate CUDA-toolkit needs to be installed

## Compilation

    # run from repo dir
    nvcc main.cpp external/Color.cpp external/utils.cpp k_means_cuda.cu -o out/k_means_cuda
    # run ./out/k_means_cuda

## When running `out/kmeans_cuda`

 - the user is asked to pick an image to be segmented
 - for quick tests, it is recommanded to pick the small one by tapping `s`
 - for measuring the gained performance of your optimizations it is recommanded to pick the big one by tapping `l`
 - input & ouptup images are loaded & saved to `./data` dir
 - increase `k` to stress your approache

## TODOs

 - [ ] upload the homework assignement PDF
 - [ ] upload ./data/input_large.bmp
 - [ ] translate `REAMDE.md` to french

## Attribution

This code uses [Bitmap Image Reader Writer Library](https://github.com/ArashPartow/bitmap) by [@ArashPartow](https://github.com/ArashPartow).
