# Neural Network on GPU (CUDA) — TIPE Project

"From scratch" implementation of a multilayer perceptron trained by
gradient backpropagation, with all matrix operations running on GPU via
CUDA. This project was built as part of a TIPE (French supervised personal
research project for preparatory-class students): **it is not intended to
be production code**, but rather to explore and illustrate how a neural
network works and the benefit of parallel computation on GPU.

The demo application trains a network to recognize handwritten digits from
the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

## Requirements

- An NVIDIA GPU and the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
  installed (`nvcc` must be in the PATH).
- A pthreads library (provided natively on Linux).

> The code can neither compile nor run without an NVIDIA GPU: all
> computations (matrices, activations, gradients...) are done on the GPU.
> This repository cleanup was therefore done without being able to
> recompile/test it, due to a lack of compatible hardware on the machine
> used — the program's logic was deliberately left unmodified (see the
> "What was not touched" section below).

## Building

```bash
./makemain.sh
```

This script calls `nvcc` to compile all `.cu` files in the project (except
`kernel.cu`, which is included directly by `matrice.cu`) and produces a
`main` executable.

## Usage

```bash
./main
```

By default, `main()` (in `main.cu`) launches `train_and_test_MNIST_opt()`:
indefinite training (infinite loop, to be stopped with Ctrl+C) of a
784 → 800 → 10 network on MNIST with the Adam optimizer. The network is
saved to `mnistDNN.nn` at the start of every generation, and the success
rate as well as the MSE are printed to standard output after each pass over
the test set.

Other scenarios are available in `neural_network_tests.cu` and can be
enabled by editing `main.cu` (uncomment a single line at a time):

| Function | Description |
|---|---|
| `train_and_test_MNIST_opt()` | MNIST training with the Adam optimizer (default). |
| `train_and_test_MNIST_batch()` | Batch, multi-threaded variant (see limitations below). |
| `train_and_test_MNIST()` | MNIST training with classic gradient descent (SGD), 2 generations. |
| `test_copy_neural_network()` | Checks the deep copy of a network. |
| `test_perf()` | Measures the time of a forward pass over 5000 images. |

The MNIST dataset is already included in `MNIST_dataset/` (IDX-format files,
downloaded from the LeCun Lab website).

## Code organization

```
matrice.h / matrice.cu   `matrice` type (GPU) and its operations (product,
                         sums, activations, transposition...)
kernel.h  / kernel.cu    Low-level CUDA kernels used by matrice.cu
                         and optimizer.cu
activation.h             Unused declarations (CPU leftover, see below)

neural_network.h / .cu   The neural network: creation, forward pass,
                         backpropagation, save/load, copy,
                         multi-threaded batch training
optimizer.h / .cu        Adam optimizer (Momentum is handled directly in
                         neural_network.cu)
Pile.h / Pile.cu         Linked stack of networks, used as a pool of
                         copies for batch training

MNIST_manager.h / .cu    Reading the MNIST dataset (IDX format) and
                         converting it to GPU matrices
network_vis.cu           Debug utility: prints a network in detail

neural_network_tests.cu  Training/test scenarios (XOR, MNIST...)
main.cu                  Entry point: chooses which scenario to run
makemain.sh              Build script (nvcc)
```

Every source file now has a header comment explaining its role, and every
function a comment summarizing what it does — see the code directly for
details.

### General principle

- A `matrice` is a small structure allocated on the CPU side whose `data`
  field points to a buffer allocated on the GPU side (`cudaMalloc`). All
  operations (`dot_par`, `sum`, `mat_sigmoid`, ...) launch a CUDA kernel and
  write their result into an output matrix already allocated by the caller,
  to limit repeated GPU allocations during training.
- A `neural_network` keeps, for each layer: its pre-activation
  (`neuronnes_somme`), its activation (`neuronnes_activ`), its weights/biases
  and the corresponding gradients (`dpoids`, `dbiais`, `dneuronnes`).
- Backpropagation (`calcul_grad` in `neural_network.cu`) computes the delta
  of the last layer then propagates it to the previous layers; for each
  layer, the product `weights^T * next_delta` is computed in a separate
  pthread while the main thread computes the derivative of the activation,
  before combining the two results.
- Three optimizers are available (`optimizer.h`): `Rien` ("none", classic
  SGD), `Momentum`, and `Adam`.

## Known limitations and unfinished code

Since this project is a TIPE and not a finished product, some parts are
deliberately incomplete or contain known bugs, left as-is during this
cleanup so as not to change the program's behavior. They are flagged with
`ATTENTION` or `TODO` comments in the code:

- **Non-differentiable softmax**: `mat_SOFT_MAX_d()` (in `matrice.cu`) is
  not implemented. Do not use `MIDLAYER`/`OUTPUTLAYER = 3` (softmax, see
  `neural_network.h`) until this is fixed.
- **`mat_SOFT_MAX`**: the variable accumulating the sum of the exponentials
  is not initialized to 0 before the loop.
- **`mat_RELU_d`** (in `matrice.cu`) calls the `cuda_RELU` kernel instead of
  `cuda_RELU_d`: with `MIDLAYER`/`OUTPUTLAYER = 1` (ReLU), backpropagation
  uses `ReLU(x)` instead of its derivative `ReLU'(x)`.
- **Batch training** (`batch_training` in `neural_network.cu`): averaging
  the gradient over the batch uses integer division (`1/batch_size`), which
  equals 0 as soon as `batch_size > 1` — the average gradient is therefore
  zeroed out instead of being averaged.
- **`activation.h`** declares scalar (CPU) activation functions that are
  never defined or called: a leftover from an early version designed to run
  without a GPU. The activations actually used are the GPU matrix versions
  (`mat_sigmoid`, `mat_RELU`, `mat_SOFT_MAX`).
- The `notes_pour_la_presentation.md` file contains the original working
  notes (observations on Adam, unfinished leads), kept for reference.
