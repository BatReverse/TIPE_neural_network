#!/bin/bash
# Compile le projet avec nvcc (nécessite le CUDA Toolkit et un GPU NVIDIA).
# kernel.cu n'est pas listé ici : il est inclus directement (#include
# "kernel.cu") par matrice.cu, donc compilé une seule fois avec lui.
# -diag-suppress 2464 : coupe un avertissement nvcc jugé sans impact ici.
nvcc -o main main.cu Pile.cu matrice.cu MNIST_manager.cu neural_network.cu optimizer.cu -lm -diag-suppress 2464

