#/bin/bash

nvcc -G -g -o main main.cu matrice.cu MNIST_manager.cu neural_network.cu -lm -G -diag-suppress 2464

