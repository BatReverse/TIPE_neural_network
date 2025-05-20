#/bin/bash

nvcc  -o main main.cu Pile.cu matrice.cu MNIST_manager.cu neural_network.cu optimizer.cu -lm  -diag-suppress 2464

