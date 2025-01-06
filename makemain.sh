#/bin/bash

nvcc -o main main.cu matrice.cu neural_network.cu -lm

