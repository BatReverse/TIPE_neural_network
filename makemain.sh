#/bin/bash

nvcc -G -g -o main main.cu matrice.cu neural_network.cu -lm -G

