#/bin/bash

nvcc -o main main.cu matrice.cu -lm
./main