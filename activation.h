// Déclarations des fonctions d'activation "scalaires" (CPU).
//
// NOTE : ces fonctions ne sont jamais définies ni appelées ailleurs dans le
// projet. Les activations réellement utilisées par le réseau de neurones
// sont les versions matricielles qui tournent sur GPU (mat_sigmoid, mat_RELU,
// mat_SOFT_MAX dans matrice.cu, elles-mêmes basées sur les kernels CUDA de
// kernel.cu). Ce header est un reliquat d'une première version pensée pour
// tourner sur CPU, conservé pour l'historique du projet.
#ifndef ACTIVATION
#define ACTIVATION

float sigmoid(float x);
float RELU(float x);
float sigmoid_d(float x);
float RELU_d(float x);

#endif