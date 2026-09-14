// kernel.h — Déclarations des kernels CUDA utilisés par l'optimiseur Adam
// (optimizer.cu) et par le calcul du coût (neural_network.cu).
//
// Note : la majorité des kernels du projet (produit matriciel, sommes,
// activations...) sont déclarés directement dans kernel.cu et inclus via
// `#include"kernel.cu"` dans matrice.cu ; ce header ne couvre que les
// kernels appelés depuis un fichier .cu différent de kernel.cu.
#ifndef KERNEL
#define KERNEL
// Accumule la somme des carrés des écarts (A_i - B_i)^2 dans *res (coût MSE).
__global__ void cuda_cout(float* A, float* B, int taille, float* res);


// Les quatre kernels suivants implémentent les étapes de l'optimiseur Adam :
// mise à jour du 1er moment (momentum), du 2e moment (vitesse), correction
// de biais ("hat"), puis application à la valeur du poids/biais.
__global__ void cuda_adam_apply_momentum(float* M,float Beta1,float* grad,int N);
__global__ void cuda_adam_apply_speed(float* V,float Beta2,float* grad,int N);
__global__ void cuda_adam_apply_hat(float* Vc,float* V, float Betat,int N);
__global__ void cuda_adam_apply_weight(float* W,float learning_rate,float* Mc,float* Vc,float eplsilon,int N);
#endif