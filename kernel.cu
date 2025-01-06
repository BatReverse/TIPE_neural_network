#include"matrice.h"
#include"stdio.h"

__global__ void cuda_dcost(double* A,double* obj,double* C,int lignes,int colonnes){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<lignes*colonnes)
        C[i] = 2*(A[i] - obj[i]);
}

__global__ void cuda_dot(double* A,int ligne_A,int colonnes_A,double* B,int ligne_B,int colonnes_B,double* C,int ligne_C,int colonnes_C) {
    int ligne = blockIdx.y * blockDim.y + threadIdx.y;
    int colonne = blockIdx.x * blockDim.x + threadIdx.x;

    if (ligne < ligne_C && colonne < colonnes_C) {
        for (int i = 0; i < ligne_B; i++)
        {
            C[ligne*colonnes_C+colonne] += A[ligne*colonnes_A+i]*B[i*colonnes_B+colonne];
        }
        
    }
}

__global__ void cuda_hadamard(double* A,double* B,double* C,int lignes,int colonnes){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<lignes*colonnes)
        C[i] = A[i] * B[i];
}
__global__ void cuda_sum(double* A,double* B,double* C,int lignes,int colonnes){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<lignes*colonnes)
        C[i] = A[i] + B[i];
}
__global__ void cuda_diff(double* A,double* B,double* C,int lignes,int colonnes){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<lignes*colonnes)
        C[i] = A[i] - B[i];
}

__global__ void cuda_transpose(double* A, double* C, int lignes_A, int colonnes_A) {
    // Calcul des indices globaux
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Ligne
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Colonne

    // Vérification pour éviter les accès hors limites
    if (i < lignes_A && j < colonnes_A) {
        // Transposer : A[i, j] devient C[j, i]
        C[j * lignes_A + i] = A[i * colonnes_A + j];
    }
}

__global__ void cuda_RELU(double* A,double* C,int taille){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille){
        if (A[i]>0) 
            C[i] = A[i];
        else 
            C[i] = 0;    
    }
}

__global__ void cuda_RELU_d(double* A,double* C,int taille){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille){
        if(A[i]>0)
            C[i] = 1;
        else 
            C[i] = 0;    
    }
}

__global__ void cuda_sigmoid(double* A,double* C,int taille){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille){
        C[i] = 1/(1+exp(-A[i]));    
    }
}

__global__ void cuda_sigmoid_d(double* A,double* C,int taille){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille){
        double sig= 1/(1+exp(-A[i]));
        C[i] = sig*(1-sig);    
    }
}

__global__ void cuda_softmax(double* A,double* C,int taille,double e){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille){
        C[i] = exp(A[i])/e;
    }
}

__global__ void cuda_multiply(double* A,int taille, double lambda){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille)
        A[i] *= 1;
}
