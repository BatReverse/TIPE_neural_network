// kernel.cu — Kernels CUDA de bas niveau : un thread traite un élément du
// tableau de sortie (indice `i = blockIdx*blockDim + threadIdx`), avec une
// vérification `i < taille` pour ignorer les threads en trop du dernier
// bloc. Ce fichier est inclus directement (pas seulement son .h) par
// matrice.cu, qui appelle ces kernels avec <<<gridDim, blockDim>>>.
#include"matrice.h"
#include"stdio.h"

// Dérivée du coût quadratique : C = 2*(A - obj).
__global__ void cuda_dcost(float* A,float* obj,float* C,int lignes,int colonnes){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<lignes*colonnes)
        C[i] = 2*(A[i] - obj[i]);
}

// Produit matriciel C = A x B. Grille 2D : chaque thread calcule un
// coefficient C[ligne, colonne] par un produit scalaire direct (pas de
// mémoire partagée / tuiling ici, volontairement simple pour ce projet).
__global__ void cuda_dot(
    float* A, int lignes_A, int colonnes_A,
    float* B, int lignes_B, int colonnes_B,
    float* C, int lignes_C, int colonnes_C) {

    // Indices globaux pour les lignes et colonnes de la matrice C
    int ligne = blockIdx.y * blockDim.y + threadIdx.y;
    int colonne = blockIdx.x * blockDim.x + threadIdx.x;

    // Vérification des limites pour éviter les accès hors limites
    if (ligne < lignes_C && colonne < colonnes_C) {
        float somme = 0.0;

        // Calcul du produit scalaire pour C[ligne, colonne]
        #pragma unroll
        for (int i = 0; i < colonnes_A; i++) { // colonnes_A == lignes_B
            somme += A[ligne * colonnes_A + i] * B[i * colonnes_B + colonne];
        }

        // Stocker le résultat dans C
        C[ligne * colonnes_C + colonne] = somme;
    }
}



// Produit terme à terme (Hadamard) : C = A .* B.
__global__ void cuda_hadamard(float* A,float* B,float* C,int lignes,int colonnes){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<lignes*colonnes)
        C[i] = A[i] * B[i];
}
// Somme terme à terme : C = A + B.
__global__ void cuda_sum(float* A,float* B,float* C,int lignes,int colonnes){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<lignes*colonnes)
        C[i] = A[i] + B[i];
}
// Différence terme à terme : C = A - B.
__global__ void cuda_diff(float* A,float* B,float* C,int lignes,int colonnes){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<lignes*colonnes)
        C[i] = A[i] - B[i];
}

// Transposition : C[j, i] = A[i, j]. Grille 2D, un thread par élément de A.
__global__ void cuda_transpose(float* A, float* C, int lignes_A, int colonnes_A) {
    // Calcul des indices globaux
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Ligne
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Colonne

    // Vérification pour éviter les accès hors limites
    if (i < lignes_A && j < colonnes_A) {
        // Transposer : A[i, j] devient C[j, i]
        C[j * lignes_A + i] = A[i * colonnes_A + j];
    }
}

// ReLU : C = max(A, 0).
__global__ void cuda_RELU(float* A,float* C,int taille){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille){
        if (A[i]>0)
            C[i] = A[i];
        else
            C[i] = 0;
    }
}

// Dérivée de ReLU : C = 1 si A > 0, sinon 0.
// (kernel non utilisé actuellement, voir le commentaire sur mat_RELU_d dans matrice.cu)
__global__ void cuda_RELU_d(float* A,float* C,int taille){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille){
        if(A[i]>0)
            C[i] = 1;
        else
            C[i] = 0;
    }
}

// Sigmoïde : C = 1 / (1 + exp(-A)).
__global__ void cuda_sigmoid(float* A,float* C,int taille){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille){
        C[i] = 1/(1+exp(-A[i]));
    }
}

// Dérivée de la sigmoïde : C = sig(A) * (1 - sig(A)).
__global__ void cuda_sigmoid_d(float* A,float* C,int taille){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille){
        float sig= 1/(1+exp(-A[i]));
        C[i] = sig*(1-sig);
    }
}

// Softmax : C_i = exp(A_i) / e, où `e` (la somme des exponentielles) est
// calculée côté CPU par l'appelant (voir mat_SOFT_MAX dans matrice.cu) et
// simplement passée en paramètre ici.
__global__ void cuda_softmax(float* A,float* C,int taille,float e){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille){
        C[i] = exp(A[i])/e;
    }
}

// Multiplication par un scalaire, en place : A *= lambda.
__global__ void cuda_multiply(float* A,int taille, float lambda){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille)
        A[i] *= lambda;
}

// Accumule la somme des carrés des écarts (A_i - B_i)^2 dans *res.
// atomicAdd est nécessaire car tous les threads écrivent dans la même case
// mémoire `res`.
__global__ void cuda_cout(float* A, float* B, int taille, float* res){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<taille)
        atomicAdd(res,(A[i]-B[i])*(A[i]-B[i]));
}



// Différence pondérée : C = A - alpha*B.
__global__ void cuda_diff_avec_constante(float* A,float* B,float* C,int lignes,int colonnes,float alpha){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<lignes*colonnes)
        C[i] = A[i] - alpha*B[i];
}

// Moyenne mobile utilisée par l'optimiseur Momentum : V = beta*V + (1-beta)*B.
__global__ void cuda_apply_momentum(float* V,float* B, float beta, int taille){
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<taille){
        V[i]= beta*V[i] + (1-beta)*B[i];
    }
}

// --- Optimiseur Adam : quatre étapes, chacune un kernel séparé ---

// 1er moment (moyenne mobile du gradient) : M = Beta1*M + (1-Beta1)*grad.
__global__ void cuda_adam_apply_momentum(float* M,float Beta1,float* grad,int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){
        M[i] = Beta1 * M[i] + (1-Beta1)*grad[i];
    }
}

// 2e moment (moyenne mobile du carré du gradient) : V = Beta2*V + (1-Beta2)*grad^2.
__global__ void cuda_adam_apply_speed(float* V,float Beta2,float* grad,int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){
        V[i] = Beta2 * V[i] + (1-Beta2)*grad[i]*grad[i];
    }
}

// Correction de biais ("hat") : Vc = V / (1 - Beta^t), t = itération courante.
__global__ void cuda_adam_apply_hat(float* Vc,float* V, float Betat,int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){
        Vc[i] = V[i]/(1-Betat);
    }
}

// Mise à jour finale du poids : W -= learning_rate * Mc / (sqrt(Vc) + epsilon).
__global__ void cuda_adam_apply_weight(float* W,float learning_rate,float* Mc,float* Vc,float eplsilon,int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){
        W[i] = W[i]-learning_rate*Mc[i]/(sqrt(Vc[i])+eplsilon);
    }
}