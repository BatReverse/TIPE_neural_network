#include"matrice.h"
#include<stdio.h>
#include"kernel.cu"

matrice* zeros(int lignes, int colonnes) {
    matrice* res;

    // Allouer la structure 'matrice' sur le CPU
    res = (matrice*)malloc(sizeof(matrice)); 

    // Initialiser les champs
    res->lignes = lignes;
    res->colonnes = colonnes;

    // Allouer la mémoire pour les données sur le GPU
    cudaMalloc(&(res->data), sizeof(double) * lignes * colonnes);

    // Initialiser les données à 0 sur le GPU
    cudaMemset(res->data, 0, sizeof(double) * lignes * colonnes);

    return res;
}

matrice* random_mat(int lignes, int colonnes, double x) {
    matrice* res;

    // Allouer la structure matrice sur le CPU
    res = (matrice*)malloc(sizeof(matrice));

    // Initialiser les champs de la structure
    res->lignes = lignes;
    res->colonnes = colonnes;

    // Allouer la mémoire pour les données sur le GPU
    cudaMalloc(&(res->data), sizeof(double) * lignes * colonnes);

    // Générer les données aléatoires sur le CPU
    double* host_data = (double*)malloc(sizeof(double) * lignes * colonnes);

    for (int i = 0; i < lignes * colonnes; i++) {
        host_data[i] = ((double)rand() / RAND_MAX) * 2.0 * x - x; // Générer un nombre entre -x et x
    }

    // Copier les données générées du CPU vers le GPU
    cudaMemcpy(res->data, host_data, sizeof(double) * lignes * colonnes, cudaMemcpyHostToDevice);

    // Libérer la mémoire temporaire sur le CPU
    free(host_data);

    return res;
}

void print_mat(matrice* mat_device) {
    // Allouer un tableau sur le CPU pour copier les données
    int total_size = mat_device->lignes * mat_device->colonnes;
    double* mat_host = (double*)malloc(sizeof(double) * total_size);

    // Copier les données de la mémoire GPU (device) vers la mémoire CPU (host)
    cudaMemcpy(mat_host, mat_device->data, sizeof(double) * total_size, cudaMemcpyDeviceToHost);

    // Imprimer la matrice
    printf("Matrice (%d x %d):\n", mat_device->lignes, mat_device->colonnes);
    for (int i = 0; i < mat_device->lignes; i++) {
        for (int j = 0; j < mat_device->colonnes; j++) {
            printf("%f ", mat_host[i * mat_device->colonnes + j]);
        }
        printf("\n");
    }

    // Libérer la mémoire CPU utilisée pour afficher
    free(mat_host);
}

void dot_par(matrice* A,matrice* B,matrice* C){
    if(A->colonnes != B->lignes || C->lignes != A->lignes || C->colonnes != B->colonnes){
        printf("produit matrice incompatible");
    }
    dim3 blockDim(16, 16);
    dim3 gridDim(C->colonnes / blockDim.x, C->lignes / blockDim.y);
    cuda_dot<<<gridDim, blockDim>>>(A->data,A->lignes,A->colonnes, B->data,B->lignes,B->colonnes, C->data,C->lignes,C->colonnes);
}

void hadamar(matrice* A,matrice* B,matrice *C){
    if(A->colonnes != B->colonnes || A->lignes != B->lignes ||
       A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("sum invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(16);
    dim3 gridDim((C->colonnes*C->lignes+blockDim.x - 1)/blockDim.x);

    if (gridDim.x == 0 ||  blockDim.x == 0 ) {
        printf("Erreur : Dimensions de la grille ou du bloc invalides.\n");
        return ;
    }

    cudaError_t err = cudaGetLastError();
    cuda_hadamard<<<gridDim,blockDim>>>(A->data,B->data,C->data,A->lignes,A->colonnes);
    
    // Vérifiez les erreurs CUDA après le lancement du kernel
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return ;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return ;
    }}

void sum(matrice* A,matrice* B,matrice* C){
if(A->colonnes != B->colonnes || A->lignes != B->lignes ||
       A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("sum invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(16);
    dim3 gridDim((C->colonnes*C->lignes+blockDim.x - 1)/blockDim.x);

    if (gridDim.x == 0 ||  blockDim.x == 0 ) {
        printf("Erreur : Dimensions de la grille ou du bloc invalides.\n");
        return ;
    }

    cudaError_t err = cudaGetLastError();
    cuda_sum<<<gridDim,blockDim>>>(A->data,B->data,C->data,A->lignes,A->colonnes);
    
    // Vérifiez les erreurs CUDA après le lancement du kernel
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return ;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return ;
    }}


void diff(matrice* A,matrice* B,matrice* C){

    if(A->colonnes != B->colonnes || A->lignes != B->lignes ||
       A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("sum invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(16);
    dim3 gridDim((C->colonnes*C->lignes+blockDim.x - 1)/blockDim.x);

    if (gridDim.x == 0 ||  blockDim.x == 0 ) {
        printf("Erreur : Dimensions de la grille ou du bloc invalides.\n");
        return ;
    }

    cudaError_t err = cudaGetLastError();
    cuda_diff<<<gridDim,blockDim>>>(A->data,B->data,C->data,A->lignes,A->colonnes);
    
    // Vérifiez les erreurs CUDA après le lancement du kernel
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return ;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return ;
    }
}

matrice* transpose(matrice* A) {
    // Créer une nouvelle matrice pour la transposition (dimensions inversées)
    matrice* res = zeros(A->colonnes, A->lignes);

    // Définir les dimensions du bloc et de la grille
    dim3 blockDim(16, 16);  // Taille du bloc 16x16, à ajuster selon votre GPU
    dim3 gridDim((A->colonnes + blockDim.x - 1) / blockDim.x, 
                 (A->lignes + blockDim.y - 1) / blockDim.y);

    // Vérifiez que les dimensions sont valides
    if (gridDim.x == 0 || gridDim.y == 0 || blockDim.x == 0 || blockDim.y == 0) {
        printf("Erreur : Dimensions de la grille ou du bloc invalides.\n");
        return NULL;
    }

    // Lancer le kernel pour la transposition
    cudaError_t err = cudaSuccess;
    cuda_transpose<<<gridDim, blockDim>>>(A->data, res->data, A->lignes, A->colonnes);
    err = cudaGetLastError();
    
    // Vérifiez les erreurs CUDA après le lancement du kernel
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    // Retourner la matrice transposée
    return res;
}

matrice* copy(matrice* A){
    matrice* res=(matrice*)malloc(sizeof(matrice));
    res->colonnes = A->colonnes;
    res->lignes = A->lignes;
    cudaMalloc(&(res->data), sizeof(double) * A->colonnes * A->lignes);
    cudaMemcpy(res->data,A->data,sizeof(double) * A->colonnes * A->lignes,cudaMemcpyDeviceToDevice);
    return res;
}