#include"matrice.h"
#include"neural_network.h"
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
    cudaMalloc(&(res->data), sizeof(float) * lignes * colonnes);

    // Initialiser les données à 0 sur le GPU
    cudaMemset(res->data, 0, sizeof(float) * lignes * colonnes);

    return res;
}

matrice* random_mat(int lignes, int colonnes, float x) {
    matrice* res;

    // Allouer la structure matrice sur le CPU
    res = (matrice*)malloc(sizeof(matrice));

    // Initialiser les champs de la structure
    res->lignes = lignes;
    res->colonnes = colonnes;

    // Allouer la mémoire pour les données sur le GPU
    cudaMalloc(&(res->data), sizeof(float) * lignes * colonnes);

    // Générer les données aléatoires sur le CPU
    float* host_data = (float*)malloc(sizeof(float) * lignes * colonnes);

    for (int i = 0; i < lignes * colonnes; i++) {
        host_data[i] = (2.0 * ((float)rand() / RAND_MAX) - 1.0) * x; // Générer un nombre entre 0 et x
    }

    // Copier les données générées du CPU vers le GPU
    cudaMemcpy(res->data, host_data, sizeof(float) * lignes * colonnes, cudaMemcpyHostToDevice);

    // Libérer la mémoire temporaire sur le CPU
    free(host_data);

    return res;
}


void print_mat(matrice* mat_device) {
    // Allouer un tableau sur le CPU pour copier les données
    int total_size = mat_device->lignes * mat_device->colonnes;
    float* mat_host = (float*)malloc(sizeof(float) * total_size);

    // Copier les données de la mémoire GPU (device) vers la mémoire CPU (host)
    cudaMemcpy(mat_host, mat_device->data, sizeof(float) * total_size, cudaMemcpyDeviceToHost);

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

void dot_par(matrice* A, matrice* B, matrice* C) {
    if (A->colonnes != B->lignes || C->lignes != A->lignes || C->colonnes != B->colonnes) {
        printf("Erreur : Dimensions incompatibles pour le produit matriciel (A: %dx%d, B: %dx%d, C: %dx%d).\n",
               A->lignes, A->colonnes, B->lignes, B->colonnes, C->lignes, C->colonnes);
        exit(EXIT_FAILURE);
    }

    // Définition des dimensions du bloc et de la grille pour CUDA
    dim3 blockDim(Nl, Nl);
    dim3 gridDim((C->colonnes + blockDim.x - 1) / blockDim.x, 
                 (C->lignes + blockDim.y - 1) / blockDim.y);

    // Vérification des dimensions de la grille et du bloc
    if (gridDim.x == 0 || blockDim.x == 0) {
        printf("Erreur : Dimensions de la grille ou du bloc invalides.\n");
        return;
    }

    // Lancement du kernel de multiplication matricielle
    cuda_dot<<<gridDim, blockDim>>>(A->data, A->lignes, A->colonnes, 
                                    B->data, B->lignes, B->colonnes, 
                                    C->data, C->lignes, C->colonnes);

    // Vérification des erreurs CUDA après le lancement du kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return;
    }
}


void hadamar(matrice* A,matrice* B,matrice *C){
    if(A->colonnes != B->colonnes || A->lignes != B->lignes ||
       A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("hadamar invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(Nl);
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
    }
}

void sum(matrice* A,matrice* B,matrice* C){
if(A->colonnes != B->colonnes || A->lignes != B->lignes ||
       A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("sum invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(Nl);
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
    }
}


void diff(matrice* A,matrice* B,matrice* C){

    if(A->colonnes != B->colonnes || A->lignes != B->lignes ||
       A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("diff invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(Nl);
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


void diff_avec_constante(matrice* A,matrice* B,matrice* C,float alpha){

    if(A->colonnes != B->colonnes || A->lignes != B->lignes ||
       A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("diff invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(Nl);
    dim3 gridDim((C->colonnes*C->lignes+blockDim.x - 1)/blockDim.x);

    if (gridDim.x == 0 ||  blockDim.x == 0 ) {
        printf("Erreur : Dimensions de la grille ou du bloc invalides.\n");
        return ;
    }

    cudaError_t err = cudaGetLastError();
    cuda_diff_avec_constante<<<gridDim,blockDim>>>(A->data,B->data,C->data,A->lignes,A->colonnes,alpha);
    
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
    matrice* res = zeros(A->colonnes, A->lignes);

    dim3 blockDim(Nl, Nl);
    dim3 gridDim((A->colonnes + blockDim.x - 1) / blockDim.x,
                 (A->lignes + blockDim.y - 1) / blockDim.y);

    // Vérifiez les dimensions
    if (gridDim.x <= 0 || gridDim.y <= 0 || blockDim.x <= 0 || blockDim.y <= 0) {
        printf("Erreur : Dimensions de la grille ou du bloc invalides.\n");
        return NULL;
    }
    if(A==NULL){
        printf("fdp");
    }
    // Lancer le kernel
    cudaError_t err = cudaSuccess;
    cuda_transpose<<<gridDim, blockDim>>>(A->data, res->data, A->lignes, A->colonnes);

    // Vérifiez les erreurs CUDA
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("A: %dx%d\n",A->colonnes,A->lignes);
        printf("transpose CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    return res;
}


void copy(matrice* A,matrice* C){
    if(A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("copy invalide");
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(C->data,A->data,sizeof(float)*A->lignes*A->colonnes,cudaMemcpyDeviceToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return ;
    }
}

void mat_RELU(matrice* A,matrice* C){
    if(A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("RELU_mat invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(Nl);
    dim3 gridDim((C->colonnes*C->lignes+blockDim.x - 1)/blockDim.x);

    if (gridDim.x == 0 ||  blockDim.x == 0 ) {
        printf("Erreur : Dimensions de la grille ou du bloc invalides.\n");
        return ;
    }
    int N = A->lignes*A->colonnes;

    cudaError_t err = cudaGetLastError();
    cuda_RELU<<<gridDim,blockDim>>>(A->data,C->data,N);
    
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
void mat_RELU_d(matrice* A,matrice* C){
    if(A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("RELU_mat invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(Nl);
    dim3 gridDim((C->colonnes*C->lignes+blockDim.x - 1)/blockDim.x);

    if (gridDim.x == 0 ||  blockDim.x == 0 ) {
        printf("Erreur : Dimensions de la grille ou du bloc invalides.\n");
        return ;
    }
    int N = A->lignes*A->colonnes;

    cudaError_t err = cudaGetLastError();
    cuda_RELU<<<gridDim,blockDim>>>(A->data,C->data,N);
    
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


void mat_sigmoid(matrice* A,matrice* C){
        if(A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("RELU_mat invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(Nl);
    dim3 gridDim((C->colonnes*C->lignes+blockDim.x - 1)/blockDim.x);

    if (gridDim.x == 0 ||  blockDim.x == 0 ) {
        printf("Erreur : Dimensions de la grille ou du bloc invalides.\n");
        return ;
    }
    int N = A->lignes*A->colonnes;

    cudaError_t err = cudaGetLastError();
    cuda_sigmoid<<<gridDim,blockDim>>>(A->data,C->data,N);
    
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

void mat_sigmoid_d(matrice* A,matrice* C){
    if(A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("sig_d invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(Nl);
    dim3 gridDim((C->colonnes*C->lignes+blockDim.x - 1)/blockDim.x);

    if (gridDim.x == 0 ||  blockDim.x == 0 ) {
        printf("Erreur : Dimensions de la grille ou du bloc invalides.\n");
        return ;
    }
    int N = A->lignes*A->colonnes;

    cudaError_t err = cudaGetLastError();
    cuda_sigmoid_d<<<gridDim,blockDim>>>(A->data,C->data,N);
    
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

void mat_SOFT_MAX(matrice* A,matrice* C){
    if(A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("RELU_mat invalide");
        exit(EXIT_FAILURE);
    }
        dim3 blockDim(Nl);
    dim3 gridDim((C->colonnes*C->lignes+blockDim.x - 1)/blockDim.x);

    int N = A->lignes*A->colonnes;
    float e;
    float* A_host=(float*)malloc(sizeof(float)*N);
    cudaMemcpy(A_host,A->data,sizeof(float)*N,cudaMemcpyDeviceToHost);

    //TODO optimiser code
    for (int i = 0; i < N; i++)
    {
        /* code */
        e+=exp(A_host[i]);
    }

    cudaError_t err = cudaGetLastError();
    cuda_softmax<<<gridDim,blockDim>>>(A->data,C->data,N,e);
    
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

//TODO
void mat_SOFT_MAX_d(matrice*A,matrice* C){
    
}

void dCOST(matrice* A,matrice* obj,matrice* C){
    if(A->colonnes != obj->colonnes || A->lignes != obj->lignes ||
       A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("dcost invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(Nl);
    dim3 gridDim((C->colonnes*C->lignes+blockDim.x - 1)/blockDim.x);

    if (gridDim.x == 0 ||  blockDim.x == 0 ) {
        printf("Erreur : dCost Dimensions de la grille ou du bloc invalides.\n");
        return ;
    }

    cudaError_t err = cudaGetLastError();
    cuda_dcost<<<gridDim,blockDim>>>(A->data,obj->data,C->data,A->lignes,A->colonnes);
    
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

void multiply(matrice* A,float lambda){

    dim3 blockDim(Nl);
    dim3 gridDim((A->colonnes*A->lignes+blockDim.x - 1)/blockDim.x);

    if (gridDim.x == 0 ||  blockDim.x == 0 ) {
        printf("Erreur : multiply Dimensions de la grille ou du bloc invalides.\n");
        return ;
    }
    cudaError_t err = cudaGetLastError();
    cuda_multiply<<<gridDim,blockDim>>>(A->data,A->lignes*A->colonnes,lambda);
    

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

void free_mat(matrice* mat){

    cudaFree(mat->data);
    free(mat);
}

void update_momentum_velocity(matrice*V,matrice* W,float beta){
    if(V->lignes != W->lignes || V->colonnes != W->colonnes){
        printf("apply_momentum invalide");
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(Nl);
    dim3 gridDim((V->colonnes*V->lignes+blockDim.x - 1)/blockDim.x);

    if (gridDim.x == 0 ||  blockDim.x == 0 ) {
        printf("Erreur : Dimensions de la grille ou du bloc invalides.\n");
        return ;
    }

    cudaError_t err = cudaGetLastError();
    cuda_apply_momentum<<<gridDim,blockDim>>>(V->data,W->data,beta,V->colonnes*V->lignes);
    
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



