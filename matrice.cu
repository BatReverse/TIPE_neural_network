// matrice.cu — Implémentation des opérations sur `matrice` déclarées dans
// matrice.h. Chaque fonction se contente de calculer les dimensions de la
// grille CUDA (gridDim/blockDim), de lancer le kernel correspondant (défini
// dans kernel.cu) puis de vérifier les erreurs CUDA. Le motif est donc très
// répétitif d'une fonction à l'autre ; seules les parties spécifiques à
// chaque opération sont commentées en détail.
#include"matrice.h"
#include"neural_network.h"
#include<stdio.h>
#include"kernel.cu"

// Alloue une matrice lignes x colonnes sur le GPU, initialisée à 0.
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

// Alloue une matrice lignes x colonnes sur le GPU et la remplit de valeurs
// aléatoires uniformes dans [-x, x] (génération faite côté CPU avec rand(),
// puis copiée sur le GPU : plus simple qu'un générateur GPU pour ce projet).
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


// Copie une matrice du GPU vers le CPU et l'affiche sur stdout. Utilisé
// uniquement pour le débogage manuel (ce n'est pas appelé pendant
// l'entraînement, trop lent pour des matrices de grande taille).
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

// Produit matriciel C = A x B (kernel 2D : un thread par élément de C).
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
        printf("CUDA error after kernel launch dot: %s\n", cudaGetErrorString(err));
        return;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return;
    }
}


// Produit terme à terme (Hadamard) : C = A .* B.
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
        printf("CUDA error after kernel launch hadamar: %s\n", cudaGetErrorString(err));
        return ;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return ;
    }
}

// Somme terme à terme : C = A + B.
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
        printf("CUDA error after kernel launch sum: %s\n", cudaGetErrorString(err));
        return ;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return ;
    }
}


// Différence terme à terme : C = A - B.
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
        printf("CUDA error after kernel launch diff: %s\n", cudaGetErrorString(err));
        return ;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return ;
    }
}


// Différence pondérée : C = A - alpha*B. Sert notamment à la mise à jour
// des poids/biais : poids = poids - vitesse_apprentissage * gradient.
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
        printf("CUDA error after kernel launch diff cst: %s\n", cudaGetErrorString(err));
        return ;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return ;
    }
}

// Renvoie une nouvelle matrice égale à la transposée de A (alloue le résultat).
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
        printf("Erreur : transpose() a reçu une matrice source NULL.\n");
    }
    // Lancer le kernel
    cudaError_t err = cudaSuccess;
    cuda_transpose<<<gridDim, blockDim>>>(A->data, res->data, A->lignes, A->colonnes);

    // Vérifiez les erreurs CUDA
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("A: %dx%d\n",A->colonnes,A->lignes);
        printf("transpose CUDA error after kernel launch transpose: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    return res;
}


// Copie le contenu de A dans C (mêmes dimensions), device to device.
void copy(matrice* A,matrice* C){
    if(A->colonnes != C->colonnes || A->lignes != C->lignes){
        printf("copy invalide");
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(C->data,A->data,sizeof(float)*A->lignes*A->colonnes,cudaMemcpyDeviceToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch copy: %s\n", cudaGetErrorString(err));
        return ;
    }
}

// Alloue une nouvelle matrice et y copie le contenu de A. Utilisé par
// copy_neural_network() pour dupliquer un réseau entier (entraînement par
// batch multi-threads).
matrice* copy_new( matrice* A) {
    // Vérification de l'entrée
    if (A == NULL || A->data == NULL) {
        fprintf(stderr, "Erreur: Matrice source invalide\n");
        exit(EXIT_FAILURE);
        return NULL;
    }

    // Allocation structure
    matrice* res = (matrice*)malloc(sizeof(matrice));
    if (res == NULL) {
        fprintf(stderr, "Erreur: Allocation CPU échouée\n");
        return NULL;
    }

    // Copie métadonnées
    res->lignes = A->lignes;
    res->colonnes = A->colonnes;
    res->data = NULL;

    // Allocation GPU
    size_t size = res->lignes * res->colonnes * sizeof(float);
    cudaError_t err = cudaMalloc(&res->data, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err));
        free(res);
        return NULL;
    }

    // Copie des données
    err = cudaMemcpy(res->data, A->data, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erreur copie CUDA: %s\n", cudaGetErrorString(err));
        cudaFree(res->data);
        free(res);
        return NULL;
    }

    return res;
}

// Applique ReLU terme à terme : C = max(A, 0).
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
// Dérivée de ReLU terme à terme : C = 1 si A > 0, sinon 0.
// ATTENTION (bug existant, non corrigé ici pour ne pas changer le
// comportement du réseau) : cette fonction appelle le kernel cuda_RELU au
// lieu de cuda_RELU_d (défini dans kernel.cu mais jamais utilisé). Avec
// MIDLAYER/OUTPUTLAYER = 1, la rétropropagation utilise donc ReLU(A) à la
// place de ReLU'(A) comme "dérivée".
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


// Applique la sigmoïde terme à terme : C = 1 / (1 + exp(-A)).
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

// Dérivée de la sigmoïde terme à terme : C = sig(A) * (1 - sig(A)).
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

// Softmax : C_i = exp(A_i) / somme_j(exp(A_j)).
// La somme au dénominateur est calculée sur CPU (copie GPU -> CPU de A),
// puis le kernel cuda_softmax calcule chaque exp(A_i)/e sur GPU.
// ATTENTION (bug existant, non corrigé ici) : la variable `e` n'est pas
// initialisée à 0 avant la boucle d'accumulation, elle part donc d'une
// valeur indéterminée. À utiliser avec prudence si MIDLAYER/OUTPUTLAYER = 3.
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

// TODO : dérivée du softmax, jamais implémentée. Ne pas utiliser
// MIDLAYER/OUTPUTLAYER = 3 (softmax) pour l'instant : la rétropropagation
// (calcul_grad dans neural_network.cu) appelle cette fonction mais elle ne
// fait rien, `C` n'est jamais modifiée.
void mat_SOFT_MAX_d(matrice*A,matrice* C){

}

// Dérivée du coût quadratique (MSE) par rapport à la sortie du réseau :
// C = 2*(A - obj), avec A l'activation de sortie et obj la sortie attendue.
void dCOST(matrice* A,matrice* obj,matrice* C){
    if (A->colonnes != obj->colonnes || A->lignes != obj->lignes ||
    A->colonnes != C->colonnes || A->lignes != C->lignes) {
    
    // Vérification détaillée pour afficher la cause exacte
    if (A->colonnes != obj->colonnes) {
        printf("Erreur : Le nombre de colonnes de A (%d) ne correspond pas à celui de obj (%d)\n", A->colonnes, obj->colonnes);
    }
    if (A->lignes != obj->lignes) {
        printf("Erreur : Le nombre de lignes de A (%d) ne correspond pas à celui de obj (%d)\n", A->lignes, obj->lignes);
    }
    if (A->colonnes != C->colonnes) {
        printf("Erreur : Le nombre de colonnes de A (%d) ne correspond pas à celui de C (%d)\n", A->colonnes, C->colonnes);
    }
    if (A->lignes != C->lignes) {
        printf("Erreur : Le nombre de lignes de A (%d) ne correspond pas à celui de C (%d)\n", A->lignes, C->lignes);
    }
    
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

// Multiplie A par un scalaire, en place : A *= lambda.
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

// Libère une matrice : buffer GPU (cudaFree) puis structure CPU (free).
void free_mat(matrice* mat){

    cudaFree(mat->data);
    free(mat);
}

// Met à jour une moyenne mobile (moment d'ordre 1) utilisée par l'optimiseur
// Momentum : V = beta*V + (1-beta)*W.
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
        printf("CUDA error after kernel launch momementu: %s\n", cudaGetErrorString(err));
        return ;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return ;
    }
}



