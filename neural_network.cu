// neural_network.cu — Cœur du projet : création du réseau, propagation
// avant, rétropropagation (calcul du gradient + mise à jour des poids),
// sérialisation sur disque, et entraînement par batch multi-threads.
#include"neural_network.h"
#include<stdio.h>
#include"matrice.h"
#include <pthread.h>
#include"kernel.h"
#include<stdarg.h>
#include"Pile.h"
#include"semaphore.h"
#include"MNIST_manager.h"
#include"assert.h"

// Non utilisé actuellement (aucune fonction ne construit/consomme de
// nn_thread) : reliquat d'une tentative antérieure de parallélisation.
typedef struct nn_thread{
    int i;
    matrice* mat;
    neural_network* reseau;
} nn_thread;


// Crée un réseau de `nb_couche` couches. Les arguments variadiques suivants
// donnent le nombre de neurones de chaque couche (ex : cree_reseau(3, 784,
// 800, 10)). Les poids sont initialisés selon Xavier/Glorot uniforme
// (bornes +-sqrt(6/(fan_in+fan_out))), les biais à zéro.
neural_network* cree_reseau(int nb_couche, ...){
    va_list ap;

    neural_network* res = (neural_network*)malloc(sizeof(neural_network));
    res->nombre_couche=nb_couche;
    res->vitesse_apprentissage = 0.01;
    res->neuronnes_parcouche = (int*)malloc(sizeof(int)*nb_couche);
    
    va_start(ap,nb_couche);
    for (int i = 0; i < nb_couche; i++)
    {
        res->neuronnes_parcouche[i] = va_arg(ap,int);
    }
    va_end(ap);
    
    res->neuronnes_somme = (matrice**)malloc(sizeof(matrice*)*nb_couche);
    res->neuronnes_activ = (matrice**)malloc(sizeof(matrice*)*nb_couche);
    
    res->poids = (matrice**)malloc(sizeof(matrice*)*nb_couche);
    res->biais = (matrice**)malloc(sizeof(matrice*)*nb_couche);
    
    res->dpoids = (matrice**)malloc(sizeof(matrice*)*nb_couche);
    res->dbiais = (matrice**)malloc(sizeof(matrice*)*nb_couche);
    res->dneuronnes = (matrice**)malloc(sizeof(matrice*)*nb_couche);



    for (int i = 0; i < nb_couche-1; i++)
    {
        int couche = res->neuronnes_parcouche[i];
        int prochaine = res->neuronnes_parcouche[i+1];
    
        if (i!=0){
            res->neuronnes_somme[i] = zeros(couche,1);
            res->neuronnes_activ[i] = zeros(couche,1);
        }
        

        float x = sqrt(6.)/sqrt(couche+prochaine);
        res->poids[i] = random_mat(prochaine,couche,x);
        res->biais[i] = zeros(prochaine,1);
    
        res->dpoids[i] = zeros(prochaine,couche);
        res->dbiais[i] = zeros(prochaine,1);
        res->dneuronnes[i] = zeros(couche,1);
    }
    int couche = res->neuronnes_parcouche[nb_couche-1];
    res->neuronnes_somme[nb_couche-1] = zeros(couche,1);
    res->neuronnes_activ[nb_couche-1] = zeros(couche,1);
    res->dneuronnes[nb_couche-1] = zeros(couche,1);

    return res;
}

// Propage un vecteur d'entrée `nourriture` à travers tout le réseau :
// neuronnes_activ[0] pointe directement sur `nourriture` (pas de copie), puis
// chaque couche suivante calcule somme = poids*activation_précédente + biais
// avant d'appliquer la fonction d'activation MIDLAYER. Le résultat final est
// disponible dans reseau->neuronnes_activ[L-1].
// (Les vérifications de NULL et de dimensions de `nourriture` ont été
// désactivées ici pour la performance ; elles ne sont plus faites.)
void propagation_avant(neural_network* reseau, matrice* nourriture) {
    int L = reseau->nombre_couche;

    // Initialisation de la première couche : pas de copie, l'entrée devient
    // directement l'activation de la couche 0.
    reseau->neuronnes_activ[0]= nourriture;

    // Propagation avant pour les couches suivantes
    for (int i = 0; i < L - 1; i++) {
        dot_par(reseau->poids[i], reseau->neuronnes_activ[i], reseau->neuronnes_somme[i + 1]);
        sum(reseau->neuronnes_somme[i + 1], reseau->biais[i], reseau->neuronnes_somme[i + 1]);

        switch (MIDLAYER) {
            case 1: // ReLU
                mat_RELU(reseau->neuronnes_somme[i + 1], reseau->neuronnes_activ[i + 1]);
                break;
            case 2: // Sigmoïde
                mat_sigmoid(reseau->neuronnes_somme[i + 1], reseau->neuronnes_activ[i + 1]);
                break;
            case 3: // Softmax
                mat_SOFT_MAX(reseau->neuronnes_somme[i + 1], reseau->neuronnes_activ[i + 1]);
                break;
            default:
                printf("Erreur : type d'activation MIDLAYER (%d) non défini.\n", MIDLAYER);
                exit(EXIT_FAILURE);
        }
    }
}






// Coût quadratique (somme des carrés des écarts) entre la sortie actuelle
// du réseau (dernière couche d'activation) et la sortie attendue `obj`.
float cout(neural_network* reseau, matrice* obj){
    float* sum;
    cudaMalloc(&sum,sizeof(float));
    cudaMemset(sum,0,sizeof(float));
    matrice* A = reseau->neuronnes_activ[reseau->nombre_couche-1];

    dim3 blockDim(Nl);
    dim3 gridDim((A->colonnes*A->lignes+blockDim.x - 1)/blockDim.x);

    if (gridDim.x == 0 ||  blockDim.x == 0 ) {
        printf("Erreur : multiply Dimensions de la grille ou du bloc invalides.\n");
        return -1;
    }
    cudaError_t err = cudaGetLastError();
    cuda_cout<<<gridDim,blockDim>>>(A->data,obj->data,A->lignes*A->colonnes,sum);
    

    // Vérifiez les erreurs CUDA après le lancement du kernel
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch ici: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return -1;
    }
    float sum_h;
    cudaMemcpy(&sum_h,sum,sizeof(float),cudaMemcpyDeviceToHost);
    return sum_h;
}

// Mise à jour "SGD classique" (sans optimiseur) : poids -= lr*dpoids,
// biais -= lr*dbiais. Utilisée par propagation_arriere() ; les variantes
// avec Momentum/Adam passent par maj_reseau_opt() à la place.
void maj_reseau(neural_network* reseau){
    int L = reseau->nombre_couche;
    for (int l = 0; l < L-1; l++) {
        multiply(reseau->dbiais[l],reseau->vitesse_apprentissage);
        multiply(reseau->dpoids[l],reseau->vitesse_apprentissage);
        diff(reseau->poids[l],reseau->dpoids[l],reseau->poids[l]);
        diff(reseau->biais[l],reseau->dbiais[l],reseau->biais[l]);
    }

}

// Remet à zéro les gradients (dbiais, dneuronnes, dpoids) avant de calculer
// une nouvelle rétropropagation. Appelé en tout début de calcul_grad().
void reset_nn(neural_network* res){
    for (int i = 0; i < res->nombre_couche-1; i++)
    {
        cudaMemset(res->dbiais[i]->data,0,sizeof(float)*res->dbiais[i]->lignes*res->dbiais[i]->colonnes);
        cudaMemset(res->dneuronnes[i]->data,0,sizeof(float)*res->dneuronnes[i]->lignes*res->dneuronnes[i]->colonnes);
        cudaMemset(res->dpoids[i]->data,0,sizeof(float)*res->dpoids[i]->lignes*res->dpoids[i]->colonnes);

    }
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("Erreur CUDA lors de la synchronisation dans reset_nn: %s\n", cudaGetErrorString(err));
    }

}

// Arguments passés au thread lancé par calcul_grad() pour chaque couche `l`.
struct structparbackprop_t{
    neural_network* reseau;
    int l;
};
typedef struct structparbackprop_t structparbackprop;


// Calcule transpose(poids[l]) * dneuronnes[l+1] dans un thread séparé,
// pendant que le thread principal calcule la dérivée de l'activation de la
// couche l (voir calcul_grad) : les deux résultats sont ensuite combinés
// par un produit de Hadamard pour obtenir dneuronnes[l].
void* bp_tmp_aux(void* res){
    structparbackprop* v = (structparbackprop*)res;
    int i = v->l;
    matrice* tr = transpose(v->reseau->poids[i]);
    matrice* tmp = zeros(tr->lignes,v->reseau->dneuronnes[i+1]->colonnes);
    dot_par(tr,v->reseau->dneuronnes[i+1],tmp);
    free_mat(tr);
    pthread_exit(tmp);
}


// Calcule les gradients (dpoids, dbiais, dneuronnes) de tout le réseau par
// rétropropagation, à partir de la sortie attendue `obj`. Suppose qu'une
// propagation_avant() a déjà été faite (neuronnes_somme/activ à jour).
//
// Principe : delta_L = dCOST(sortie, obj) .* activation'(somme_L) pour la
// dernière couche, puis pour chaque couche l < L en partant de la fin :
// delta_l = (poids_l^T * delta_{l+1}) .* activation'(somme_l).
// Le gradient des poids de la couche l est enfin delta_{l+1} * activ_l^T,
// et celui du biais est simplement delta_{l+1}.
void calcul_grad(neural_network* reseau,matrice* obj){
    int L = reseau->nombre_couche;
    reset_nn(reseau);

    // delta de la dernière couche : dCOST(sortie, obj) .* activation'(somme).
    matrice* tmp1 = zeros(reseau->neuronnes_activ[L-1]->lignes,reseau->neuronnes_activ[L-1]->colonnes);
    matrice* tmp2 = zeros(reseau->neuronnes_activ[L-1]->lignes,reseau->neuronnes_activ[L-1]->colonnes);
    copy(reseau->neuronnes_somme[L-1],tmp2);
    dCOST(reseau->neuronnes_activ[L-1],obj,tmp1);


    switch (OUTPUTLAYER)
        {
            case 1:
                mat_RELU_d(reseau->neuronnes_somme[L-1],tmp2);
                break;
            case 2:
                mat_sigmoid_d(reseau->neuronnes_somme[L-1],tmp2);
                break;
            case 3:
                mat_SOFT_MAX_d(reseau->neuronnes_somme[L-1],tmp2);
                break;
            default:
                printf("OUTPUT pas defini backprop\n");
                exit(EXIT_FAILURE);
                break;
        }
    hadamar(tmp1,tmp2,reseau->dneuronnes[L-1]);
    free_mat(tmp1);
    free_mat(tmp2);
    
    // Propage le delta vers les couches cachées, de la fin vers le début.
    // Pour chaque couche i, le produit matriciel poids[i]^T * delta[i+1] est
    // calculé dans un thread séparé (bp_tmp_aux) pendant que le thread
    // principal calcule activation'(somme[i]) ; les deux sont ensuite
    // combinés par un produit de Hadamard.
    matrice* tmp;
    void* r_tmp;
    pthread_t th_tmp;
    structparbackprop tmp_struct;
    tmp_struct.reseau = reseau;
    for (int i = L-2; i > 0; i--){
        tmp_struct.l = i;

        pthread_create(&th_tmp,NULL,bp_tmp_aux,&tmp_struct);

        matrice* derivs = zeros(reseau->neuronnes_parcouche[i],1);

        switch (MIDLAYER)
        {
            case 1:
                mat_RELU_d(reseau->neuronnes_somme[i],derivs);
                break;
            case 2:
                mat_sigmoid_d(reseau->neuronnes_somme[i],derivs);
                break;
            case 3:
                mat_SOFT_MAX_d(reseau->neuronnes_somme[i],derivs);
                break;

            default:
                printf("MIDLAYER pas defini backprop\n");
                exit(EXIT_FAILURE);
                break;
        }

        pthread_join(th_tmp,&r_tmp);
        tmp = (matrice*)r_tmp;
        hadamar(tmp,derivs,reseau->dneuronnes[i]);
        
        free_mat(tmp);
        free_mat(derivs);
    }
    
    // Gradient des poids/biais à partir des deltas : dpoids[l-1] =
    // dneuronnes[l] * activ[l-1]^T, dbiais[l-1] = dneuronnes[l].
    for(int l=1;l<L;l++){
        matrice* tr_a = transpose(reseau->neuronnes_activ[l - 1]);
        dot_par(reseau->dneuronnes[l], tr_a, reseau->dpoids[l - 1]);
        copy(reseau->dneuronnes[l], reseau->dbiais[l - 1]);
        free_mat(tr_a);
    }
}

// Rétropropagation "simple" : calcule le gradient (calcul_grad) puis met à
// jour les poids/biais par SGD classique (maj_reseau), sans optimiseur.
void propagation_arriere(neural_network* reseau,matrice* obj){
    calcul_grad(reseau,obj);
    maj_reseau(reseau);
}

// Renvoie l'indice du neurone de sortie le plus activé (classe prédite) et
// sa valeur d'activation. Copie d'abord la dernière couche du GPU vers le
// CPU pour en chercher le maximum.
result obtenir_resultat(neural_network* reseau){
    float max = -1;
    int indice =0;
    int L = reseau->nombre_couche-1;
    int N = reseau->neuronnes_activ[L]->colonnes * reseau->neuronnes_activ[L]->lignes;
    float* host_d = (float*)malloc(sizeof(float) * N);
    cudaError_t err = cudaMemcpy(host_d,reseau->neuronnes_activ[L]->data,sizeof(float)*N,cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Erreur CUDA lors du transfert mémoire : %s\n", cudaGetErrorString(err));
        free(host_d); // Libération de la mémoire allouée
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N; i++)
    {
        float d =host_d[i]; 

        if(d>max){
            max = d;
            indice = i;
        }
    }
    result res;
    res.indice = indice;
    res.valeur = max;
    free(host_d);
    return res;
}

// Sauvegarde l'architecture et les poids/biais de `reseau` dans un fichier
// texte "maison" lisible par importer(). Ne sauvegarde pas les gradients
// (dpoids/dbiais/dneuronnes), recalculés à la prochaine rétropropagation.
void save_neural_network(neural_network* reseau,char* filename){
    //il ne sert a rien de sauvegarder le matrices de derivée partielle
    //le format est du type L \n n1 n2 n3 ...
    //les poids et biais de chaque couche
    //la vitesse d'apprentissage
    FILE* file = fopen(filename,"w");

    if(file==NULL){
        exit(EXIT_FAILURE);
    }
    int L= reseau->nombre_couche;
    fprintf(file,"%d\n",L);
    for (int i = 0; i < L; i++)
    {
        fprintf(file,"%d ",reseau->neuronnes_parcouche[i]);
    }
    fprintf(file,"\n");
    for (int i = 0; i < L-1; i++)
    {
        int Nw= (reseau->poids[i]->colonnes)*(reseau->poids[i]->lignes);
        int Nb = (reseau->biais[i]->colonnes)*(reseau->biais[i]->lignes);
        float* tmpw = (float*)malloc(sizeof(float)*Nw);
        float* tmpb = (float*)malloc(sizeof(float)*Nb);

        cudaMemcpy(tmpw,reseau->poids[i]->data,
            sizeof(float)*Nw
            ,cudaMemcpyDeviceToHost);

        cudaMemcpy(tmpb,reseau->biais[i]->data,
            sizeof(float)*Nb
            ,cudaMemcpyDeviceToHost);
        for (int j = 0; j < Nw; j++)
        {
            fprintf(file,"%f ",tmpw[j]);
        }
        fprintf(file,"\n");
        for (int j = 0; j < Nb; j++)
        {
            fprintf(file,"%f ",tmpb[j]);
        }
        fprintf(file,"\n");
        free(tmpb);
        free(tmpw);
    }
    fprintf(file,"%f",reseau->vitesse_apprentissage);
    fclose(file);
    return;
}

// Recharge un réseau depuis un fichier écrit par save_neural_network().
// (Fonction volontairement très défensive — vérifie chaque allocation et
// chaque lecture — générée avec l'aide d'une IA.)
neural_network* importer(char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Creation d'un nouveau fichier neuralnetwork\n");
        return NULL;
    }

    int L;
    if (fscanf(file, "%d", &L) != 1) {
        perror("Failed to read number of layers");
        fclose(file);
        return NULL;
    }

    neural_network* res = (neural_network*)malloc(sizeof(neural_network));
    if (res == NULL) {
        perror("Failed to allocate memory for neural network");
        fclose(file);
        return NULL;
    }

    res->nombre_couche = L;
    res->neuronnes_parcouche = (int*)malloc(sizeof(int) * L);
    if (res->neuronnes_parcouche == NULL) {
        perror("Failed to allocate memory for neuronnes_parcouche");
        free(res);
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < L; i++) {
        if (fscanf(file, "%d", &(res->neuronnes_parcouche[i])) != 1) {
            perror("Failed to read neuronnes_parcouche");
            free(res->neuronnes_parcouche);
            free(res);
            fclose(file);
            return NULL;
        }
    }

    res->neuronnes_somme = (matrice**)malloc(sizeof(matrice*) * L);
    res->neuronnes_activ = (matrice**)malloc(sizeof(matrice*) * L);
    res->poids = (matrice**)malloc(sizeof(matrice*) * (L - 1));
    res->biais = (matrice**)malloc(sizeof(matrice*) * (L - 1));
    res->dpoids = (matrice**)malloc(sizeof(matrice*) * (L - 1));
    res->dbiais = (matrice**)malloc(sizeof(matrice*) * (L - 1));
    res->dneuronnes = (matrice**)malloc(sizeof(matrice*) * L);

    if (res->neuronnes_somme == NULL || res->neuronnes_activ == NULL ||
        res->poids == NULL || res->biais == NULL ||
        res->dpoids == NULL || res->dbiais == NULL ||
        res->dneuronnes == NULL) {
        perror("Failed to allocate memory for matrices");
        free(res->neuronnes_parcouche);
        free(res);
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < L - 1; i++) {
        int couche = res->neuronnes_parcouche[i];
        int prochaine = res->neuronnes_parcouche[i + 1];

        if (i != 0) {
            res->neuronnes_somme[i] = zeros(couche, 1);
            res->neuronnes_activ[i] = zeros(couche, 1);
        }

        float* tmpw = (float*)malloc(sizeof(float) * prochaine * couche);
        float* tmpb = (float*)malloc(sizeof(float) * prochaine);

        if (tmpw == NULL || tmpb == NULL) {
            perror("Failed to allocate memory for tmpw or tmpb");
            free(res->neuronnes_parcouche);
            free(res);
            fclose(file);
            return NULL;
        }

        matrice* p = (matrice*)malloc(sizeof(matrice));
        matrice* b = (matrice*)malloc(sizeof(matrice));

        if (p == NULL || b == NULL) {
            perror("Failed to allocate memory for poids or biais");
            free(tmpw);
            free(tmpb);
            free(res->neuronnes_parcouche);
            free(res);
            fclose(file);
            return NULL;
        }

        p->lignes = prochaine;
        p->colonnes = couche;
        b->lignes = prochaine;
        b->colonnes = 1;

        for (int j = 0; j < prochaine * couche; j++) {
            if (fscanf(file, "%f", &(tmpw[j])) != 1) {
                perror("Failed to read poids");
                free(tmpw);
                free(tmpb);
                free(p);
                free(b);
                free(res->neuronnes_parcouche);
                free(res);
                fclose(file);
                return NULL;
            }
        }

        for (int j = 0; j < prochaine; j++) {
            if (fscanf(file, "%f", &(tmpb[j])) != 1) {
                perror("Failed to read biais");
                free(tmpw);
                free(tmpb);
                free(p);
                free(b);
                free(res->neuronnes_parcouche);
                free(res);
                fclose(file);
                return NULL;
            }
        }

        cudaError_t errw = cudaMalloc(&(p->data), sizeof(float) * prochaine * couche);
        cudaError_t errb = cudaMalloc(&(b->data), sizeof(float) * prochaine);

        if (errw != cudaSuccess || errb != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(errw));
            free(tmpw);
            free(tmpb);
            free(p);
            free(b);
            free(res->neuronnes_parcouche);
            free(res);
            fclose(file);
            return NULL;
        }

        errw = cudaMemcpy(p->data, tmpw, sizeof(float) * prochaine * couche, cudaMemcpyHostToDevice);
        errb = cudaMemcpy(b->data, tmpb, sizeof(float) * prochaine, cudaMemcpyHostToDevice);

        if (errw != cudaSuccess || errb != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(errw));
            cudaFree(p->data);
            cudaFree(b->data);
            free(tmpw);
            free(tmpb);
            free(p);
            free(b);
            free(res->neuronnes_parcouche);
            free(res);
            fclose(file);
            return NULL;
        }

        res->poids[i] = p;
        res->biais[i] = b;
        res->dpoids[i] = zeros(prochaine, couche);
        res->dbiais[i] = zeros(prochaine, 1);
        res->dneuronnes[i] = zeros(couche, 1);

        free(tmpw);
        free(tmpb);
    }

    int couche = res->neuronnes_parcouche[L - 1];
    res->neuronnes_somme[L - 1] = zeros(couche, 1);
    res->neuronnes_activ[L - 1] = zeros(couche, 1);
    res->dneuronnes[L - 1] = zeros(couche, 1);

    if (fscanf(file, "%f", &(res->vitesse_apprentissage)) != 1) {
        perror("Failed to read vitesse_apprentissage");
        free(res->neuronnes_parcouche);
        free(res);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return res;
}


// Alloue un optimiseur de type `type` (Rien/Momentum/Adam) et initialise à
// zéro ses matrices auxiliaires (moments) aux dimensions des poids/biais de
// `reseau`. Beta1/Beta2 sont les coefficients de décroissance des moments
// (utilisés par Momentum et/ou Adam selon `type`).
optimizer* creer_optimizer(int type,neural_network* reseau,float Beta1,float Beta2){
    optimizer* res = (optimizer*)malloc(sizeof(optimizer));
    res->Beta1 = Beta1;
    res->Beta2 = Beta2;
    res->Beta1t=1;
    res->Beta2t=1;
    res->epsilon = 1e-8f;
    res->type = type;
    int L = reseau->nombre_couche;

    switch (type)
    {
        case Rien:
        
            break;

        case Momentum:
            res->Vcw=NULL;
            res->Vcb=NULL;
            res->Mcw=NULL;
            res->Mcb=NULL;
            res->Mw=NULL;
            res->Mb=NULL;
            res->Vw = (matrice**)malloc(sizeof(matrice*)*L);
            res->Vb = (matrice**)malloc(sizeof(matrice*)*L);
            
            for(int i=0; i< L-1;i++){
                res->Vw[i] = zeros(reseau->poids[i]->lignes,reseau->poids[i]->colonnes);
                res->Vb[i] = zeros(reseau->biais[i]->lignes,reseau->biais[i]->colonnes);
            }
            break;


        case Adam:
            res->Vcw=(matrice**)malloc(sizeof(matrice*)*L);
            res->Vcb=(matrice**)malloc(sizeof(matrice*)*L);
            res->Mcw=(matrice**)malloc(sizeof(matrice*)*L);
            res->Mcb=(matrice**)malloc(sizeof(matrice*)*L);
            res->Mw=(matrice**)malloc(sizeof(matrice*)*L);
            res->Mb=(matrice**)malloc(sizeof(matrice*)*L);
            res->Vw = (matrice**)malloc(sizeof(matrice*)*L);
            res->Vb = (matrice**)malloc(sizeof(matrice*)*L);
            
            for(int i=0; i< L-1;i++){
                res->Vw[i] = zeros(reseau->poids[i]->lignes,reseau->poids[i]->colonnes);
                res->Vcw[i]= zeros(reseau->poids[i]->lignes,reseau->poids[i]->colonnes);
                res->Mcw[i]= zeros(reseau->poids[i]->lignes,reseau->poids[i]->colonnes);
                res->Mw[i]= zeros(reseau->poids[i]->lignes,reseau->poids[i]->colonnes);

                res->Vb[i] = zeros(reseau->biais[i]->lignes,reseau->biais[i]->colonnes);
                res->Vcb[i] = zeros(reseau->biais[i]->lignes,reseau->biais[i]->colonnes);
                res->Mb[i] = zeros(reseau->biais[i]->lignes,reseau->biais[i]->colonnes);
                res->Mcb[i] = zeros(reseau->biais[i]->lignes,reseau->biais[i]->colonnes);
            }

            break;
        
        default:
            exit(EXIT_FAILURE);
            break;
    }
    return res;
}

// Met à jour les poids/biais de chaque couche avec l'optimiseur `opt`, à
// partir des gradients dpoids/dbiais déjà calculés par calcul_grad().
void maj_reseau_opt(optimizer* opt,neural_network* reseau){
    int L = reseau->nombre_couche;
    for (int l = 0; l < L-1; l++) {
        
        switch (opt->type)
        {
        case Momentum:
            
            //poids
            update_momentum_velocity(opt->Vw[l],reseau->dpoids[l],opt->Beta1);
            diff_avec_constante(reseau->poids[l],opt->Vw[l],reseau->poids[l],reseau->vitesse_apprentissage);
            
            //biais
            update_momentum_velocity(opt->Vb[l],reseau->dbiais[l],opt->Beta1);
            diff_avec_constante(reseau->biais[l],opt->Vb[l],reseau->biais[l],reseau->vitesse_apprentissage);
            
            break;
        
        case Adam:
            
            apply_adam(opt,reseau->poids[l],reseau->biais[l],l,reseau->dpoids[l],reseau->dbiais[l],reseau->vitesse_apprentissage);
            
            opt->iteration++;

            break;
        default:
            break;
        }
        


    }
}

// Rétropropagation avec optimiseur : calcule le gradient (calcul_grad) puis
// met à jour le réseau via `opt` (Momentum/Adam) au lieu du SGD classique.
void propagation_arriere_opt(optimizer* opt, neural_network* reseau,matrice* obj){
    calcul_grad(reseau,obj);
    maj_reseau_opt(opt,reseau);
}

// Arguments passés à chaque thread de batch_training_aux().
typedef struct batch_t{
    pthread_mutex_t* mutex_pile;
    pthread_mutex_t* mutex_poids;
    pile* p;
    
    matrice* nourriture;
    matrice* obj;
    matrice** dpoids;
    matrice** dbiais;
}  batch_t;

// Limite le nombre de threads de batch actifs en même temps à THREAD_MAX
// (voir batch_training) : évite de dépiler plus de copies de réseau que la
// pile n'en contient.
sem_t semaphore;

// Corps d'un thread de batch : dépile une copie du réseau (protégée par
// mutex_pile), calcule son gradient sur un exemple, accumule ce gradient
// dans tmp->dpoids/dbiais (protégé par mutex_poids), puis rempile la copie.
void* batch_training_aux(void* res){
    batch_t* tmp = (batch_t*)res;
    neural_network* reseau;

    sem_wait(&semaphore);

    pthread_mutex_lock(tmp->mutex_pile);
        reseau = pop(&(tmp->p));
    pthread_mutex_unlock(tmp->mutex_pile);

    propagation_avant(reseau,tmp->nourriture);
    assert(tmp->obj!=NULL);
    calcul_grad(reseau,tmp->obj);

    pthread_mutex_lock(tmp->mutex_poids);
    for (int i = 0; i < reseau->nombre_couche-1; i++)
    {
        sum(tmp->dpoids[i],reseau->dpoids[i],tmp->dpoids[i]);
        sum(tmp->dbiais[i],reseau->dbiais[i],tmp->dbiais[i]);
    }
    pthread_mutex_unlock(tmp->mutex_poids);
    
    pthread_mutex_lock(tmp->mutex_pile);
        empiler(&(tmp->p),reseau);
    pthread_mutex_unlock(tmp->mutex_pile);

    sem_post(&semaphore);

    
    return NULL;
}




// Entraîne un batch de `batch_size` exemples en parallèle (un thread par
// exemple, via batch_training_aux) : chaque thread accumule son gradient
// dans reseau->dpoids/dbiais, puis on divise par batch_size pour obtenir la
// moyenne du batch.
// ATTENTION (bug existant, non corrigé ici) : `1/batch_size` est une
// division entière, donc égale à 0 dès que batch_size > 1 — le gradient
// moyen du batch est alors annulé au lieu d'être moyenné. Utiliser
// `1.0f/batch_size` corrigerait ce calcul sans changer le reste de la logique.
void batch_training(int debut,int batch_size,data* nourriture,matrice** obj, int N,neural_network* reseau,pile* p){
    pthread_t threads[batch_size];
    pthread_mutex_t mutex_pile;
    pthread_mutex_t mutex_poids;

    pthread_mutex_init(&mutex_pile, NULL);   // ✅ initialise les mutex
    pthread_mutex_init(&mutex_poids, NULL);
    sem_init(&semaphore, PTHREAD_PROCESS_SHARED, THREAD_MAX);

    for(int i = debut; i<debut+batch_size && i<N; i++){
        batch_t* tmp = (batch_t*)malloc(sizeof(batch_t));
        tmp->mutex_poids = &mutex_poids;
        tmp->mutex_pile = &mutex_pile;
        tmp->nourriture= nourriture[i].data;
        tmp->obj = obj[nourriture[i].label];

        tmp->dpoids = reseau->dpoids;
        tmp->dbiais = reseau->dbiais;
        tmp->p = p;
        pthread_create(&threads[i-debut], NULL, batch_training_aux, tmp);
    }

    for(int i = debut; i<debut+batch_size && i<N; i++){
        pthread_join(threads[i-debut], NULL);
    }
    for (int i = 0; i < reseau->nombre_couche-1; i++){
        multiply(reseau->dpoids[i],1/batch_size);
        multiply(reseau->dbiais[i],1/batch_size);
    }
        pthread_mutex_destroy(&mutex_pile);     // ✅ destruction des mutex
    pthread_mutex_destroy(&mutex_poids);
        sem_destroy(&semaphore);
    printf("fin batch\n");
}

// Duplique entièrement un réseau (poids, biais, gradients, activations...)
// par copie profonde de chaque matrice (copy_new). Utilisé pour donner à
// chaque thread de batch_training sa propre copie indépendante du réseau.
neural_network* copy_neural_network(neural_network* reseau) {
    if (reseau == NULL) {
        fprintf(stderr, "Erreur: réseau source est NULL\n");
        return NULL;
    }

    neural_network* res = (neural_network*)malloc(sizeof(neural_network));
    if (res == NULL) {
        fprintf(stderr, "Erreur d'allocation pour la structure réseau\n");
        return NULL;
    }

    // Copie des membres simples
    res->nombre_couche = reseau->nombre_couche;
    res->vitesse_apprentissage = reseau->vitesse_apprentissage;

    // Allocation des tableaux
    res->neuronnes_parcouche = (int*)malloc(sizeof(int) * res->nombre_couche);
    res->neuronnes_somme = (matrice**)malloc(sizeof(matrice*) * res->nombre_couche);
    res->neuronnes_activ = (matrice**)malloc(sizeof(matrice*) * res->nombre_couche);
    res->poids = (matrice**)malloc(sizeof(matrice*) * res->nombre_couche);
    res->biais = (matrice**)malloc(sizeof(matrice*) * res->nombre_couche);
    res->dpoids = (matrice**)malloc(sizeof(matrice*) * res->nombre_couche);
    res->dbiais = (matrice**)malloc(sizeof(matrice*) * res->nombre_couche);
    res->dneuronnes = (matrice**)malloc(sizeof(matrice*) * res->nombre_couche);

    // Vérification des allocations
    if (!res->neuronnes_parcouche || !res->neuronnes_somme || !res->neuronnes_activ || 
        !res->poids || !res->biais || !res->dpoids || !res->dbiais || !res->dneuronnes) {
        fprintf(stderr, "Erreur d'allocation mémoire pour les tableaux\n");
        return NULL;
    }

    // Copie profonde couche par couche
    for (int i = 0; i < res->nombre_couche; i++) {
        res->neuronnes_parcouche[i] = reseau->neuronnes_parcouche[i];

        // Copie des matrices
        if(i>0){
            res->neuronnes_activ[i] = copy_new(reseau->neuronnes_activ[i]);
            res->neuronnes_somme[i] = copy_new(reseau->neuronnes_somme[i]);
        }
        res->dneuronnes[i] = copy_new(reseau->dneuronnes[i]);


        // Pour la dernière couche, on ne copie pas poids/biais (si c'est bien le cas)
        if (i < res->nombre_couche - 1) {

            res->poids[i] = copy_new(reseau->poids[i]);
            res->biais[i] = copy_new(reseau->biais[i]);
            res->dpoids[i] = copy_new(reseau->dpoids[i]);
            res->dbiais[i] = copy_new(reseau->dbiais[i]);
        }

    }

    return res;
}

// Libère toute la mémoire (CPU et GPU) associée au réseau : matrices
// poids/biais/gradients/activations puis la structure elle-même.
void liberer_reseau(neural_network* reseau){
    free(reseau->neuronnes_parcouche);
    for (int i = 0; i < reseau->nombre_couche; i++) {

        // Copie des matrices
        if(i>0){
            free_mat(reseau->neuronnes_activ[i]);
            free_mat(reseau->neuronnes_somme[i]);
        }
        free_mat(reseau->dneuronnes[i]);


        // Pour la dernière couche, on ne copie pas poids/biais (si c'est bien le cas)
        if (i < reseau->nombre_couche - 1) {

            free_mat(reseau->poids[i]);
            free_mat(reseau->biais[i]);
            free_mat(reseau->dpoids[i]);
            free_mat(reseau->dbiais[i]);
        }
    }
    free(reseau);
}