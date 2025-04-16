#include"neural_network.h"
#include<stdio.h>
#include"matrice.h"
#include <pthread.h>
#include"kernel.h"
typedef struct nn_thread{
    int i;
    matrice* mat;
    neural_network* reseau;
} nn_thread;





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

void propagation_avant(neural_network* reseau, matrice* nourriture) {
    // if (reseau == NULL || nourriture == NULL) {
    //     printf("Erreur : le réseau ou la matrice d'entrée est NULL.\n");
    //     exit(EXIT_FAILURE);
    // }

    int L = reseau->nombre_couche;

    // // Vérification des dimensions de l'entrée
    // if (nourriture->lignes*nourriture->colonnes != reseau->neuronnes_parcouche[0]) {
    //     printf("Erreur : les dimensions de l'entrée (%d) ne correspondent pas à la première couche (%d).\n",
    //            nourriture->lignes, reseau->neuronnes_parcouche[0]);
    //     exit(EXIT_FAILURE);
    // }

    // Initialisation de la première couche
    // copy(nourriture, reseau->neuronnes_somme[0]);
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

void maj_reseau(neural_network* reseau){
    int L = reseau->nombre_couche;
    for (int l = 0; l < L-1; l++) {
        multiply(reseau->dbiais[l],reseau->vitesse_apprentissage);
        multiply(reseau->dpoids[l],reseau->vitesse_apprentissage);
        diff(reseau->poids[l],reseau->dpoids[l],reseau->poids[l]);
        diff(reseau->biais[l],reseau->dbiais[l],reseau->biais[l]);
    }

}

void reset_nn(neural_network* res){
    for (int i = 0; i < res->nombre_couche-1; i++)
    {
        cudaMemset(res->dbiais[i]->data,0,sizeof(float)*res->dbiais[i]->lignes*res->dbiais[i]->colonnes);
        cudaMemset(res->dneuronnes[i]->data,0,sizeof(float)*res->dneuronnes[i]->lignes*res->dneuronnes[i]->colonnes);
        cudaMemset(res->dpoids[i]->data,0,sizeof(float)*res->dpoids[i]->lignes*res->dpoids[i]->colonnes);
        
    }
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("aaaaaa");
    }
    
}

struct structparbackprop_t{
    neural_network* reseau;
    int l;
};
typedef struct structparbackprop_t structparbackprop;


void* bp_tmp_aux(void* res){
    structparbackprop* v = (structparbackprop*)res;
    int i = v->l;
    matrice* tr = transpose(v->reseau->poids[i]);
    matrice* tmp = zeros(tr->lignes,v->reseau->dneuronnes[i+1]->colonnes);
    dot_par(tr,v->reseau->dneuronnes[i+1],tmp);
    free_mat(tr);
    pthread_exit(tmp);
}

void propagation_arriere(neural_network* reseau,matrice* obj){
    int L = reseau->nombre_couche;
    reset_nn(reseau);

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
    
    for(int l=1;l<L;l++){
        matrice* tr_a = transpose(reseau->neuronnes_activ[l - 1]);
        dot_par(reseau->dneuronnes[l], tr_a, reseau->dpoids[l - 1]);
        copy(reseau->dneuronnes[l], reseau->dbiais[l - 1]);
        free_mat(tr_a);
    }
    // print_mat(reseau->dbiais[0]);
    maj_reseau(reseau);
}

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


optimizer* creer_optimizer(int type,neural_network* reseau,float Beta1,float Beta2){
    optimizer* res = (optimizer*)malloc(sizeof(optimizer));
    res->Beta1 = Beta1;
    res->Beta2 = Beta2;
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

            
        default:
            break;
        }
        


    }
}

void propagation_arriere_opt(optimizer* opt, neural_network* reseau,matrice* obj){
    int L = reseau->nombre_couche;
    reset_nn(reseau);

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
    
    for(int l=1;l<L;l++){
        matrice* tr_a = transpose(reseau->neuronnes_activ[l - 1]);
        dot_par(reseau->dneuronnes[l], tr_a, reseau->dpoids[l - 1]);
        copy(reseau->dneuronnes[l], reseau->dbiais[l - 1]);
        free_mat(tr_a);
    }
    maj_reseau_opt(opt,reseau);
}
