#include"neural_network.h"
#include<stdio.h>
#include"matrice.h"


neural_network* cree_reseau(int nb_couche, ...){
    va_list ap;

    neural_network* res = (neural_network*)malloc(sizeof(neural_network));
    res->nombre_couche=nb_couche;
    res->vitesse_apprentissage = 0.1;
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
    
        
        res->neuronnes_somme[i] = zeros(couche,1);
        res->neuronnes_activ[i] = zeros(couche,1);
        

    
        res->poids[i] = random_mat(prochaine,couche,0.5);
        res->biais[i] = random_mat(prochaine,1,0.5);
    
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

void propagation_avant(neural_network* reseau,matrice* nourriture){
    int L = reseau->nombre_couche;

    reseau->neuronnes_activ[0] = nourriture;


    for (int i = 0; i < L-1; i++)
    {   
        dot_par(reseau->poids[i],reseau->neuronnes_activ[i],reseau->neuronnes_somme[i+1]);
        sum(reseau->neuronnes_somme[i+1],reseau->biais[i],reseau->neuronnes_somme[i+1]);         
        switch (MIDLAYER)
        {
            case 1:
                mat_RELU(reseau->neuronnes_somme[i+1],reseau->neuronnes_activ[i+1]);
                break;
            case 2:
                mat_sigmoid(reseau->neuronnes_somme[i+1],reseau->neuronnes_activ[i+1]);
                
                break;
            case 3:
                mat_SOFT_MAX(reseau->neuronnes_somme[i+1],reseau->neuronnes_activ[i+1]);
                break;

            default:
                printf("MIDLAYER pas defini\n");
                exit(EXIT_FAILURE);
                break;
        }
    }

}

void maj_reseau(neural_network* reseau){
    for(int l=0;l < reseau->nombre_couche-1;l++){
        multiply(reseau->dpoids[l],reseau->vitesse_apprentissage);
        multiply(reseau->dbiais[l],reseau->vitesse_apprentissage);

        diff(reseau->biais[l],reseau->dbiais[l],reseau->biais[l]);
        diff(reseau->poids[l],reseau->dpoids[l],reseau->poids[l]);
    }
}
void reset_nn(neural_network* res){
    for (int i = 0; i < res->nombre_couche-1; i++)
    {
        cudaMemset(res->dbiais[i]->data,0,sizeof(double)*res->dbiais[i]->lignes*res->dbiais[i]->colonnes);
        cudaMemset(res->dneuronnes[i]->data,0,sizeof(double)*res->dneuronnes[i]->lignes*res->dneuronnes[i]->colonnes);
        cudaMemset(res->dpoids[i]->data,0,sizeof(double)*res->dpoids[i]->lignes*res->dpoids[i]->colonnes);
        
    }
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("aaaaaa");
    }
    
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
    print_mat(tmp2);
    print_mat(reseau->neuronnes_somme[L-1]);
    
    free_mat(tmp1);
    free_mat(tmp2);


    for (int i = L-2; i >= 0; i--){
        
        matrice* tr = transpose(reseau->poids[i]);
        matrice* tmp = zeros(tr->lignes,reseau->dneuronnes[i+1]->colonnes);
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


        dot_par(tr,reseau->dneuronnes[i+1],tmp);
        hadamar(tmp,derivs,reseau->dneuronnes[i]);
        
        free_mat(tr);
        free_mat(tmp);
        free_mat(derivs);
    }
    
    for(int l=1;l<L;l++){
        matrice* tr_a = transpose(reseau->neuronnes_activ[l-1]);
        dot_par(reseau->dneuronnes[l],tr_a,reseau->dpoids[l-1]);
        copy(reseau->dneuronnes[l],reseau->dbiais[l-1]);
        free_mat(tr_a);
    }
    maj_reseau(reseau);
}

result obtenir_resultat(neural_network* reseau){
    double max = -1;
    int indice =0;
    int L = reseau->nombre_couche-1;
    int N = reseau->neuronnes_activ[L]->colonnes * reseau->neuronnes_activ[L]->lignes;
    double* host_d = (double*)malloc(sizeof(double) * N);
    cudaError_t err = cudaMemcpy(host_d,reseau->neuronnes_activ[L]->data,sizeof(double)*N,cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Erreur CUDA lors du transfert mémoire : %s\n", cudaGetErrorString(err));
        free(host_d); // Libération de la mémoire allouée
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N; i++)
    {
        double d =host_d[i]; 

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

