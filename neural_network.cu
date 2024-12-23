#include"neural_network.h"
#include<stdio.h>
#include"matrice.h"


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
    
        if(i!=0){
            res->neuronnes_somme[i] = zeros(couche,1);
            res->neuronnes_activ[i] = zeros(couche,1);
        }

    
        res->poids[i] = random_mat(prochaine,couche,10);
        res->biais[i] = random_mat(prochaine,1,10);
    
        res->dpoids[i] = zeros(prochaine,couche);
        res->dbiais[i] = zeros(prochaine,1);
    
    }

    int couche = res->neuronnes_parcouche[nb_couche-1];
    res->neuronnes_somme[nb_couche-1] = zeros(couche,1);
    res->neuronnes_activ[nb_couche-1] = zeros(couche,1);


    return res;
}

void propagation_avant(neural_network* reseau,matrice* nourriture){
    int L = reseau->nombre_couche;

    reseau->neuronnes_activ[0] = nourriture;


    for (int i = 0; i < L-2; i++)
    {
        dot_par(reseau->neuronnes_activ[i],reseau->poids[i],reseau->neuronnes_somme[i+1]);
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

    dot_par(reseau->neuronnes_activ[L-2],reseau->poids[L-2],reseau->neuronnes_somme[L-1]);
    sum(reseau->neuronnes_somme[L-1],reseau->biais[L-2],reseau->neuronnes_somme[L-1]);         
    switch (OUTPUTLAYER)
    {
        case 1:
            mat_RELU(reseau->neuronnes_somme[L-1],reseau->neuronnes_activ[L-1]);
            break;
        case 2:
            mat_sigmoid(reseau->neuronnes_somme[L-1],reseau->neuronnes_activ[L-1]);
            
            break;
        case 3:
            mat_SOFT_MAX(reseau->neuronnes_somme[L-1],reseau->neuronnes_activ[L-1]);
            break;

        default:
            printf("OUTPUTLAYER pas defini\n");
            exit(EXIT_FAILURE);
            break;
    }
}

void maj_reseau(neural_network* reseau){

}

void propagation_arriere(neural_network* reseau,matrice* obj){
    int L = reseau->nombre_couche;


    matrice* tmp1 = zeros(reseau->neuronnes_activ[L-1]->lignes,reseau->neuronnes_activ[L-1]->colonnes);
    matrice* tmp2 = zeros(reseau->neuronnes_activ[L-1]->lignes,reseau->neuronnes_activ[L-1]->colonnes);
    
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

    for (int i = L-2; i >= 0; i++)
    {
        /* code */
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
    
    for(int l=1;l<=L;l++){
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
    memccpy(host_d,reseau->neuronnes_activ[L]->data,sizeof(double)*N,cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
    {
        double d =host_d[i]; 

        if(d>max){
            max = d;
            indice = i;
        }
    }
    result res ;
    res.indice = indice;
    res.valeur = max;
    free(host_d);
    return res;
}

