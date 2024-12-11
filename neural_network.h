#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include"matrice.h"
struct neural_network
{
    int nombre_couche;
    matrice** neuronnes_somme;
    matrice** neuronnes_activ;

    matrice** poids;
    matrice** biais;

    matrice** dpoids;
    matrice** dneuronnes;

    double vitesse_apprentissage;
};
typedef struct neural_network neural_network;

struct result
{
    int indice;
    double valeur;
};
typedef struct result result;

//TODO mettre les bons arguments
neural_network* cree_reseau();

void propagation_avant(neural_network* reseau,matrice* nourriture);
void propagation_arriere(neural_network* reseau,matrice* obj);

result resultat(neural_network* reseau){
    
}

#endif